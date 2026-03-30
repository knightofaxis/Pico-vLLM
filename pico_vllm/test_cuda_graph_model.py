import torch
import time
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig, RoPE
from cache import PagedKVCache, BlockManager, pagedblocktype
from weights import load_weights

device = 'cuda'
dtype = torch.bfloat16
cfg = ModelConfig()

# ============================================================
# 初始化 BlockManager（全局唯一）
# ============================================================
NUM_GPU_BLOCKS = 200
BLOCK_SIZE = 16

bm = BlockManager(
    num_gpu_blocks=NUM_GPU_BLOCKS,
    num_cpu_blocks=0,
    block_size=BLOCK_SIZE,
    num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads,
    head_dim=cfg.head_dim,
    dtype=dtype,
)

def make_paged_cache(max_seq_len=1024):
    return PagedKVCache(
        block_manager=bm,
        num_layers=cfg.num_hidden_layers,
        max_seq_len=max_seq_len,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        device=device,
        dtype=dtype,
    )
MAX_SEQ_LEN = 1024
MAX_BLOCKS = MAX_SEQ_LEN // BLOCK_SIZE

# ============================================================
# 加载模型
# ============================================================
model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(dtype).to(device)
model.eval()
# 在 model = load_weights(...) 之后加
print(f"模型加载后显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# model.forward_decode = torch.compile(
#     model.forward_decode,
#     mode="default",      # 只做算子融合，不做 CUDA Graph
#     fullgraph=False
# )

tokenizer = AutoTokenizer.from_pretrained("./weights")

# ============================================================
# 创建静态buffer
# ============================================================
B = 1
# Step 1：预分配所有静态 buffer（形状固定，值可变）
static_input_ids    = torch.zeros(B, 1, dtype=torch.long, device=device)
static_slot_mapping = torch.zeros(B, dtype=torch.int32, device=device)
static_position_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
static_block_table  = torch.full((B, MAX_BLOCKS), -1, dtype=torch.int32, device=device)
static_context_lens = torch.zeros(B, dtype=torch.int32, device=device)
static_output       = torch.empty(B, 1, cfg.vocab_size, dtype=dtype, device=device)

# ============================================================
# Prefill
# ============================================================
prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
seq_len = input_ids.shape[1]
cache = make_paged_cache()

cache._allocate_for_prefill(seq_len)
slot_mapping = cache.get_prefill_slot_mapping(seq_len)
position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    logits = model(
        input_ids,
        kv_cache_k=bm.gpu_kv_cache[0],
        kv_cache_v=bm.gpu_kv_cache[1],
        position_ids=position_ids,
        slot_mapping=slot_mapping,
        is_prefill=True,
    )

# 手动更新 seq_len
cache._seq_len += seq_len

next_token = logits[0, -1].argmax().item()
next_word = tokenizer.decode([next_token])
print(next_word, end='')

# ============================================================
# 准备Decode的静态数据
# ============================================================

cache.prepare_decode_step()
static_input_ids[0, 0] = next_token                 # 索引赋值，原地修改
static_slot_mapping[0] = cache.get_decode_slot()    # 同上
static_position_ids[0, 0] = cache.seq_len
static_block_table[0, :cache.allocated_cache_block_num].copy_(cache.get_block_table())
static_context_lens[0] = cache.seq_len + 1

# Step 2：预热（触发 compile 编译，必须在 capture 之前）
for _ in range(3):
    with torch.no_grad():
        _ = model.forward_decode(
            static_input_ids,
            kv_cache_k=bm.gpu_kv_cache[0],
            kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=static_position_ids,
            slot_mapping=static_slot_mapping,
            block_table=static_block_table,
            context_lens=static_context_lens,
        )
torch.cuda.synchronize()  # 确保编译完成
# ============================================================
# 录制CUDA Graph
# ============================================================
# Step 2：capture（只做一次）
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model.forward_decode(
        static_input_ids,
        kv_cache_k=bm.gpu_kv_cache[0],
        kv_cache_v=bm.gpu_kv_cache[1],
        position_ids=static_position_ids,
        slot_mapping=static_slot_mapping,
        block_table=static_block_table,
        context_lens=static_context_lens,
    )

next_token = static_output[0, -1].argmax()
print(tokenizer.decode([next_token.item()]))

# Step 3：每步 decode 只更新 buffer 的值，然后 replay
# ============================================================
# Decode Loop 和 Profiling (Zero CPU-GPU Synchronization)
# ============================================================
PROFILING_TOKENS = 100  # 强制生成的 token 数量

print(f"\n🚀 开始纯异步吞吐量测试 (生成 {PROFILING_TOKENS} tokens)...")

# 1. 极其关键：在计时开始前，强制同步，确保前面所有的显存操作和 Capture 彻底收尾
torch.cuda.synchronize()
start_time = time.perf_counter()

for step in range(PROFILING_TOKENS):
    # 逻辑步进
    cache._seq_len += 1 
    cache.prepare_decode_step()

    # 纯异步更新静态 Buffer (non_blocking=True 和 fill_ 绝不阻塞 CPU)
    static_input_ids.copy_(next_token, non_blocking=True)
    static_slot_mapping.fill_(cache.get_decode_slot())
    static_position_ids.fill_(cache._seq_len)
    static_context_lens.fill_(cache._seq_len + 1)

    bt = cache.get_block_table()
    static_block_table[0, :bt.shape[0]].copy_(bt, non_blocking=True)

    # 一键回放 Graph (纯 CPU 下发，耗时 ~0.01ms)
    g.replay()

    # 找最大概率 token (纯 GPU 操作，Tensor 留在显存，不拉回 CPU)
    next_token = static_output[0, -1].argmax()

    # ⚠️ 绝对不要在这里 print，也不要调 .item() 判断 EOS！
    # CPU 会以极快的速度把这 100 次循环全部塞进 CUDA 队列，然后跑出 for 循环

# ============================================================
# 统计与输出
# ============================================================
# 2. 循环结束，强行拉闸同步，等待 GPU 把队列里这 100 步的活儿全部干完
torch.cuda.synchronize()
end_time = time.perf_counter()

total_time = end_time - start_time
tok_per_sec = PROFILING_TOKENS / total_time
ms_per_tok = (total_time / PROFILING_TOKENS) * 1000

print("-" * 40)
print(f"=== Profiling ===")
print(f"生成数量: {PROFILING_TOKENS} tokens")
print(f"总耗时:   {total_time:.4f} 秒")
print(f"单步延迟: {ms_per_tok:.2f} ms/tok")
print(f"平均吞吐: {tok_per_sec:.2f} tokens/s")
print("-" * 40)