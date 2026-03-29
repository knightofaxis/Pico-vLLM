# profile_nsys.py
import torch
import os
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager

device = 'cuda'
cfg = ModelConfig()
BLOCK_SIZE = 16

model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(torch.bfloat16).to(device)
model.eval()
# model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
tokenizer = AutoTokenizer.from_pretrained("./weights")

bm = BlockManager(
    num_gpu_blocks=200, num_cpu_blocks=0,
    block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads,
    head_dim=cfg.head_dim, dtype=torch.bfloat16,
)
cache = PagedKVCache(
    block_manager=bm, num_layers=cfg.num_hidden_layers,
    max_seq_len=512, num_kv_heads=cfg.num_key_value_heads,
    head_dim=cfg.head_dim, device=device, dtype=torch.bfloat16,
)

prompt_ids = tokenizer.encode(
    "The quick brown fox jumps over the lazy dog. " * 3,
    return_tensors='pt'
).to(device)[:, :64]

seq_len = prompt_ids.shape[1]  # ← 用实际长度，不要硬编码

# prefill
cache._allocate_for_prefill(seq_len)
slot_mapping = cache.get_prefill_slot_mapping(seq_len)
position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    logits = model(
        prompt_ids,
        kv_cache_k=bm.gpu_kv_cache[0],
        kv_cache_v=bm.gpu_kv_cache[1],
        position_ids=position_ids,
        slot_mapping=slot_mapping,
        is_prefill=True,
    )
cache._seq_len += seq_len
last_token = logits[0, -1:].argmax(-1, keepdim=True)

prefill_seq_len = cache._seq_len
prefill_allocated = cache.allocated_cache_block_num

def make_decode_tensors(cache):
    """构造一次 decode 所需的所有 tensor"""
    cache.prepare_decode_step()
    slot = cache.get_decode_slot()
    slot_mapping = torch.tensor([slot], dtype=torch.int32, device=device)
    position_ids = torch.tensor([[cache._seq_len]], dtype=torch.long, device=device)
    bt = cache.get_block_table()
    block_table = torch.full(
        (1, cfg.MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32, device=device
    )
    block_table[0, :len(bt)] = bt
    context_lens = torch.tensor([cache._seq_len + 1], dtype=torch.int32, device=device)
    return slot_mapping, position_ids, block_table, context_lens

def reset_cache():
    cache._seq_len = prefill_seq_len
    cache.allocated_cache_block_num = prefill_allocated

# 预热
for _ in range(5):
    reset_cache()
    sm, pid, bt, cl = make_decode_tensors(cache)
    with torch.no_grad():
        _ = model(
            last_token,
            kv_cache_k=bm.gpu_kv_cache[0],
            kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=pid,
            slot_mapping=sm,
            is_prefill=False,
            block_table=bt,
            context_lens=cl,
        )
torch.cuda.synchronize()

# nsys 捕获
torch.cuda.cudart().cudaProfilerStart()

for _ in range(20):
    reset_cache()
    sm, pid, bt, cl = make_decode_tensors(cache)
    with torch.no_grad():
        _ = model(
            last_token,
            kv_cache_k=bm.gpu_kv_cache[0],
            kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=pid,
            slot_mapping=sm,
            is_prefill=False,
            block_table=bt,
            context_lens=cl,
        )

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()

print("Done. Run with: nsys profile -c cudaProfilerApi python profile_nsys.py")

cache.reset()