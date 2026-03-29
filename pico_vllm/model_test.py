# test_paged_attention.py
import torch
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

def make_paged_cache(max_seq_len=512):
    return PagedKVCache(
        block_manager=bm,
        num_layers=cfg.num_hidden_layers,
        max_seq_len=max_seq_len,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        device=device,
        dtype=dtype,
    )

# ============================================================
# 测试一：RoPE 数值验证
# ============================================================
print("=" * 60)
print("测试一：RoPE cos/sin 查表")
print("=" * 60)

rope = RoPE(cfg.head_dim, cfg.rope_theta)
rope = rope.to(device)

# prefill position_ids
position_ids_prefill = torch.arange(10, dtype=torch.long, device=device).unsqueeze(0)  # (1, 10)
cos_p, sin_p = rope.get_cos_sin(position_ids_prefill)
assert cos_p.shape == (1, 10, cfg.head_dim), f"cos shape 错误: {cos_p.shape}"
assert sin_p.shape == (1, 10, cfg.head_dim)

# decode position_ids（B=3，每个请求不同位置）
position_ids_decode = torch.tensor([[5], [10], [20]], dtype=torch.long, device=device)  # (3, 1)
cos_d, sin_d = rope.get_cos_sin(position_ids_decode)
assert cos_d.shape == (3, 1, cfg.head_dim)

# 验证 cos²+sin²=1
assert torch.allclose(cos_p[0, 0].pow(2) + sin_p[0, 0].pow(2),
                       torch.ones(cfg.head_dim, device=device), atol=1e-3)
print("  RoPE shape 和数值正确 ✓")

# ============================================================
# 测试二：单层 GQAAttention prefill
# ============================================================
print("\n" + "=" * 60)
print("测试二：GQAAttention prefill（单层）")
print("=" * 60)

model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(dtype).to(device)
model.eval()
# 在 model = load_weights(...) 之后加
print(f"模型加载后显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained("./weights")

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
seq_len = input_ids.shape[1]

cache = make_paged_cache()
seq_len = input_ids.shape[1]

# 新接口
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

assert logits.shape == (1, seq_len, cfg.vocab_size)
assert cache.seq_len == seq_len, f"prefill 后 seq_len 应为 {seq_len}，实际 {cache.seq_len}" 
assert cache.allocated_cache_block_num == (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

next_token = logits[0, -1].argmax().item()
next_word = tokenizer.decode([next_token])
print(f"  prompt: {prompt!r}")
print(f"  next token: {next_word!r}")
print(f"  cache.seq_len={cache.seq_len}, allocated_blocks={cache.allocated_cache_block_num} ✓")

# ============================================================
# 测试三：decode 单步
# ============================================================
print("\n" + "=" * 60)
print("测试三：GQAAttention decode 单步")
print("=" * 60)

decode_input = torch.tensor([[next_token]], dtype=torch.long, device=device)  # (1, 1)
cache.prepare_decode_step()
slot = cache.get_decode_slot()
slot_mapping = torch.tensor([slot], dtype=torch.int32, device=device)
position_ids = torch.tensor([[cache.seq_len]], dtype=torch.long, device=device)

block_table = torch.full((1, cfg.MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32, device=device)
bt = cache.get_block_table()
block_table[0, :len(bt)] = bt

context_lens = torch.tensor([cache.seq_len + 1], dtype=torch.int32, device=device)

with torch.no_grad():
    logits2 = model(
        decode_input,
        kv_cache_k=bm.gpu_kv_cache[0],
        kv_cache_v=bm.gpu_kv_cache[1],
        position_ids=position_ids,
        slot_mapping=slot_mapping,
        is_prefill=False,
        block_table=block_table,
        context_lens=context_lens,
    )

cache._seq_len += 1

assert logits2.shape == (1, 1, cfg.vocab_size)
assert cache.seq_len == seq_len + 1, f"decode 后 seq_len 应为 {seq_len+1}，实际 {cache.seq_len}"

next_token2 = logits2[0, -1].argmax().item()
print(f"  decode step 1 next token: {tokenizer.decode([next_token2])!r}")
print(f"  cache.seq_len={cache.seq_len} ✓")

cache.reset()
del cache

# ============================================================
# 测试四：完整生成对比（Paged vs 旧 NaiveKVCache）
# ============================================================
print("\n" + "=" * 60)
print("测试四：完整生成，对比 Paged 和参考输出")
print("=" * 60)
def run_paged_generate(input_ids, cache, max_new=20):
    generated = input_ids[0].tolist()
    seq_len = input_ids.shape[1]

    # prefill
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
    cache._seq_len += seq_len
    next_id = logits[0, -1].argmax().item()
    generated.append(next_id)

    # decode loop
    for step in range(max_new - 1):
        cache.prepare_decode_step()
        slot = cache.get_decode_slot()
        slot_mapping = torch.tensor([slot], dtype=torch.int32, device=device)

        # print(f"=== decode step {step}, _seq_len={cache._seq_len}, slot={slot} ===")
        position_ids = torch.tensor([[cache.seq_len]], dtype=torch.long, device=device)

        block_table = torch.full(
            (1, cfg.MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32, device=device
        )
        bt = cache.get_block_table()
        block_table[0, :len(bt)] = bt
        context_lens = torch.tensor([cache.seq_len + 1], dtype=torch.int32, device=device)

        # print(f"_seq_len={cache._seq_len}")
        # print(f"position_ids={position_ids.tolist()}")
        # print(f"context_lens={context_lens.tolist()}")
        # print(f"slot={slot}")

        dec_input = torch.tensor([[next_id]], device=device)
        with torch.no_grad():
            logits = model(
                dec_input,
                kv_cache_k=bm.gpu_kv_cache[0],
                kv_cache_v=bm.gpu_kv_cache[1],
                position_ids=position_ids,
                slot_mapping=slot_mapping,
                is_prefill=False,
                block_table=block_table,
                context_lens=context_lens,
            )

        # # decode 读取时：打印 gather 会读哪些 slot，对应的值是什么
        # forward 之后验证 decode 新 token 是否写入
        if step == 0:
            # 用完整序列喂给 HF 看 logits 对不对
            ref_ids = torch.tensor([generated], device=device)  # 所有已生成 token
            print(f"generated so far: {tokenizer.decode(generated)!r}")
            print(f"next_id from paged: {tokenizer.decode([next_id])!r}")


        cache._seq_len += 1
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break

    return generated

# 用 HuggingFace 作为 ground truth
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained("./weights", torch_dtype=dtype).to(device)
hf_model.eval()

prompts = [
    "The capital of France is",
    "1 + 1 =",
]
MAX_NEW = 20
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        hf_out = hf_model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False)
    hf_text = tokenizer.decode(hf_out[0])

    cache = make_paged_cache(max_seq_len=512)
    generated = run_paged_generate(input_ids, cache, MAX_NEW)
    paged_text = tokenizer.decode(generated)

    print(f"\n  prompt: {prompt!r}")
    print(f"  HF:    {hf_text!r}")
    print(f"  Paged: {paged_text!r}")
    print(f"  一致: {'✓' if paged_text == hf_text else '✗'}")
    cache.reset()
    del cache

# print("\n" + "=" * 60)
# print("诊断：逐步对比 logits")
# print("=" * 60)
# prompt = "1 + 1 ="
# input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
# cache = make_paged_cache(max_seq_len=512)

# generated_ids = input_ids[0].tolist()

# with torch.no_grad():
#     # prefill
#     paged_logits = model(input_ids, kv_caches=[cache], is_prefill=True)
#     paged_tok = paged_logits[0, -1].argmax().item()
#     generated_ids.append(paged_tok)

#     for step in range(5):
#         # paged decode
#         p_in = torch.tensor([[paged_tok]], device=device)
#         # cache.prepare_decode_step()
#         p_logits = model(p_in, kv_caches=[cache], is_prefill=False)[0, -1]

#         # HF 用完整序列重算（公平对比）
#         full_ids = torch.tensor([generated_ids], device=device)
#         h_logits = hf_model(full_ids).logits[0, -1].float()

#         diff = (p_logits.float() - h_logits).abs()
#         p_tok = p_logits.argmax().item()
#         h_tok = h_logits.argmax().item()

#         print(f"Step {step+1}: 最大误差={diff.max():.4f} "
#               f"paged={tokenizer.decode([p_tok])!r} "
#               f"hf={tokenizer.decode([h_tok])!r} "
#               f"{'✓' if p_tok == h_tok else '✗'}")

#         paged_tok = p_tok
#         generated_ids.append(paged_tok)

# cache.reset()
# ============================================================
# 测试五：batch decode（多请求）
# ============================================================
print("\n" + "=" * 60)
print("测试五：batch decode（2个请求）")
print("=" * 60)

del hf_model
torch.cuda.empty_cache()

prompts_batch = [
    "The capital of France is",
    "1 + 1 =",
]
# prefill（逐个）
caches = [make_paged_cache() for _ in prompts_batch]
first_tokens = []

for prompt, cache in zip(prompts_batch, caches):
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    seq_len = ids.shape[1]
    cache._allocate_for_prefill(seq_len)
    slot_mapping = cache.get_prefill_slot_mapping(seq_len)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            ids,
            kv_cache_k=bm.gpu_kv_cache[0],
            kv_cache_v=bm.gpu_kv_cache[1],
            position_ids=position_ids,
            slot_mapping=slot_mapping,
            is_prefill=True,
        )
    cache._seq_len += seq_len
    first_tokens.append(logits[0, -1].argmax().item())

print(f"  prefill 完成，first_tokens={[tokenizer.decode([t]) for t in first_tokens]}")

# batch decode
B = len(caches)
for c in caches:
    c.prepare_decode_step()

slots = [c.get_decode_slot() for c in caches]
slot_mapping = torch.tensor(slots, dtype=torch.int32, device=device)
position_ids = torch.tensor([[c.seq_len] for c in caches], dtype=torch.long, device=device)

block_table = torch.full((B, cfg.MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32, device=device)
for i, c in enumerate(caches):
    bt = c.get_block_table()
    block_table[i, :len(bt)] = bt

context_lens = torch.tensor([c.seq_len + 1 for c in caches], dtype=torch.int32, device=device)
dec_input = torch.tensor([[t] for t in first_tokens], dtype=torch.long, device=device)

with torch.no_grad():
    logits_batch = model(
        dec_input,
        kv_cache_k=bm.gpu_kv_cache[0],
        kv_cache_v=bm.gpu_kv_cache[1],
        position_ids=position_ids,
        slot_mapping=slot_mapping,
        is_prefill=False,
        block_table=block_table,
        context_lens=context_lens,
    )

for c in caches:
    c._seq_len += 1

assert logits_batch.shape == (B, 1, cfg.vocab_size), f"batch decode 输出 shape 错误: {logits_batch.shape}"

for i in range(B):
    next_tok = logits_batch[i, -1].argmax().item()
    print(f"  请求 {i} decode 结果: {tokenizer.decode([next_tok])!r}, cache.seq_len={caches[i].seq_len}")

# 验证 batch decode 和单独 decode 结果一致
print("\n  验证 batch decode 和单独 decode 结果一致...")
caches_single = [make_paged_cache() for _ in prompts_batch]

with torch.no_grad():
    single_results = []
    for i, (prompt, cache_s) in enumerate(zip(prompts_batch, caches_single)):
        ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        logits = model(ids, kv_caches=[cache_s], is_prefill=True)
        tok = logits[0, -1].argmax().item()

        dec = torch.tensor([[tok]], device=device)
        logits2 = model(dec, kv_caches=[cache_s], is_prefill=False)
        single_results.append(logits2[0, -1].argmax().item())

all_match = all(
    logits_batch[i, -1].argmax().item() == single_results[i]
    for i in range(B)
)
print(f"  batch vs single 一致: {'✓' if all_match else '✗'}")

for c in caches + caches_single:
    c.reset()

print("\n" + "=" * 60)
print("所有测试完成")
print("=" * 60)