# test_engine.py
import torch
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager
from engine import Engine
from scheduler import Scheduler

device = 'cuda'
dtype = torch.bfloat16
cfg = ModelConfig()

# ============================================================
# 初始化
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

model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(dtype).to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./weights")

engine = Engine(
    model=model,
    tokenizer=tokenizer,
    block_manager=bm,
    cache_cls=PagedKVCache,
    use_cuda_graph=True,
    max_batch_size=4,
    device=device,
)

print(f"Engine 初始化完成，CUDA Graph: {engine.use_cuda_graph}")
print(f"显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB / "
      f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# ============================================================
# 测试一：单请求生成，验证正确性
# ============================================================
print("\n" + "=" * 60)
print("测试一：单请求生成，对比 HuggingFace")
print("=" * 60)

from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained("./weights", dtype=dtype).to(device)
hf_model.eval()

prompts = [
    "The capital of France is",
    "1 + 1 =",
]
MAX_NEW = 20

for prompt in prompts:
    # HuggingFace ground truth
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        hf_out = hf_model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False)
    hf_text = tokenizer.decode(hf_out[0])

    # Engine 生成
    req_id = engine.submit(prompt, max_new_tokens=MAX_NEW, temperature=0.0, top_p=1.0)

    result_text = None
    for _ in range(MAX_NEW + 10):
        completed = engine.step()
        for rid, text in completed:
            if rid == req_id:
                result_text = text
                break
        if result_text is not None:
            break

    match = result_text == hf_text
    print(f"\n  prompt:  {prompt!r}")
    print(f"  HF:      {hf_text!r}")
    print(f"  Engine:  {result_text!r}")
    print(f"  一致:    {'✓' if match else '✗'}")

del hf_model
torch.cuda.empty_cache()

# ============================================================
# 测试二：多请求并发，验证 batch decode 正确性
# ============================================================
print("\n" + "=" * 60)
print("测试二：多请求并发（batch decode）")
print("=" * 60)

prompts_batch = [
    "The capital of France is",
    "1 + 1 =",
    "Hello, my name is",
    "The best programming language is",
]

req_ids = []
for prompt in prompts_batch:
    rid = engine.submit(prompt, max_new_tokens=20, temperature=0.0, top_p=1.0)
    req_ids.append(rid)

print(f"  提交了 {len(req_ids)} 个请求: {req_ids}")

results = {}
max_steps = 50
for step in range(max_steps):
    completed = engine.step()
    for rid, text in completed:
        results[rid] = text
    if len(results) == len(req_ids):
        print(f"  所有请求在第 {step+1} 步完成")
        break

for rid, prompt in zip(req_ids, prompts_batch):
    text = results.get(rid, "未完成")
    print(f"\n  [{rid}] prompt: {prompt!r}")
    print(f"       output: {text!r}")

print(f"\n  完成率: {len(results)}/{len(req_ids)} ✓" if len(results) == len(req_ids) else
      f"\n  ❌ 未全部完成: {len(results)}/{len(req_ids)}")

# ============================================================
# 测试三：请求陆续到达（模拟真实服务场景）
# ============================================================
print("\n" + "=" * 60)
print("测试三：请求陆续到达")
print("=" * 60)

prompts_stream = [
    ("The meaning of life is", 15),
    ("Once upon a time", 20),
    ("The best way to learn programming is", 15),
]

results_stream = {}
submit_schedule = {0: 0, 2: 1, 4: 2}  # step -> prompt_idx，模拟不同时间提交

step = 0
pending_rids = []
max_steps = 80

while len(results_stream) < len(prompts_stream) and step < max_steps:
    # 按计划提交新请求
    if step in submit_schedule:
        idx = submit_schedule[step]
        prompt, max_new = prompts_stream[idx]
        rid = engine.submit(prompt, max_new_tokens=max_new, temperature=0.0, top_p=1.0)
        pending_rids.append(rid)
        print(f"  step {step}: 提交请求 {rid} ({prompt!r})")

    completed = engine.step()
    for rid, text in completed:
        results_stream[rid] = text
        print(f"  step {step}: 请求 {rid} 完成")

    step += 1

for rid, (prompt, _) in zip(pending_rids, prompts_stream):
    text = results_stream.get(rid, "未完成")
    print(f"\n  [{rid}] {prompt!r}")
    print(f"       → {text!r}")

# ============================================================
# 测试四：吞吐量测试
# ============================================================
print("\n" + "=" * 60)
print("测试四：吞吐量测试")
print("=" * 60)

import time

NUM_REQUESTS = 8
TOKENS_PER_REQUEST = 50
test_prompt = "The quick brown fox"

req_ids = []
for _ in range(NUM_REQUESTS):
    rid = engine.submit(test_prompt, max_new_tokens=TOKENS_PER_REQUEST,
                        temperature=0.0, top_p=1.0)
    req_ids.append(rid)

results_tput = {}
torch.cuda.synchronize()
t0 = time.perf_counter()

max_steps = NUM_REQUESTS * TOKENS_PER_REQUEST + 20
for _ in range(max_steps):
    completed = engine.step()
    for rid, text in completed:
        results_tput[rid] = text
    if len(results_tput) == NUM_REQUESTS:
        break

torch.cuda.synchronize()
t1 = time.perf_counter()

total_tokens = sum(
    len(tokenizer.encode(text)) - len(tokenizer.encode(test_prompt))
    for text in results_tput.values()
)
elapsed = t1 - t0
print(f"  请求数: {NUM_REQUESTS}")
print(f"  总生成 token 数: {total_tokens}")
print(f"  总耗时: {elapsed:.2f}s")
print(f"  吞吐量: {total_tokens/elapsed:.1f} tok/s")
print(f"  完成率: {len(results_tput)}/{NUM_REQUESTS}")

# ============================================================
print("\n" + "=" * 60)
print("所有测试完成")
print("=" * 60)