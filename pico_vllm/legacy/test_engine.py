# test_batch_engine.py
import torch
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import NaiveKVCache
from engine import Engine

device = 'cuda'
cfg = ModelConfig()
model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("./weights")

engine = Engine(
    model=model,
    tokenizer=tokenizer,
    cache_cls=NaiveKVCache,
    device=device,
)

# 测试用的 prompts，长度不同
prompts = [
    "The capital of France is",
    "1 + 1 =",
    "用Python写一个快速排序：",
    "What is the meaning of life?",
    "Hello,",
]

max_new_tokens = 30

# ===== 测试一：batch 模式 =====
print("=" * 60)
print("测试一：Batch 模式")
print("=" * 60)

request_ids = []
for prompt in prompts:
    rid = engine.submit(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
    request_ids.append(rid)
    print(f"提交请求 {rid}: {prompt!r}")

print(f"\n开始执行，共 {len(prompts)} 个请求...\n")

results = {}
step_count = 0
while len(results) < len(prompts):
    completed = engine.step()
    step_count += 1
    for rid, text in completed:
        results[rid] = text
        print(f"[Step {step_count}] 请求 {rid} 完成:")
        print(f"  {text[:80]}")

print(f"\n共执行 {step_count} 步")

# ===== 测试二：和单请求 generate() 对比，验证结果一致 =====
print("\n" + "=" * 60)
print("测试二：Batch 结果 vs 单请求 generate() 对比")
print("=" * 60)

# 重置 engine（清空 scheduler 状态）
engine.scheduler.waiting.clear()
engine.scheduler.prefilling.clear()
engine.scheduler.decoding.clear()
engine.scheduler.finished.clear()

all_match = True
for i, prompt in enumerate(prompts[:3]):  # 只对比前3个，省时间
    # 单请求
    single_output = engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
    batch_output = results[request_ids[i]]
    
    match = single_output == batch_output
    if not match:
        all_match = False
    print(f"\nPrompt: {prompt!r}")
    print(f"  single:  {single_output[:60]}")
    print(f"  batch:   {batch_output[:60]}")
    print(f"  一致: {'✓' if match else '✗'}")

print(f"\n{'所有结果一致 ✓' if all_match else '存在不一致 ✗'}")

# ===== 测试三：验证并发请求的调度顺序 =====
print("\n" + "=" * 60)
print("测试三：调度顺序验证")
print("=" * 60)

engine.scheduler.waiting.clear()
engine.scheduler.prefilling.clear()
engine.scheduler.decoding.clear()
engine.scheduler.finished.clear()

# 提交 5 个请求，max_batch_size=4
for i in range(5):
    rid = engine.submit(f"Count to {i}:", max_new_tokens=10, temperature=0.0, top_p=1.0)

step = 0
results2 = {}
while len(results2) < 5:
    prefilling, decoding = engine.scheduler.prefilling[:], engine.scheduler.decoding[:]
    completed = engine.step()
    step += 1
    print(f"Step {step}: prefilling={[r.request_id for r in prefilling]} "
          f"decoding={[r.request_id for r in decoding]} "
          f"completed={[rid for rid, _ in completed]}")
    for rid, text in completed:
        results2[rid] = text

print("\n所有测试完成")