# run.py
import torch
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from engine import Engine
from sampler import sample
from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

cfg = ModelConfig()
model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(torch.bfloat16)  # 省显存
tokenizer = AutoTokenizer.from_pretrained("./weights")

engine = Engine(model, tokenizer, device=device)

prompts = [
    "The capital of France is",
    "用Python写一个快速排序：",
    "What is the meaning of life?",
]

# for prompt in prompts:
#     print(f"\nPrompt: {prompt}")
#     print("Output:", engine.generate(prompt, max_new_tokens=50, temperature=0, top_p=1.0))
#     print("-" * 40)
for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    # greedy
    print("Greedy:  ", engine.generate(prompt, max_new_tokens=50, temperature=0))
    # 加温度
    print("T=0.7:   ", engine.generate(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9))
    print("-" * 40)

import time

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# 预热
engine.generate(prompt, max_new_tokens=10, temperature=0)

# 计时
start = time.time()
output = engine.generate(prompt, max_new_tokens=100, temperature=0)
elapsed = time.time() - start

# 算生成了多少新 token
new_tokens = len(tokenizer.encode(output)) - len(input_ids[0])
print(f"生成 {new_tokens} tokens，耗时 {elapsed:.2f}s")
print(f"速度: {new_tokens / elapsed:.1f} tokens/sec")

# run.py 加这段，看看模型参数和中间计算的 dtype
for name, param in model.named_parameters():
    if param.dtype != torch.bfloat16:
        print(f"非 bfloat16: {name} {param.dtype}")