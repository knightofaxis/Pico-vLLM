# test_kvcache.py 极简版，不加载 HF
import torch
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import NaiveKVCache
from engine import Engine
import sampler

device = 'cuda'
cfg = ModelConfig()
model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("./weights")
# print("模型和 tokenizer 加载完成")

engine = Engine(model, tokenizer, NaiveKVCache, device=device)

prompts = [
    "The capital of France is",
    "1 + 1 =",
    "用Python写一个快速排序：",
]

for prompt in prompts:
    output = engine.generate(prompt, max_new_tokens=30, temperature=0)
    print(f"Prompt:  {prompt}")
    print(f"Output:  {output}")
    print("-" * 50)