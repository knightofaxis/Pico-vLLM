
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from engine import Engine
from blockmanager import BlockManager
from cache import PagedKVCache
from transformers import AutoTokenizer
import torch

cfg = ModelConfig()
model = Qwen25_15B(cfg)
model = load_weights(model, "./weights")
model = model.to(torch.bfloat16).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("./weights")

bm = BlockManager(
    num_gpu_blocks=500, num_cpu_blocks=0,
    block_size=16, num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads,
    head_dim=cfg.head_dim, dtype=torch.bfloat16,
)

engine = Engine(
    model=model, tokenizer=tokenizer, block_manager=bm,
    cache_cls=PagedKVCache, device="cuda",
    use_cuda_graph=True,
    enable_prefix_cache=True,
)

engine.submit("The capital of France is", max_new_tokens=20, temperature=0, top_p=1.0)

while True:
    completed = engine.step()
    for req_id, text in completed:
        print(f"[{req_id}] {text}")
    if completed:
        break