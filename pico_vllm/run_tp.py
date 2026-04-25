# run_tp.py
import torch
import torch.distributed as dist
import os
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager
from engine import Engine

def main():
    tp_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if tp_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = ModelConfig(tp_size=tp_size)
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights", tp_size=tp_size, tp_rank=rank)
    model = model.to(torch.bfloat16).to(device)

    tokenizer = AutoTokenizer.from_pretrained("./weights")
    BLOCK_SIZE = 16

    bm = BlockManager(
        num_gpu_blocks=200, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim, dtype=torch.bfloat16,
    )

    engine = Engine(
        model=model, tokenizer=tokenizer, block_manager=bm,
        cache_cls=PagedKVCache, device=device,
        use_cuda_graph=True,
        local_tp_size=tp_size, rank=rank,
    )

    # 提交请求
    engine.submit("The capital of France is", max_new_tokens=20, temperature=1, top_p=0.9)

    # 运行直到完成
    while True:
        completed = engine.step()
        for req_id, text in completed:
            if rank == 0:
                print(f"[Request {req_id}] {text}")
        if completed:
            break

    print(f"[Rank {rank}] generation done", flush=True)
    if tp_size > 1:
        # 释放 CUDA Graph 持有的 NCCL 资源
        if engine.use_cuda_graph:
            del engine.cuda_graph
            del engine.static_output
            torch.cuda.synchronize()
        dist.destroy_process_group()
        print(f"[Rank {rank}] destroyed process group", flush=True)

if __name__ == "__main__":
    main()