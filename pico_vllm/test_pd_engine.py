# test_pd_engine.py
import torch
import torch.distributed as dist
import os, sys
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager
from engine import Engine

def run_single(prompts, max_new_tokens):
    """单卡 role='pd' 对照"""
    device = torch.device("cuda")
    cfg = ModelConfig()
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights")
    model = model.to(torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./weights")

    bm = BlockManager(
        num_gpu_blocks=200, num_cpu_blocks=0,
        block_size=16, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, dtype=torch.bfloat16,
    )

    engine = Engine(
        model=model, tokenizer=tokenizer, block_manager=bm,
        cache_cls=PagedKVCache, device=device,
        use_cuda_graph=True, tp_size=1, rank=0, role="pd",
    )

    for p in prompts:
        engine.submit(p, max_new_tokens=max_new_tokens, temperature=0, top_p=1.0)
    # engine.mark_finished()
    # 先不 mark_finished，用 run_tp 同样的退出方式
    while True:
        completed = engine.step()
        for req_id, text in completed:
            print(f"  [Request {req_id}] {text}")
        if completed:
            break

    # results = {}
    # while not engine.is_done():
    #     completed = engine.step()
    #     for req_id, text in completed:
    #         results[req_id] = text

    # print("\n=== Single GPU (role='pd') ===")
    # for req_id in sorted(results):
    #     print(f"  [Request {req_id}] {results[req_id]}")

    # return results

def run_pd():
    """双卡 PD 分离"""
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    )
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = ModelConfig()
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights")
    model = model.to(torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./weights")

    bm = BlockManager(
        num_gpu_blocks=200, num_cpu_blocks=0,
        block_size=16, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, dtype=torch.bfloat16,
    )

    role = "p" if rank == 0 else "d"

    engine = Engine(
        model=model, tokenizer=tokenizer, block_manager=bm,
        cache_cls=PagedKVCache, device=device,
        use_cuda_graph=(role == "d"),
        tp_size=1, rank=rank, role=role,
    )

    prompts = [
        "The capital of France is",
        "The capital of France is",
        "1 + 1 =",
    ]
    max_new_tokens = 20

    if rank == 0:
        for p in prompts:
            engine.submit(p, max_new_tokens=max_new_tokens, temperature=0, top_p=1.0)
        engine.mark_finished()

    results = {}
    while not engine.is_done():
        completed = engine.step()
        for req_id, text in completed:
            results[req_id] = text

    if rank == 1:
        print("\n=== PD Disaggregated (rank 0=P, rank 1=D) ===")
        for req_id in sorted(results):
            print(f"  [Request {req_id}] {results[req_id]}")

    dist.barrier()
    if hasattr(engine, 'cuda_graph'):
        del engine.cuda_graph
        if hasattr(engine, 'static_output'):
            del engine.static_output
        torch.cuda.synchronize()
    dist.destroy_process_group()

if __name__ == "__main__":
    mode = sys.argv[-1] if len(sys.argv) > 1 else "single"

    if mode == "single":
        prompts = [
            "The capital of France is",
            "The capital of France is",
            "The capital of France is",
            "1 + 1 =",
        ]
        run_single(prompts, max_new_tokens=20)
    elif mode == "pd":
        run_pd()
    else:
        print(f"Usage: python test_pd_engine.py single")
        print(f"       torchrun --nproc_per_node=2 test_pd_engine.py pd")