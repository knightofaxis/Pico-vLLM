# benchmark_pd.py
import torch
import torch.distributed as dist
import time
import os
import sys
import json
import math
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager
from engine import Engine
import sampler

def create_engine(device, role="pd", use_cuda_graph=True, tp_size=1, rank=0):
    cfg = ModelConfig(tp_size=tp_size)
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights", tp_size=tp_size, rank=rank)
    model = model.to(torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./weights")
    
    bm = BlockManager(
        num_gpu_blocks=500, num_cpu_blocks=0,
        block_size=16, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim, dtype=torch.bfloat16,
    )

    engine = Engine(
        model=model, tokenizer=tokenizer, block_manager=bm,
        cache_cls=PagedKVCache, device=device,
        use_cuda_graph=use_cuda_graph and role in ("d", "pd"),
        tp_size=tp_size, rank=rank, role=role,
    )
    return engine, tokenizer


def benchmark_scenario_a(engine, tokenizer, rank, role, label):
    """场景 A：单请求，测 TTFT + ITL + KV 传输开销"""
    MAX_NEW_TOKENS = 50
    prompt = "The quick brown fox jumps over the lazy dog. " * 5  # ~50 tokens prompt

    engine.submit(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0, top_p=1.0)
    engine.mark_finished()

    token_times = []
    ttft = None

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    while not engine.is_done():
        t_step_start = time.perf_counter()
        completed = engine.step()
        t_step_end = time.perf_counter()

        if role in ("d", "pd"):
            step_ms = (t_step_end - t_step_start) * 1000
            if ttft is None and step_ms > 0:
                # 第一个有实际工作的 step
                # 对 D 侧：包含 recv + 第一步 decode
                # 对 pd 侧：包含 prefill + 第一步 decode
                ttft = (t_step_end - t_start) * 1000
            elif ttft is not None:
                token_times.append(step_ms)

        if completed:
            break

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    if (role == "pd") or (role == "d"):
        total_ms = (t_end - t_start) * 1000
        itl_list = token_times[1:]  # 去掉第一步（可能包含 recv 开销）

        if itl_list:
            itl_list.sort()
            p50 = itl_list[len(itl_list) // 2]
            p95 = itl_list[int(len(itl_list) * 0.95)]
            p99 = itl_list[min(int(len(itl_list) * 0.99), len(itl_list) - 1)]
            avg_itl = sum(itl_list) / len(itl_list)
        else:
            p50 = p95 = p99 = avg_itl = 0

        print(f"\n{'='*60}")
        print(f"  场景 A：单请求 [{label}]")
        print(f"{'='*60}")
        print(f"  TTFT:      {ttft:.2f} ms")
        print(f"  总时间:    {total_ms:.2f} ms")
        print(f"  ITL avg:   {avg_itl:.2f} ms")
        print(f"  ITL P50:   {p50:.2f} ms")
        print(f"  ITL P95:   {p95:.2f} ms")
        print(f"  ITL P99:   {p99:.2f} ms")
        print(f"  Tokens:    {len(token_times)}")
        print(f"  Tok/s:     {len(token_times) / total_ms * 1000:.1f}")
        print(f"{'='*60}")


def benchmark_scenario_b(engine, tokenizer, rank, role, label):
    """场景 B：多请求同时提交，测 batch decode 的 ITL 稳定性"""
    MAX_NEW_TOKENS = 30
    prompts = [
        "The capital of France is",
        "The quick brown fox jumps over",
        "In the year 2024, artificial intelligence",
        "The meaning of life is",
    ]

    for p in prompts:
        engine.submit(p, max_new_tokens=MAX_NEW_TOKENS, temperature=0, top_p=1.0)
    engine.mark_finished()

    step_times = []
    completed_count = 0

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    while not engine.is_done():
        t_step_start = time.perf_counter()
        completed = engine.step()
        t_step_end = time.perf_counter()

        if role in ("d", "pd"):
            step_ms = (t_step_end - t_step_start) * 1000
            step_times.append(step_ms)

        completed_count += len(completed)

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    if (role == "pd") or (role == "d"):
        total_ms = (t_end - t_start) * 1000

        # 分离 prefill 步和 decode 步
        # prefill 步通常远长于 decode 步，用阈值分离
        if step_times:
            median = sorted(step_times)[len(step_times) // 2]
            threshold = median * 3  # prefill 步通常 >3x decode 步
            decode_steps = [t for t in step_times if t < threshold]
            prefill_steps = [t for t in step_times if t >= threshold]
        else:
            decode_steps = []
            prefill_steps = []

        if decode_steps:
            decode_steps.sort()
            d_avg = sum(decode_steps) / len(decode_steps)
            d_p50 = decode_steps[len(decode_steps) // 2]
            d_p95 = decode_steps[int(len(decode_steps) * 0.95)]
            d_p99 = decode_steps[min(int(len(decode_steps) * 0.99), len(decode_steps) - 1)]
            d_min = decode_steps[0]
            d_max = decode_steps[-1]
        else:
            d_avg = d_p50 = d_p95 = d_p99 = d_min = d_max = 0

        print(f"\n{'='*60}")
        print(f"  场景 B：{len(prompts)} 请求同时提交 [{label}]")
        print(f"{'='*60}")
        print(f"  总时间:       {total_ms:.2f} ms")
        print(f"  总步数:       {len(step_times)}")
        print(f"  Prefill 步:   {len(prefill_steps)} (avg {sum(prefill_steps)/max(len(prefill_steps),1):.2f} ms)")
        print(f"  Decode 步:    {len(decode_steps)}")
        print(f"  Decode ITL avg: {d_avg:.2f} ms")
        print(f"  Decode ITL P50: {d_p50:.2f} ms")
        print(f"  Decode ITL P95: {d_p95:.2f} ms")
        print(f"  Decode ITL P99: {d_p99:.2f} ms")
        print(f"  Decode ITL min: {d_min:.2f} ms")
        print(f"  Decode ITL max: {d_max:.2f} ms")
        print(f"  完成请求:     {completed_count}")
        print(f"{'='*60}")


def benchmark_scenario_c(engine, tokenizer, rank, role, label):
    """场景 C：交错到达，先提交 2 个，decode 几步后再提交 2 个"""
    MAX_NEW_TOKENS = 30

    prompts_wave1 = [
        "The capital of France is",
        "The quick brown fox jumps over",
    ]
    prompts_wave2 = [
        "In the year 2024, artificial intelligence",
        "The meaning of life is",
    ]

    # 第一波
    for p in prompts_wave1:
        engine.submit(p, max_new_tokens=MAX_NEW_TOKENS, temperature=0, top_p=1.0)

    step_times = []
    wave2_submitted = False
    completed_count = 0

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    step_idx = 0
    while True:
        # 在第 10 步插入第二波请求
        if step_idx == 10 and not wave2_submitted:
            for p in prompts_wave2:
                engine.submit(p, max_new_tokens=MAX_NEW_TOKENS, temperature=0, top_p=1.0)
            engine.mark_finished()
            wave2_submitted = True
            if role in ("d", "pd"):
                print(f"  [Step {step_idx}] Wave 2 submitted")

        t_step_start = time.perf_counter()
        completed = engine.step()
        t_step_end = time.perf_counter()

        if role in ("d", "pd"):
            step_ms = (t_step_end - t_step_start) * 1000
            step_times.append((step_idx, step_ms, len(completed) > 0))

        completed_count += len(completed)
        step_idx += 1

        if engine.is_done():
            break

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    if (role == "pd") or (role == "d"):
        total_ms = (t_end - t_start) * 1000

        # 分析 wave2 插入前后的 ITL 变化
        before_wave2 = [t for idx, t, _ in step_times if idx < 10 and idx > 0]
        after_wave2 = [t for idx, t, _ in step_times if 10 <= idx < 15]
        steady_decode = [t for idx, t, _ in step_times if idx >= 15]

        def stats(lst, name):
            if not lst:
                print(f"    {name}: 无数据")
                return
            lst.sort()
            print(f"    {name}: avg={sum(lst)/len(lst):.2f} ms, "
                  f"P50={lst[len(lst)//2]:.2f} ms, "
                  f"max={lst[-1]:.2f} ms, "
                  f"n={len(lst)}")

        print(f"\n{'='*60}")
        print(f"  场景 C：交错到达 [{label}]")
        print(f"{'='*60}")
        print(f"  总时间:    {total_ms:.2f} ms")
        print(f"  总步数:    {len(step_times)}")
        stats(before_wave2, "Wave1 decode (step 1-9)")
        stats(after_wave2, "Wave2 插入后 (step 10-14)")
        stats(steady_decode, "稳定 decode (step 15+)")
        print(f"  完成请求:  {completed_count}")
        print(f"{'='*60}")


def run_single():
    device = torch.device("cuda")
    for scenario_fn, name in [
        (benchmark_scenario_a, "A"),
        (benchmark_scenario_b, "B"),
        (benchmark_scenario_c, "C"),
    ]:
        engine, tokenizer = create_engine(device, role="pd", use_cuda_graph=True)
        scenario_fn(engine, tokenizer, 0, "pd", f"单卡 pd")
        # 清理
        if hasattr(engine, 'cuda_graph'):
            del engine.cuda_graph
            if hasattr(engine, 'static_output'):
                del engine.static_output
        del engine
        torch.cuda.synchronize()


def run_pd():
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    )
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    role = "p" if rank == 0 else "d"

    for scenario_fn, name in [
        (benchmark_scenario_a, "A"),
        (benchmark_scenario_b, "B"),
        (benchmark_scenario_c, "C"),
    ]:
        dist.barrier()
        engine, tokenizer = create_engine(
            device, role=role, use_cuda_graph=(role == "d"),
        )
        
        if role in ("p", "d"):
            warmup = torch.zeros(1, device=device)
            if rank == 0:
                dist.send(warmup, dst=1)
                dist.recv(warmup, src=1)
                dist.isend(warmup, dst=1)
                dist.irecv(warmup, src=1)
            else:
                dist.recv(warmup, src=0)
                dist.send(warmup, dst=0)
                dist.isend(warmup, dst=0)
                dist.irecv(warmup, src=0)
            torch.cuda.synchronize()
        
        scenario_fn(engine, tokenizer, rank, role, f"PD 分离 ({role})")
        # 清理
        if hasattr(engine, 'cuda_graph'):
            del engine.cuda_graph
            if hasattr(engine, 'static_output'):
                del engine.static_output
        del engine
        torch.cuda.synchronize()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    mode = sys.argv[-1] if len(sys.argv) > 1 else "single"

    if mode == "single":
        run_single()
    elif mode == "pd":
        run_pd()
    else:
        print("Usage:")
        print("  python benchmark_pd.py single")
        print("  torchrun --nproc_per_node=2 benchmark_pd.py pd")