# profile_paged.py
import torch
import time
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from cache import PagedKVCache, BlockManager

device = 'cuda'
cfg = ModelConfig()

BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 512  # 512 × 16 tokens × 28层 × 2(KV) × 2heads × 128dim × 2bytes ≈ 1.8GB

def make_block_manager():
    return BlockManager(
        num_gpu_blocks=NUM_GPU_BLOCKS,
        num_cpu_blocks=0,
        block_size=BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.bfloat16,
    )

def make_paged_cache(bm, max_seq_len=4096):
    return PagedKVCache(
        block_manager=bm,
        num_layers=cfg.num_hidden_layers,
        max_seq_len=max_seq_len,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )


def measure_decode_step(model, prompt_ids, device, cfg):
    bm = make_block_manager()
    cache = make_paged_cache(bm, max_seq_len=prompt_ids.shape[1] + 64)

    # prefill 只做一次
    with torch.no_grad():
        logits = model(prompt_ids, kv_caches=[cache], is_prefill=True)
    last_token = logits[0, -1:].argmax(-1, keepdim=True)  # (1, 1)
    prefill_seq_len = cache.seq_len
    prefill_allocated = cache.allocated_cache_block_num
    torch.cuda.synchronize()

    runs = 5
    times = []

    with torch.no_grad():
        for i in range(runs + 1):
            # 回滚 cache 状态到 prefill 结束
            cache._seq_len = prefill_seq_len
            cache._updated_layer = 0
            cache.allocated_cache_block_num = prefill_allocated

            # 预分配下一个 token 的 block（模拟真实 decode 入口）
            cache.prepare_decode_step()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(last_token, kv_caches=[cache], is_prefill=False)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if i > 0:
                times.append(elapsed_ms)

    cache.reset()
    del bm

    times.sort()
    return times[len(times) // 2]


def profile_decode_over_length(model, tokenizer, device, seq_lengths, cfg):
    results = []
    base_text = "The quick brown fox jumps over the lazy dog. " * 200
    base_ids = tokenizer.encode(base_text, return_tensors='pt').to(device)

    for seq_len in seq_lengths:
        long_ids = base_ids.repeat(1, (seq_len // base_ids.shape[1]) + 2)
        prompt_ids = long_ids[:, :seq_len]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        try:
            ms = measure_decode_step(model, prompt_ids, device, cfg)
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3

            results.append({
                'seq_len': seq_len,
                'ms_per_step': ms,
                'tokens_per_sec': 1000 / ms,
                'peak_mem_gb': peak_mem,
            })
            print(f"seq_len={seq_len:5d}  {ms:7.2f}ms  "
                  f"{1000/ms:6.1f} tok/s  "
                  f"alloc={peak_mem:.3f}GB  reserved={reserved:.3f}GB")

        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
            print(f"seq_len={seq_len:5d}  OOM: {e}")
            torch.cuda.empty_cache()
            break

    return results


def plot_results(results, save_path="profile_paged_Triton.png"):
    seq_lens = [r['seq_len'] for r in results]
    ms_steps = [r['ms_per_step'] for r in results]
    tps      = [r['tokens_per_sec'] for r in results]
    mems     = [r['peak_mem_gb'] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Paged KV Cache Inference Profiling (Non-Gather)', fontsize=14)

    # 图1：ms/step
    axes[0].plot(seq_lens, ms_steps, 'b-o', markersize=4)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('ms / step')
    axes[0].set_title('Latency per Decode Step vs Sequence Length')
    axes[0].grid(True, alpha=0.3)
    if len(seq_lens) > 1:
        x = np.array(seq_lens)
        for deg, color, label in [(1, 'r', 'linear fit'), (2, 'g', 'quadratic fit')]:
            coeffs = np.polyfit(x, ms_steps, deg)
            x_fit = np.linspace(x[0], x[-1], 200)
            axes[0].plot(x_fit, np.polyval(coeffs, x_fit),
                        f'{color}--', alpha=0.7, label=label)
        axes[0].legend()

    # 图2：tokens/sec
    axes[1].plot(seq_lens, tps, 'g-o', markersize=4)
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Tokens / sec')
    axes[1].set_title('Throughput vs Sequence Length')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=max(tps), color='r', linestyle='--',
                   alpha=0.5, label=f'peak: {max(tps):.1f} tok/s')
    axes[1].legend()

    # 图3：显存
    axes[2].plot(seq_lens, mems, 'r-o', markersize=4)
    axes[2].set_xlabel('Sequence Length')
    axes[2].set_ylabel('Peak Memory (GB)')
    axes[2].set_title('Peak GPU Memory vs Sequence Length')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=8.0, color='k', linestyle='--',
                   alpha=0.5, label='8GB limit')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到 {save_path}")


if __name__ == "__main__":
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights")
    model = model.to(torch.bfloat16).to(device)
    model.eval()
    # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    tokenizer = AutoTokenizer.from_pretrained("./weights")

    seq_lengths = list(range(16, 4097, 16))

    print(f"{'seq_len':>8} {'ms/step':>10} {'tok/s':>8} {'alloc(GB)':>10} {'reserved(GB)':>13}")
    print("-" * 55)

    results = profile_decode_over_length(model, tokenizer, device, seq_lengths, cfg)

    with open("profile_paged_Triton.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n数据已保存到 profile_paged_Triton.json")

    plot_results(results)

    print("\n=== 关键统计 ===")
    def get_ms(target):
        for r in results:
            if r['seq_len'] >= target:
                return r['ms_per_step']
        return None

    for length in [128, 512, 1024, 2048, 4096]:
        ms = get_ms(length)
        if ms:
            print(f"seq_len={length:5d}: {ms:.2f}ms  {1000/ms:.1f} tok/s")

    if len(results) > 1:
        ratio = results[-1]['ms_per_step'] / results[0]['ms_per_step']
        print(f"增长倍数 (first→last): {ratio:.2f}x")