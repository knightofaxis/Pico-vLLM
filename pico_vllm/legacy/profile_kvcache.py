# profile_kvcache.py
import torch
import time
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from engine import Engine
from sampler import sample

def measure_decode_step(model, prompt_ids, device, cfg):
    from cache import NaiveKVCache
    
    kv_cache = NaiveKVCache(
        num_layers=cfg.num_hidden_layers,
        max_seq_len=4096,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        device=device,
        dtype=next(model.parameters()).dtype
    )
    
    # prefill 只做一次
    with torch.no_grad():
        logits = model(prompt_ids, kv_cache=kv_cache)
    last_token = logits[0, -1:].argmax(-1, keepdim=True)
    prefill_len = kv_cache.seq_len
    torch.cuda.synchronize()
    
    # 多次测 decode，每次测完回滚 seq_len
    runs = 5
    times = []
    with torch.no_grad():
        for i in range(runs + 1):
            # 回滚到 prefill 结束的状态
            kv_cache._seq_len = prefill_len
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(last_token, kv_cache=kv_cache)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            
            if i > 0:  # 跳过第一次预热
                times.append(elapsed_ms)
    
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
                  
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError):
            print(f"seq_len={seq_len:5d}  OOM")
            torch.cuda.empty_cache()
            break
    
    return results

def plot_results(results, save_path="profile_kvcache.png"):
    seq_lens = [r['seq_len'] for r in results]
    ms_steps = [r['ms_per_step'] for r in results]
    tps = [r['tokens_per_sec'] for r in results]
    mems = [r['peak_mem_gb'] for r in results]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('KV Cache Inference Profiling (No CUDA Kernel)', fontsize=14)
    
    # 图1：ms/step vs seq_len
    axes[0].plot(seq_lens, ms_steps, 'b-o', markersize=4)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('ms / step')
    axes[0].set_title('Latency per Decode Step vs Sequence Length')
    axes[0].grid(True, alpha=0.3)
    # 标注理论二次方趋势线
    if len(seq_lens) > 1:
        import numpy as np
        x = np.array(seq_lens)
        # 用前几个点拟合线性和二次
        coeffs_linear = np.polyfit(x, ms_steps, 1)
        coeffs_quad = np.polyfit(x, ms_steps, 2)
        x_fit = np.linspace(x[0], x[-1], 200)
        axes[0].plot(x_fit, np.polyval(coeffs_linear, x_fit), 
                    'r--', alpha=0.7, label='linear fit')
        axes[0].plot(x_fit, np.polyval(coeffs_quad, x_fit), 
                    'g--', alpha=0.7, label='quadratic fit')
        axes[0].legend()
    
    # 图2：tokens/sec vs seq_len
    axes[1].plot(seq_lens, tps, 'g-o', markersize=4)
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Tokens / sec')
    axes[1].set_title('Throughput vs Sequence Length')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=max(tps), color='r', linestyle='--', 
                   alpha=0.5, label=f'peak: {max(tps):.1f} tok/s')
    axes[1].legend()
    
    # 图3：显存 vs seq_len
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
    device = 'cuda'
    cfg = ModelConfig()
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights")
    model = model.to(torch.bfloat16).to(device)
    # this can increase performance by ~65%, but we are doing it ourself.
    # model = torch.compile(model, mode="reduce-overhead") 
    tokenizer = AutoTokenizer.from_pretrained("./weights")

    # seq_len 从 16 到 4096，间隔 16
    import numpy as np
    seq_lengths = list(range(16, 4097, 16))  # 256 个数据点

    print(f"{'seq_len':>8} {'ms/step':>10} {'tok/s':>8} {'mem(GB)':>10}")
    print("-" * 45)
    
    results = profile_decode_over_length(model, tokenizer, device, seq_lengths, cfg=cfg)
    
    # 保存原始数据
    with open("profile_kvcache.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n数据已保存到 profile_kvcache.json")
    
    # 出图
    plot_results(results)
    
    # 打印关键统计
    print("\n=== 关键统计 ===")
    print(f"seq_len=16:   {results[0]['ms_per_step']:.2f}ms")
    if len(results) >= 32:
        print(f"seq_len=512:  {results[31]['ms_per_step']:.2f}ms")
    if len(results) >= 64:
        print(f"seq_len=1024: {results[63]['ms_per_step']:.2f}ms")
    if len(results) >= 128:
        print(f"seq_len=2048: {results[127]['ms_per_step']:.2f}ms")
    if len(results) >= 256:
        print(f"seq_len=4096: {results[255]['ms_per_step']:.2f}ms")
    if len(results) > 1:
        ratio = results[-1]['ms_per_step'] / results[0]['ms_per_step']
        print(f"增长倍数 (first→last): {ratio:.2f}x")