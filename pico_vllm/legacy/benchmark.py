# 这个py文件用来测试引擎的效率，并且做成图表
# benchmark.py
import torch
import time
from transformers import AutoTokenizer
from model import Qwen25_15B, ModelConfig
from weights import load_weights
from engine import Engine
from sampler import sample
from cache import NaiveKVCache

def benchmark(engine, tokenizer, prompt, max_new_tokens=100, runs=3):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(engine.device)
    prompt_len = input_ids.shape[1]

    # 预热
    engine.generate(prompt, max_new_tokens=10, temperature=0)
    torch.cuda.synchronize()

    # 分别测 prefill 和 decode
    # --- TTFT ---
    ttft_list = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = engine.model(input_ids)
        torch.cuda.synchronize()
        ttft_list.append((time.perf_counter() - t0) * 1000)  # ms

    # --- 端到端 decode ---
    tps_list = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        output_ids = tokenizer.encode(output)
        new_tokens = len(output_ids) - prompt_len
        tps_list.append(new_tokens / elapsed)

    # --- 显存 ---
    torch.cuda.reset_peak_memory_stats()
    engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB

    return {
        'prompt_len': prompt_len,
        'ttft_ms': sum(ttft_list) / runs,
        'tokens_per_sec': sum(tps_list) / runs,
        'ms_per_token': 1000 / (sum(tps_list) / runs),
        'peak_mem_gb': peak_mem,
    }

def benchmark_decode_curve(engine, tokenizer, prompt, max_new_tokens=50):
    """测量每一步 decode 的耗时，画出随序列长度增长的曲线"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(engine.device)
    
    times = []
    current_ids = input_ids.clone()
    
    engine.model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            logits = engine.model(current_ids)[0, -1, :]
            next_token = logits.argmax()
            
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms
            
            current_ids = torch.cat([
                current_ids,
                next_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return {
        'seq_lengths': list(range(input_ids.shape[1], input_ids.shape[1] + len(times))),
        'step_times_ms': times,
    }

if __name__ == "__main__":
    device = 'cuda'
    cfg = ModelConfig()
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights")
    model = model.to(torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("./weights")
    engine = Engine(model, tokenizer, NaiveKVCache, device=device)

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # 预热
    engine.generate(prompt, max_new_tokens=10, temperature=0)
    torch.cuda.synchronize()

    start = time.perf_counter()
    output = engine.generate(prompt, max_new_tokens=100, temperature=0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = len(tokenizer.encode(output)) - len(input_ids[0])
    print(f"KV Cache: {new_tokens / elapsed:.1f} tokens/sec")
    print(f"Naive baseline: 19-21 tokens/sec")

    prompts_and_lengths = [
        ("Hello " * 20,    "short ~20tok"),
        ("Hello " * 200,   "mid ~200tok"),
        ("Hello " * 800,   "long ~800tok"),
        ("Hello " * 3200,   "very long ~3200tok"),
    ]

    for prompt, label in prompts_and_lengths:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_len = input_ids.shape[1]
        
        # 预热
        engine.generate(prompt, max_new_tokens=5, temperature=0)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        output = engine.generate(prompt, max_new_tokens=50, temperature=0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        new_tokens = len(tokenizer.encode(output)) - prompt_len
        tps = new_tokens / elapsed
        
        print(f"{label:20s} prompt_len={prompt_len:4d}  {tps:.1f} tok/s")
    # # 不同 prompt 长度
    # prompts = {
    #     8:   "Hello world. " * 1,
    #     64:  "Hello world. " * 8,
    #     256: "Hello world. " * 32,
    #     512: "Hello world. " * 64,
    # }

    # print(f"{'prompt_len':>12} {'TTFT(ms)':>10} {'tokens/sec':>12} {'ms/token':>10} {'mem(GB)':>10}")
    # print("-" * 60)

    # for target_len, prompt in prompts.items():
    #     result = benchmark(engine, tokenizer, prompt, max_new_tokens=100)
    #     print(
    #         f"{result['prompt_len']:>12} "
    #         f"{result['ttft_ms']:>10.1f} "
    #         f"{result['tokens_per_sec']:>12.1f} "
    #         f"{result['ms_per_token']:>10.1f} "
    #         f"{result['peak_mem_gb']:>10.3f}"
    #     )

    
    # # 把序列拉到 2000+
    # prompt_long = "Hello world. " * 200  # 大约 600 token
    # result = benchmark_decode_curve(engine, tokenizer, prompt_long, max_new_tokens=200)

    # print(f"{'seq_len':>10} {'ms/step':>10}")
    # print("-" * 25)
    # for seq_len, t in zip(result['seq_lengths'][::20], result['step_times_ms'][::20]):
    #     print(f"{seq_len:>10} {t:>10.1f}")

    # first = result['step_times_ms'][0]
    # last = result['step_times_ms'][-1]
    # print(f"\n第1步: {first:.1f}ms → 第{len(result['step_times_ms'])}步: {last:.1f}ms")
    # print(f"增长倍数: {last/first:.2f}x")