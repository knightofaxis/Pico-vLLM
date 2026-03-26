# vllm_benchmark.py
from vllm import LLM, SamplingParams
import time

def run_benchmark():
    llm = LLM(model="./weights", dtype="bfloat16", gpu_memory_utilization=0.80, max_model_len=8192)
    params = SamplingParams(temperature=0, max_tokens=100)
    
    prompts_and_lengths = [
        "Hello " * 1,
        "Hello " * 50,
        "Hello " * 200,
        "Hello " * 400,
    ]
    
    for prompt in prompts_and_lengths:
        # 预热
        llm.generate([prompt], SamplingParams(temperature=0, max_tokens=5))
        
        start = time.perf_counter()
        outputs = llm.generate([prompt], params)
        elapsed = time.perf_counter() - start
        
        input_len = len(llm.get_tokenizer().encode(prompt))
        output_len = len(outputs[0].outputs[0].token_ids)
        tps = output_len / elapsed
        
        print(f"prompt_len={input_len:5d}  {tps:.1f} tok/s  "
              f"output_tokens={output_len}")

if __name__ == '__main__':
    run_benchmark()