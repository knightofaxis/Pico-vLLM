# Pico-vLLM

从零手写的 LLM 推理引擎，单人一个月开发。在 vLLM 的 PagedAttention 架构上融合 SGLang 的 RadixAttention 设计，实现了工业框架的核心特性栈。

目标模型：Qwen2.5-1.5B（bfloat16）

## 亮点

**推理引擎**：Qwen2.5-1.5B 从零实现，逐层数值对齐 HuggingFace。5 个手写 Triton kernel，CUDA Graph 加速，Continuous Batching。单卡 97 tok/s，带宽利用率 78%，超过 vLLM 同硬件的 95 tok/s。

**Prefix Caching**：在 vLLM 的 block-level BlockManager 上实现 SGLang 风格的 token-level radix tree。双层引用计数模型（lock_ref + logical_ref_count），LRU 驱逐 + lazy deletion。2083-token 共享前缀下 warm TTFT 平均 2.56x 加速，峰值 3.45x。

**分布式推理**：Tensor Parallelism + PD 分离（同步/异步两版传输层）。支持异构并行度组合（P(TP=2)+D(TP=1) 和反向），解决跨并行度的 KV head 重映射。PD 分离 ITL 提升 5.2x，tail latency 50ms→2ms。

**性能分析**：4 次 nsys profiling，跨硬件对比（5090 PCIe vs B200 NVLink）。AllReduce 延迟拆解（PCIe 下占双卡 GPU 时间 48%，NVLink 下 2-5%）。CPU overhead 仅占总执行时间 6%。

## 特性清单

| 类别 | 特性 | 状态 |
|:---|:---|:---:|
| 模型 | Qwen2.5-1.5B 全手写（RoPE / GQA / SwiGLU / RMSNorm） | ✅ |
| 模型 | 算子融合（QKV fused / gate_up fused / rotate_half in-place） | ✅ |
| Kernel | PagedAttention Prefill Kernel (Triton) | ✅ |
| Kernel | PagedAttention Decode Kernel (Triton) | ✅ |
| Kernel | Fused RoPE + KV Cache Store (Triton) | ✅ |
| Kernel | Fused RMSNorm + Residual Add (Triton) | ✅ |
| Kernel | SwiGLU Fused (Triton) | ✅ |
| 调度 | Continuous Batching + FCFS Scheduler | ✅ |
| 加速 | CUDA Graph（auto capture/replay，可回退 eager） | ✅ |
| 存储 | PagedAttention + BlockManager | ✅ |
| 缓存 | Radix Tree Prefix Caching（token 粒度索引 + block 粒度存储） | ✅ |
| 缓存 | 双层引用计数（lock_ref + logical_ref_count） | ✅ |
| 缓存 | LRU 驱逐 + Lazy Deletion + Recompute 策略 | ✅ |
| 分布式 | Tensor Parallelism（同构） | ✅ |
| 分布式 | PD 分离（同步 + 异步，KVTransferBase 可插拔） | ✅ |
| 分布式 | 同构 TP + PD | ✅ |
| 分布式 | 异构 TP + PD（跨并行度 KV head 重映射） | ✅ |

## 性能数据

### 单卡推理（5090 PCIe, bfloat16）

| 指标 | Pico-vLLM | vLLM (同硬件) |
|:---|:---:|:---:|
| Decode Throughput | 97 tok/s | 95 tok/s |
| 带宽利用率 | 78% | — |
| CPU Overhead 占比 | 6% | 62% (v0.5 优化前) |

### Prefix Cache（2083-token 共享前缀）

| 场景 | OFF | ON | 加速比 |
|:---|:---:|:---:|:---:|
| Cold (首次) | 48.92ms | 41.59ms | 1.18x |
| Warm (平均) | 41.17ms | 16.06ms | 2.56x |
| Warm (最佳) | 41.71ms | 12.08ms | 3.45x |
| Hit Rate | — | 62.8% (短) / 98.5% (长) | — |

### PD 分离

| 指标 | 单卡 | PD 分离 | 改善 |
|:---|:---:|:---:|:---:|
| ITL | ~10ms | ~2ms | 5.2x |
| Tail Latency | 50ms | 2ms | 25x |

## Quick Start

### 环境要求

- Python 3.10+
- PyTorch 2.1+ (with CUDA)
- Triton 2.1+
- transformers

### 下载权重

```bash
bash download_weights.sh
```

### 单卡推理

```python
from pico_vllm.model import Qwen25_15B, ModelConfig
from pico_vllm.weights import load_weights
from pico_vllm.engine import Engine
from pico_vllm.blockmanager import BlockManager
from pico_vllm.cache import PagedKVCache
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
```

### 多卡 TP

```bash
torchrun --nproc_per_node=2 your_script.py
```

### 运行测试

```bash
cd pico_vllm
python -m pytest tests/ -v
```

### 运行 Benchmark

```bash
cd pico_vllm
python benchmarks/benchmark_prefix_cache_long.py
```

## 项目结构

```
PicovLLM/
├── README.md
├── download_weights.sh
│
└── pico_vllm/
    ├── model.py              # Qwen2.5-1.5B 模型定义
    ├── weights.py            # 权重加载（含 TP 分片）
    ├── sampler.py            # 采样器（greedy / temperature / top_p）
    ├── engine.py             # 推理引擎主循环 + CUDA Graph
    ├── scheduler.py          # Continuous Batching 调度器
    ├── cache.py              # PagedKVCache
    ├── blockmanager.py       # 物理块管理 + 引用计数
    ├── radix_tree.py         # Radix Tree（token 粒度前缀索引）
    ├── prefix_cache.py       # Prefix Cache 桥接层
    ├── kv_transfer.py        # PD 分离传输层
    │
    ├── kernels/              # 5 个手写 Triton kernel
    │   ├── attention.py
    │   ├── fused_add_norm.py
    │   ├── fused_rope_kvcache_store.py
    │   ├── swiglu.py
    │   └── store_kvcache.py
    │
    ├── tests/                # 单元测试
    ├── benchmarks/           # 性能测试
    └── profiling/            # nsys profiling 结果
```

## 博客

- 我的博客：https://koas-w.github.io/
- Pico-vLLM开发日志系列：https://koas-w.github.io/tags/vllm/

## 未来计划

- TP 通信异步化 + 层间通算重叠
- PD 传输后端替换为 NIXL
- Scheduler 的 Chunked Prefill 策略
- Prefix 共享 block 的写时复制（COW）
- GPU和CPU间的 Offload 驱逐策略
- 驱逐策略与 Radix Tree 结构的解耦
- 其他CPU侧代码的整体性能优化

## 参考

- [vLLM: PagedAttention](https://github.com/vllm-project/vllm)
- [SGLang: RadixAttention](https://github.com/sgl-project/sglang)
- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)