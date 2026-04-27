# Pico-vLLM

这是一个从零手写的 LLM 推理引擎，以个人项目形式开发。在 vLLM 的 PagedAttention 架构上融合 SGLang 的 RadixAttention 设计，实现了工业框架的核心特性栈。

目标模型：Qwen2.5-1.5B（bfloat16）

## 亮点

**推理引擎**：支持 Qwen2.5-1.5B 模型，输出对齐 HuggingFace。通过 Triton kernel 的 kernel fusion 实现、CUDA Graph 加速、Continuous Batching和 Prefix Caching 实现推理加速。在 5070 单卡上单 batch 实现 Qwen2.5-1.5B 推理速度 97 tok/s，带宽利用率 78%，超过 vLLM 在同测试场景下的 95 tok/s。

**Prefix Caching**：在 vLLM 的 block-level BlockManager 上实现 SGLang 风格的 token-level radix tree。通过实现双层引用计数模型（lock_ref + logical_ref_count），实现驱逐策略和实际缓存释放时机的解耦。实现了 LRU 驱逐策略 + lazy deletion 驱逐实现。在 2000-token 的共享前缀测试场景下下，实现 TTFT 平均 2.56x，峰值 3.45x 的加速比。

**分布式推理**：实现了 Tensor Parallelism + PD 分离（NCCL后端，支持同步/异步模式）。支持异构并行度组合（P(TP=2)+D(TP=1) ，或者反过来），解决跨并行度的 KV head 重映射。通过 PD 分离，实现 ITL 提升 5.2x，tail latency 从 ~50ms 降低到 ~2ms。

**性能分析**：完整的 nsys profiling 和跨硬件对比（5090 PCIe vs B200 NVLink）（详情参考[Fain的blog](https://koas-w.github.io/)）。在 Qwen2.5-1.5B 模型的 ~2000 token 请求长度下， CPU overhead 仅占总执行时间 6%。

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
| 带宽利用率 | 78% | 77% |

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
```
一键配置环境：bash scripts/venv.sh
```
### 下载权重

```bash
python download_qwen.py
```

### 单卡推理入口示例

```bash
cd pico_vllm/
python run_single.py
```

### 多卡（2卡）推理入口示例： TP=2

```bash
cd pico_vllm/
torchrun --nproc_per_node=2 run_tp.py
```

### 多卡（4卡）推理入口示例： TP=2 + PD分离

```bash
cd pico_vllm/
torchrun --nproc_per_node=4 run_tp_pd.py
```

### 运行 Benchmark

```bash
cd pico_vllm
python benchmarks/benchmark_prefix_cache_long.py
```

### 运行 TP+PD 异构测试

```bash
cd pico_vllm
torchrun --nproc_per_node=3 test/test_hetero_tp_pd.py
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
    ├── tests/                # 单元测试（有废弃）
    ├── benchmarks/           # 性能测试
    └── profiling/            # profiling 脚本和结果
```

## 博客

- 我的博客：[Fain的blog](https://koas-w.github.io/)
- Pico-vLLM开发日志系列：[Pico-vLLM开发日志](https://koas-w.github.io/tags/vllm/)

## Road Map 未来计划

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