import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _fused_swiglu_kernel(
    gate_up_ptr, out_ptr,
    D: tl.constexpr,      # intermediate_size
    total_elements: int,  # 输出张量的总元素个数
    BLOCK_SIZE: tl.constexpr,
):
    # 1D 的线程块索引
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # 为了支持任意的 B 和 seq_len，我们把输入展平看待
    # 对于输出的第 idx 个元素，算出它属于哪一行（row）和哪一列（col）
    rows = offsets // D
    cols = offsets % D

    # 在 gate_up 张量中，gate 和 up 是在最后一维拼接的
    # 它的最后一维长度是 2 * D
    # 所以 gate 的物理索引是：row * (2 * D) + col
    # up 的物理索引是：row * (2 * D) + D + col
    gate_idx = rows * (2 * D) + cols
    up_idx = gate_idx + D

    # 仅发生 1 次显存读取！
    gate = tl.load(gate_up_ptr + gate_idx, mask=mask)
    up = tl.load(gate_up_ptr + up_idx, mask=mask)

    # 在寄存器内进行 SiLU 计算: x * sigmoid(x)
    gate_f32 = gate.to(tl.float32)
    sigmoid_gate = tl.sigmoid(gate_f32)
    silu_gate = gate_f32 * sigmoid_gate

    # 在寄存器内完成乘法
    result = silu_gate * up.to(tl.float32)

    # 仅发生 1 次显存写回！
    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)

# 封装成 PyTorch 可调用的函数
def fused_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    # gate_up 形状: (B, seq_len, 2 * intermediate_size) 或任意前缀维度
    D = gate_up.shape[-1] // 2
    
    # 分配输出内存 (CUDA Graph 完美兼容这种 empty_like，因为形状是静态的)
    out = torch.empty_like(gate_up[..., :D])
    
    total_elements = out.numel()
    
    # 启动 1D 网格
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    _fused_swiglu_kernel[grid](
        gate_up, out,
        D=D,
        total_elements=total_elements,
        BLOCK_SIZE=1024,
    )
    return out