import torch
import triton
import triton.language as tl

@triton.jit
def _fused_add_rmsnorm_kernel(
    X_ptr, Add_ptr, Y_ptr, W_ptr,
    stride_x, stride_add, stride_y,
    N, eps, BLOCK_SIZE: tl.constexpr
):
    # 行索引 (对应 Batch * SeqLen 中的哪一个 token)
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride_x
    Add_ptr += row_idx * stride_add
    Y_ptr += row_idx * stride_y

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 1. 读入旧残差(x) 和 新增量(attn_out/ffn_out)
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    add = tl.load(Add_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 2. 寄存器内相加
    x_new = x + add

    # 3. 将更新后的残差原地写回显存
    tl.store(X_ptr + cols, x_new.to(X_ptr.dtype.element_ty), mask=mask)

    # 4. 紧接着在寄存器里算 RMSNorm
    var = tl.sum(x_new * x_new, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(var + eps)
    out = x_new * rsqrt

    # 5. 乘权重并输出给下一层网络
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = out * w
    tl.store(Y_ptr + cols, out.to(Y_ptr.dtype.element_ty), mask=mask)

def fused_add_rmsnorm(x: torch.Tensor, residual_add: torch.Tensor, weight: torch.Tensor, eps: float):
    """
    注意：输入张量 x 会被 In-place (原地) 更新为 x + residual_add！
    返回的是 norm(x + residual_add) 的结果。
    """
    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size)
    add_2d = residual_add.view(-1, hidden_size)
    
    # 静态分配 norm 的输出
    y_2d = torch.empty_like(x_2d)
    
    block_size = triton.next_power_of_2(hidden_size)
    
    _fused_add_rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d, add_2d, y_2d, weight,
        x_2d.stride(0), add_2d.stride(0), y_2d.stride(0),
        hidden_size, eps,
        BLOCK_SIZE=block_size
    )
    return y_2d.view_as(x)