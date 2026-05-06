import triton
import triton.language as tl
import torch
import torch.nn as nn

@triton.jit
def _rmsnorm_kernel(
    X_ptr, Y_ptr, W_ptr, 
    stride_x, stride_y, 
    N, eps, 
    BLOCK_SIZE: tl.constexpr
):
    # 针对 Decode 阶段（B=1 或 B=小Batch），每行分配一个 block
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride_x
    Y_ptr += row_idx * stride_y

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # 加载数据并转为 float32 防止溢出
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # 计算方差
    var = tl.sum(x * x, axis=0) / N
    rsqrt = 1.0 / tl.sqrt(var + eps)
    
    # 归一化并乘上权重
    out = x * rsqrt
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = out * w
    
    # 写回
    tl.store(Y_ptr + cols, out.to(Y_ptr.dtype.element_ty), mask=mask)

class FastRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size
        # 找到大于 hidden_size 的最小 2 的幂次方作为 BLOCK_SIZE
        self.block_size = triton.next_power_of_2(hidden_size)

    def forward(self, x):
        # x: (B, seq, hidden_size) 或 (B, hidden_size)
        x_shape = x.shape
        x_2d = x.view(-1, self.hidden_size)
        y_2d = torch.empty_like(x_2d)
        
        # 启动 kernel (行数等于总 token 数)
        _rmsnorm_kernel[(x_2d.shape[0],)](
            x_2d, y_2d, self.weight,
            x_2d.stride(0), y_2d.stride(0),
            self.hidden_size, self.eps,
            BLOCK_SIZE=self.block_size
        )
        return y_2d.view(x_shape)