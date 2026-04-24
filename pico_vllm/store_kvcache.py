import torch
from torch import Tensor
import triton
import triton.language as tl

@triton.jit
def store_kvcache_kernel(
    k_ptr, v_ptr,              # 源 K/V: (total_tokens, n_kv_heads, HEAD_DIM)
    k_cache_ptr, v_cache_ptr,  # 目标 Cache: (num_blocks, n_kv_heads, block_size, HEAD_DIM)
    slot_mapping_ptr,          # (total_tokens,) 每个 token 的物理 slot 绝对编号
    # 源 tensor 的 stride（处理非连续内存）
    stride_k_token, stride_k_head, stride_k_dim,
    stride_v_token, stride_v_head, stride_v_dim,
    total_tokens,              # 运行时变量（虽然现在不用，留着备用）
    N_KV_HEADS: tl.constexpr, # ← constexpr
    BLOCK_SIZE: tl.constexpr, # ← constexpr
    HEAD_DIM: tl.constexpr,
):
    # 1. 映射 2D Grid
    pid_token = tl.program_id(0)
    pid_head = tl.program_id(1)

    # 2. 读取物理 slot 并解析 block_id 和 offset
    slot = tl.load(slot_mapping_ptr + pid_token)
    block_id = slot // BLOCK_SIZE
    offset = slot % BLOCK_SIZE

    # 3. 用 stride 计算源数据 (K/V) 的一维内存偏移，不假设输入连续
    dim_offsets = tl.arange(0, HEAD_DIM)
    src_offset = pid_token * stride_k_token \
               + pid_head  * stride_k_head  \
               + dim_offsets * stride_k_dim

    k_vec = tl.load(k_ptr + src_offset)
    v_src = pid_token * stride_v_token \
          + pid_head  * stride_v_head  \
          + dim_offsets * stride_v_dim
    v_vec = tl.load(v_ptr + v_src)
    
    # 4. 计算目标数据 (KV Cache) 的一维内存偏移
    # 基于你的形状 (num_blocks, n_kv_heads, block_size, HEAD_DIM)
    # 因为切片后的后4个维度是严格连续存放的，我们可以直接用乘法算出 strides
    dst_offset = (block_id * N_KV_HEADS * BLOCK_SIZE * HEAD_DIM) + \
                 (pid_head * BLOCK_SIZE * HEAD_DIM) + \
                 (offset * HEAD_DIM) + \
                 dim_offsets

    # 5. 写入 Cache
    tl.store(k_cache_ptr + dst_offset, k_vec)
    tl.store(v_cache_ptr + dst_offset, v_vec)

@torch.compiler.disable
def store_kvcache(k, v, k_cache, v_cache, slot_mapping, block_size=16):
    """
    k, v:           (total_tokens, n_kv_heads, head_dim)
    k_cache, v_cache: (num_blocks, n_kv_heads, block_size, head_dim)
    slot_mapping:   (total_tokens,) int32
    """
    total_tokens, N_KV_HEADS, HEAD_DIM = k.shape
    grid = (total_tokens, N_KV_HEADS)
    store_kvcache_kernel[grid](
        k, v, k_cache, v_cache, slot_mapping,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        total_tokens,
        N_KV_HEADS=N_KV_HEADS,
        BLOCK_SIZE=block_size,
        HEAD_DIM=HEAD_DIM,
    )


if __name__ == "__main__":
    import torch
    device = 'cuda'
    dtype = torch.bfloat16
    BLOCK_SIZE = 16
    N_KV_HEADS = 2
    HEAD_DIM = 128
    NUM_BLOCKS = 32
    TOTAL_TOKENS = 5  # 模拟 prefill 5 个 token

    # 初始化 kv_cache
    k_cache = torch.zeros(NUM_BLOCKS, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM,
                          device=device, dtype=dtype)
    v_cache = torch.zeros(NUM_BLOCKS, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM,
                          device=device, dtype=dtype)

    # 构造测试 k/v
    k = torch.randn(TOTAL_TOKENS, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(TOTAL_TOKENS, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)

    # slot_mapping: token i 写入 slot i（block 0 的前 5 个位置）
    slot_mapping = torch.arange(TOTAL_TOKENS, dtype=torch.int32, device=device)

    # 运行 kernel
    store_kvcache(k, v, k_cache, v_cache, slot_mapping, BLOCK_SIZE)

    # 验证：k_cache[block=0, :, offset=i, :] == k[i]
    for i in range(TOTAL_TOKENS):
        block_id = i // BLOCK_SIZE  # = 0
        offset = i % BLOCK_SIZE     # = i
        assert torch.allclose(k_cache[block_id, :, offset, :], k[i], atol=1e-3), \
            f"token {i} k 不一致"
        assert torch.allclose(v_cache[block_id, :, offset, :], v[i], atol=1e-3), \
            f"token {i} v 不一致"

    # 测试跨 block 的情况
    k2 = torch.randn(TOTAL_TOKENS, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v2 = torch.randn(TOTAL_TOKENS, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    # slot 14, 15, 16, 17, 18（跨 block 0 和 block 1）
    slot_mapping2 = torch.tensor([14, 15, 16, 17, 18], dtype=torch.int32, device=device)
    store_kvcache(k2, v2, k_cache, v_cache, slot_mapping2, BLOCK_SIZE)

    for i, slot in enumerate(slot_mapping2.tolist()):
        block_id = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        assert torch.allclose(k_cache[block_id, :, offset, :], k2[i], atol=1e-3), \
            f"slot {slot} k 不一致"

    print("所有验证通过 ✓")