import torch
import triton
import triton.language as tl

@triton.jit
def _fused_decode_rope_and_cache_kernel(
    # --- Pointers ---
    q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr,
    q_rot_ptr, k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    context_lens_ptr,  # 传入 context_lens 的指针
    
    # --- Strides ---
    stride_q_tok, stride_q_h, stride_q_d,
    stride_k_tok, stride_k_h, stride_k_d,
    stride_v_tok, stride_v_h, stride_v_d,
    stride_cos_tok, stride_cos_d,
    stride_sin_tok, stride_sin_d,
    stride_q_rot_tok, stride_q_rot_h, stride_q_rot_d,
    stride_kc_blk, stride_kc_h, stride_kc_seq, stride_kc_d,
    stride_vc_blk, stride_vc_h, stride_vc_seq, stride_vc_d,
    
    # --- Dimensions ---
    num_q_heads: tl.constexpr, 
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr, 
    block_size: tl.constexpr,
):
    """
    Grid: (total_tokens * num_q_heads,)
    每个 program 处理 1 个 token 的 1 个 Q head。
    附带处理 K 和 V 的写入（仅当 head_idx < num_kv_heads 时）。
    """
    pid = tl.program_id(0)
    tok_idx = pid // num_q_heads
    q_head_idx = pid % num_q_heads

    ctx_len = tl.load(context_lens_ptr + tok_idx)
    if ctx_len == 0:
            return
    
    # 计算一半维度的 offset，用于 RoPE 旋转
    HALF_DIM: tl.constexpr = head_dim // 2
    d_offsets_1 = tl.arange(0, HALF_DIM)
    d_offsets_2 = HALF_DIM + tl.arange(0, HALF_DIM)
    d_offsets = tl.arange(0, head_dim)

    # =======================================================
    # 1. 提取 Cos 和 Sin
    # =======================================================
    cos_1 = tl.load(cos_ptr + tok_idx * stride_cos_tok + d_offsets_1)
    cos_2 = tl.load(cos_ptr + tok_idx * stride_cos_tok + d_offsets_2)
    sin_1 = tl.load(sin_ptr + tok_idx * stride_sin_tok + d_offsets_1)
    sin_2 = tl.load(sin_ptr + tok_idx * stride_sin_tok + d_offsets_2)

    # =======================================================
    # 2. 计算 Q 的 RoPE 并写回连续显存
    # =======================================================
    q_base = q_ptr + tok_idx * stride_q_tok + q_head_idx * stride_q_h
    q_1 = tl.load(q_base + d_offsets_1)
    q_2 = tl.load(q_base + d_offsets_2)

    # Qwen 的 concat style 旋转: [-x2, x1]
    q_rot_1 = q_1 * cos_1 - q_2 * sin_1
    q_rot_2 = q_2 * cos_2 + q_1 * sin_2

    q_rot_base = q_rot_ptr + tok_idx * stride_q_rot_tok + q_head_idx * stride_q_rot_h
    tl.store(q_rot_base + d_offsets_1, q_rot_1)
    tl.store(q_rot_base + d_offsets_2, q_rot_2)

    # =======================================================
    # 3. 计算 K 的 RoPE，并将 KV 写入 Paged KV Cache
    # =======================================================
    # 只有前 num_kv_heads 个 block 负责写入 KV，以兼容 GQA
    if q_head_idx < num_kv_heads:
        kv_head_idx = q_head_idx
        
        # 读取 Cache 物理块的索引和偏移
        slot_idx = tl.load(slot_mapping_ptr + tok_idx)
        blk_idx = slot_idx // block_size
        blk_off = slot_idx % block_size

        # ---- 处理 K ----
        k_base = k_ptr + tok_idx * stride_k_tok + kv_head_idx * stride_k_h
        k_1 = tl.load(k_base + d_offsets_1)
        k_2 = tl.load(k_base + d_offsets_2)

        k_rot_1 = k_1 * cos_1 - k_2 * sin_1
        k_rot_2 = k_2 * cos_2 + k_1 * sin_2

        # 写入 Paged Cache
        kc_base = k_cache_ptr + blk_idx * stride_kc_blk + kv_head_idx * stride_kc_h + blk_off * stride_kc_seq
        tl.store(kc_base + d_offsets_1, k_rot_1)
        tl.store(kc_base + d_offsets_2, k_rot_2)

        # ---- 处理 V ----
        v_base = v_ptr + tok_idx * stride_v_tok + kv_head_idx * stride_v_h
        v = tl.load(v_base + d_offsets)
        
        vc_base = v_cache_ptr + blk_idx * stride_vc_blk + kv_head_idx * stride_vc_h + blk_off * stride_vc_seq
        tl.store(vc_base + d_offsets, v)


def fused_decode_rope_and_cache(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    kv_cache_k: torch.Tensor, kv_cache_v: torch.Tensor,
    slot_mapping: torch.Tensor,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """
    输入:
        q, k, v: 形状为 (B, seq_len, num_heads, head_dim) 的原始切分投影
        cos, sin: 形状为 (B, seq_len, head_dim)
        kv_cache: 形状为 (num_blocks, num_kv_heads, block_size, head_dim)
        slot_mapping: (B * seq_len,) 的 int32 一维张量
    输出:
        q_rot: 形状为 (B, seq_len, num_q_heads, head_dim)，供 Attention 继续使用
        K 和 V 同时写入 kv_cache
    """
    B, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    block_size = kv_cache_k.shape[2]
    
    total_tokens = B * seq_len

    # 将所有输入展平为
    q_2d = q.view(total_tokens, num_q_heads, head_dim)
    k_2d = k.view(total_tokens, num_kv_heads, head_dim)
    v_2d = v.view(total_tokens, num_kv_heads, head_dim)
    cos_2d = cos.view(total_tokens, head_dim)
    sin_2d = sin.view(total_tokens, head_dim)
    slot_mapping_1d = slot_mapping.view(-1)

    # 预分配 q_rot，在 CUDA Graph 下会被缓存池接管，无开销
    q_rot_2d = torch.empty_like(q_2d)

    # 启动 1D 网格
    grid = (total_tokens * num_q_heads,)

    _fused_decode_rope_and_cache_kernel[grid](
        q_2d, k_2d, v_2d, cos_2d, sin_2d,
        q_rot_2d, kv_cache_k, kv_cache_v,
        slot_mapping_1d,
        context_lens,
        
        q_2d.stride(0), q_2d.stride(1), q_2d.stride(2),
        k_2d.stride(0), k_2d.stride(1), k_2d.stride(2),
        v_2d.stride(0), v_2d.stride(1), v_2d.stride(2),
        cos_2d.stride(0), cos_2d.stride(1),
        sin_2d.stride(0), sin_2d.stride(1),
        q_rot_2d.stride(0), q_rot_2d.stride(1), q_rot_2d.stride(2),
        kv_cache_k.stride(0), kv_cache_k.stride(1), kv_cache_k.stride(2), kv_cache_k.stride(3),
        kv_cache_v.stride(0), kv_cache_v.stride(1), kv_cache_v.stride(2), kv_cache_v.stride(3),
        
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )

    # 恢复原有的形状返回给 Attention
    return q_rot_2d.view(B, seq_len, num_q_heads, head_dim)