import torch
from torch import Tensor
import triton
import triton.language as tl

@triton.jit
def Decode_Paged_GQAAttention_Kernel(
        q                         ,  # (B, n_heads, 1, head_dim)         query，decode 每步只有 1 个 token
        k_cache                   ,  # (num_blocks, n_kv_heads, block_size, head_dim)  全局 K cache
        v_cache                   ,  # (num_blocks, n_kv_heads, block_size, head_dim)  全局 V cache
        block_table               ,  # (B, MAX_BLOCKS_PER_SEQ)  int32，每个请求的物理块 id，-1 表示未分配
        context_lens              ,  # (B,)             int32，每个请求当前的有效 token 数
        scale              ,         # 1.0 / sqrt(head_dim)
        out,
        
        # Meta-parameters
        # 元参数
        MAX_BLOCKS_PER_SEQ: tl.constexpr,  # 启动时固定，不是运行时变量
        BLOCK_SIZE: tl.constexpr,  #
        HEAD_DIM: tl.constexpr,  #
        N_KV_HEAD: tl.constexpr,
        N_HEAD: tl.constexpr,
    ):               # (B, n_heads, 1, head_dim)
    # grid = (B, n_heads)
    # 每个 program 处理一个 (batch, head) 对
    # program_id[0] = batch_idx
    # program_id[1] = head_idx
    
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    kv_head_idx = (pid_head // (N_HEAD // N_KV_HEAD))

    m = float('-inf')
    l = float(0)
    o = tl.zeros((HEAD_DIM, ), dtype=tl.float32)
    
    
    q_ptrs = q + pid_batch * (HEAD_DIM * N_HEAD) + pid_head * (HEAD_DIM) + tl.arange(0, HEAD_DIM)
    q_vec = tl.reshape(tl.load(q_ptrs), (HEAD_DIM, 1))

    context_len = tl.load(context_lens + pid_batch)
    if context_len == 0:
        return
    max_block_index = tl.cdiv(context_len, BLOCK_SIZE)  # 向上取整
    offs_kv = tl.arange(0, BLOCK_SIZE * HEAD_DIM)
    
    for block_idx in range(0, max_block_index):
        
        physical_idx = tl.load(block_table + pid_batch * MAX_BLOCKS_PER_SEQ + block_idx)
        physical_idx = tl.maximum(physical_idx, 0).to(tl.int64)
        base = (physical_idx * N_KV_HEAD * BLOCK_SIZE * HEAD_DIM + kv_head_idx* BLOCK_SIZE * HEAD_DIM)

        # 加载时 mask 掉超出 context_len 的 token
        token_start = block_idx * BLOCK_SIZE
        valid_in_block = tl.minimum(BLOCK_SIZE, context_len - token_start)
        kv_token_mask = tl.arange(0, BLOCK_SIZE * HEAD_DIM) < valid_in_block * HEAD_DIM

        k_ptrs = k_cache + base + offs_kv
        # k_block: (block_size, head_dim)
        k_block = tl.load(k_ptrs, mask = kv_token_mask, other=0.0)
        k_block = tl.reshape(k_block, (BLOCK_SIZE, HEAD_DIM))
        v_ptrs = v_cache + base + offs_kv
        v_block = tl.load(v_ptrs, mask = kv_token_mask, other=0.0)
        v_block = tl.reshape(v_block, (BLOCK_SIZE, HEAD_DIM))

        # mask 最后一个 block 的无效 token
        valid = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < context_len
        q_row = tl.reshape(q_vec, (1, HEAD_DIM))           # (1, HEAD_DIM)
        s = tl.sum(k_block * q_row, axis=1)                 # (BLOCK_SIZE,)
        s = s.to(tl.float32) * scale
        s = tl.where(valid, s, float('-inf'))

        m_new = tl.maximum(m, tl.max(s))
        alpha = tl.exp(m - m_new)           # 旧的缩放因子
        p = tl.exp(s - m_new)          # 当前 block 的权重

        l = l * alpha + tl.sum(p)
        p_col = tl.reshape(p, (BLOCK_SIZE, 1))             # (BLOCK_SIZE, 1)
        o = o * alpha + tl.sum(p_col * v_block, axis=0)    # (HEAD_DIM,)
        m = m_new

    o = o / l  # (HEAD_DIM,)
    o_casted = tl.cast(o, out.dtype.element_ty)
    
    tl.store(out + pid_batch * (HEAD_DIM * N_HEAD) + pid_head * (HEAD_DIM) + tl.arange(0, HEAD_DIM), o_casted)

@torch.compiler.disable
def paged_decode_attention(q, k_cache, v_cache, block_table, context_lens,
                           MAX_BLOCKS_PER_SEQ, BLOCK_SIZE=16):
    B, N_HEAD, _, HEAD_DIM = q.shape
    N_KV_HEAD = k_cache.shape[1]
    scale = 1.0 / (HEAD_DIM ** 0.5)

    out = torch.empty(B, N_HEAD, 1, HEAD_DIM, device=q.device, dtype=q.dtype)

    grid = (B, N_HEAD)
    Decode_Paged_GQAAttention_Kernel[grid](
        q, k_cache, v_cache, block_table, context_lens,
        scale, out,
        MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
        BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=HEAD_DIM,
        N_KV_HEAD=N_KV_HEAD,
        N_HEAD=N_HEAD,
    )
    return out

@triton.jit
def Prefill_Paged_GQAAttention_Kernel(
    q,                          # (total_new_tokens, n_heads, head_dim)
    k_cache,                    # (num_blocks, n_kv_heads, block_size, head_dim)
    v_cache,
    block_table,                # (B, MAX_BLOCKS_PER_SEQ) int32
    context_lens,               # (B,) int32，总长度 = prefix_len + new_len
    new_token_lens,             # (B,) int32，本次 prefill 的新 token 数 M
    q_start_loc,                # (B,) int32，每个 batch 的 Q 在 q tensor 里的起始 offset
    scale,
    out,                        # (total_new_tokens, n_heads, head_dim)

    MAX_BLOCKS_PER_SEQ: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,      # Q tile 大小（一般 = BLOCK_SIZE）
    HEAD_DIM: tl.constexpr,
    N_KV_HEAD: tl.constexpr,
    N_HEAD: tl.constexpr,
):
    """
    Grid: (B, num_q_blocks_max, N_HEAD)
      num_q_blocks_max = cdiv(max_new_tokens_in_batch, BLOCK_M)
      超出本 batch 实际 Q 数量的 program 直接 return
    
    每个 program 处理：
      batch pid_batch 的第 pid_q_block 个 Q tile 的 pid_head 头
      输出一个 (BLOCK_M, HEAD_DIM) 的 tile
    """
    pid_batch = tl.program_id(0)
    pid_q_block = tl.program_id(1)
    pid_head = tl.program_id(2)
    kv_head_idx = pid_head // (N_HEAD // N_KV_HEAD)

    # 本 batch 的元数据
    new_len = tl.load(new_token_lens + pid_batch)         # M
    total_len = tl.load(context_lens + pid_batch)          # prefix_len + M
    prefix_len = total_len - new_len
    q_offset = tl.load(q_start_loc + pid_batch)            # Q 在全局 tensor 里的起始

    # 本 program 负责的 Q tile 范围（在本 batch 的 M 个 Q 里）
    q_tile_start = pid_q_block * BLOCK_M
    if q_tile_start >= new_len:
        return  # 超出本 batch 的 Q，直接退出

    # 初始化 online softmax
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # 加载 Q tile: (BLOCK_M, HEAD_DIM)
    q_row_idx = q_offset + q_tile_start + tl.arange(0, BLOCK_M)
    q_row_mask = tl.arange(0, BLOCK_M) < (new_len - q_tile_start)
    q_ptrs = q + q_row_idx[:, None] * (N_HEAD * HEAD_DIM) + pid_head * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    q_tile = tl.load(q_ptrs, mask=q_row_mask[:, None], other=0.0)  # (BLOCK_M, HEAD_DIM)

    # Q 在全局序列里的位置（用于 causal mask）
    q_pos_global = prefix_len + q_tile_start + tl.arange(0, BLOCK_M)  # (BLOCK_M,)

    # 遍历所有 KV block（cache 里已经包含了新写入的 K/V）
    num_kv_blocks = tl.cdiv(total_len, BLOCK_SIZE)

    for block_idx in range(0, num_kv_blocks):
        physical_idx = tl.load(block_table + pid_batch * MAX_BLOCKS_PER_SEQ + block_idx)
        physical_idx = tl.maximum(physical_idx, 0).to(tl.int64)
        base = physical_idx * N_KV_HEAD * BLOCK_SIZE * HEAD_DIM + kv_head_idx * BLOCK_SIZE * HEAD_DIM

        # 加载 K block: (BLOCK_SIZE, HEAD_DIM)
        k_token_start = block_idx * BLOCK_SIZE
        valid_in_block = tl.minimum(BLOCK_SIZE, total_len - k_token_start)

        offs_kv_flat = tl.arange(0, BLOCK_SIZE * HEAD_DIM)
        kv_token_mask = offs_kv_flat < valid_in_block * HEAD_DIM

        k_ptrs = k_cache + base + offs_kv_flat
        k_block = tl.load(k_ptrs, mask=kv_token_mask, other=0.0)
        k_block = tl.reshape(k_block, (BLOCK_SIZE, HEAD_DIM))

        v_ptrs = v_cache + base + offs_kv_flat
        v_block = tl.load(v_ptrs, mask=kv_token_mask, other=0.0)
        v_block = tl.reshape(v_block, (BLOCK_SIZE, HEAD_DIM))

        # s = Q @ K^T: (BLOCK_M, BLOCK_SIZE)
        s = tl.dot(q_tile, tl.trans(k_block))
        s = s * scale

        # causal mask: q_pos >= k_pos
        k_pos_global = k_token_start + tl.arange(0, BLOCK_SIZE)  # (BLOCK_SIZE,)
        valid_k = k_pos_global < total_len                        # block 末尾的无效 token
        causal = q_pos_global[:, None] >= k_pos_global[None, :]   # (BLOCK_M, BLOCK_SIZE)
        mask = causal & valid_k[None, :]
        s = tl.where(mask, s, float('-inf'))

        # online softmax 更新
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])                            # (BLOCK_M, BLOCK_SIZE)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_i = o_i * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
        m_i = m_new

    # 归一化并写回
    o_i = o_i / l_i[:, None]

    out_ptrs = out + q_row_idx[:, None] * (N_HEAD * HEAD_DIM) + pid_head * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    tl.store(out_ptrs, o_i.to(out.dtype.element_ty), mask=q_row_mask[:, None])


@torch.compiler.disable
def paged_prefill_attention(
    q,                  # (total_new_tokens, n_heads, head_dim)
    k_cache, v_cache,
    block_table,        # (B, MAX_BLOCKS_PER_SEQ)
    context_lens,       # (B,) 总长度（prefix + new）
    new_token_lens,     # (B,) 本次 prefill 的 new token 数
    q_start_loc,        # (B,) Q 在 q tensor 里的起始 offset
    MAX_BLOCKS_PER_SEQ,
    BLOCK_SIZE=16,
    BLOCK_M=16,
):
    total_new_tokens, N_HEAD, HEAD_DIM = q.shape
    N_KV_HEAD = k_cache.shape[1]
    B = context_lens.shape[0]
    scale = 1.0 / (HEAD_DIM ** 0.5)

    out = torch.empty_like(q)

    # num_q_blocks_max = 本 batch 中最长的 new_len 对应的 tile 数
    max_new_len = int(new_token_lens.max().item())
    num_q_blocks = (max_new_len + BLOCK_M - 1) // BLOCK_M

    grid = (B, num_q_blocks, N_HEAD)
    Prefill_Paged_GQAAttention_Kernel[grid](
        q, k_cache, v_cache, block_table,
        context_lens, new_token_lens, q_start_loc,
        scale, out,
        MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_M=BLOCK_M,
        HEAD_DIM=HEAD_DIM,
        N_KV_HEAD=N_KV_HEAD,
        N_HEAD=N_HEAD,
    )
    return out

if __name__ == "__main__":
    import torch
    from cache import BlockManager, PagedKVCache, pagedblocktype
    from model import ModelConfig
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'

    cfg = ModelConfig()
    device = 'cuda'
    dtype = torch.bfloat16
    BLOCK_SIZE = 16
    MAX_BLOCKS = 64

    bm = BlockManager(
        num_gpu_blocks=256, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=1,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, dtype=dtype,
    )

    print(bm.gpu_kv_cache[0, 0].is_contiguous())
    print(bm.gpu_kv_cache[0, 0].stride())
    print(bm.gpu_kv_cache[0, 0].shape)
    # 构造随机 KV cache，模拟 seq_len=48 的历史
    SEQ_LEN = 48
    B = 2

    caches = []
    for b in range(B):
        c = PagedKVCache(bm, num_layers=1, max_seq_len=128,
                         num_kv_heads=cfg.num_key_value_heads,
                         head_dim=cfg.head_dim, device=device, dtype=dtype)
        # 填入随机 KV
        k_rand = torch.randn(SEQ_LEN, cfg.num_key_value_heads, cfg.head_dim,
                             device=device, dtype=dtype)
        v_rand = torch.randn(SEQ_LEN, cfg.num_key_value_heads, cfg.head_dim,
                             device=device, dtype=dtype)
        c.prefill_update(torch.tensor(0), k_rand, v_rand)
        # decode 一步，写入新 token
        c.prepare_decode_step()
        k_new = torch.randn(1, cfg.num_key_value_heads, cfg.head_dim,
                            device=device, dtype=dtype)
        v_new = torch.randn(1, cfg.num_key_value_heads, cfg.head_dim,
                            device=device, dtype=dtype)
        c.update(torch.tensor(0), k_new.squeeze(0), v_new.squeeze(0))
        caches.append(c)

    # 构造 q
    q = torch.randn(B, cfg.num_attention_heads, 1, cfg.head_dim,
                    device=device, dtype=dtype)

    # 构造 block_table 和 context_lens
    context_lens = torch.tensor([c.seq_len for c in caches],
                                 dtype=torch.int32, device=device)
    block_table = torch.full((B, MAX_BLOCKS), -1, dtype=torch.int32, device=device)
    for i, c in enumerate(caches):
        bt = c.get_block_table()
        block_table[i, :len(bt)] = bt

    print(f"context_lens: {context_lens}")
    print(f"block_table:\n{block_table}")
    for i in range(B):
        n_blocks = (context_lens[i].item() + BLOCK_SIZE - 1) // BLOCK_SIZE
        print(f"请求{i}有效块: {block_table[i, :n_blocks]}")

    # Triton kernel 输出
    k_cache_layer = bm.gpu_kv_cache[0, 0]  # (num_blocks, n_kv_heads, block_size, head_dim)
    v_cache_layer = bm.gpu_kv_cache[1, 0]
    # 加这行检查
    print(f"k_cache contiguous: {k_cache_layer.is_contiguous()}")
    k_cache_layer = k_cache_layer.contiguous()  # 确保连续
    v_cache_layer = v_cache_layer.contiguous()
    triton_out = paged_decode_attention(
        q, k_cache_layer, v_cache_layer,
        block_table, context_lens,
        MAX_BLOCKS_PER_SEQ=MAX_BLOCKS,
    )

    # gather 版本对比（用 F.scaled_dot_product_attention）
    scale = 1.0 / (cfg.head_dim ** 0.5)
    gather_outputs = []
    for i in range(B):
        seq_len = context_lens[i].item()
        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        phys = block_table[i, :num_blocks]
        k_i = k_cache_layer[phys].permute(1, 0, 2, 3) \
               .reshape(cfg.num_key_value_heads, -1, cfg.head_dim)[:, :seq_len] \
               .repeat_interleave(cfg.num_kv_groups, dim=0).unsqueeze(0)
        v_i = v_cache_layer[phys].permute(1, 0, 2, 3) \
               .reshape(cfg.num_key_value_heads, -1, cfg.head_dim)[:, :seq_len] \
               .repeat_interleave(cfg.num_kv_groups, dim=0).unsqueeze(0)
        q_i = q[i].unsqueeze(0)
        out_i = torch.nn.functional.scaled_dot_product_attention(
            q_i, k_i, v_i, is_causal=False)
        gather_outputs.append(out_i)
    gather_out = torch.cat(gather_outputs, dim=0)

    diff = (triton_out.float() - gather_out.float()).abs()
    print(f"最大误差: {diff.max().item():.4f}")
    print(f"平均误差: {diff.mean().item():.4f}")
    print("通过!" if diff.max().item() < 0.1 else "❌ 误差过大")

    for c in caches:
        c.reset()