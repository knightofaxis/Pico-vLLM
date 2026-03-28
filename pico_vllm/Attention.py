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
        # kv_mask = token_mask[:, None] & (tl.arange(0, HEAD_DIM)[None, :] < HEAD_DIM)
        # # 展平给 load 用
        # kv_mask_flat = tl.reshape(kv_mask, (BLOCK_SIZE * HEAD_DIM,))

        k_ptrs = k_cache + base + offs_kv
        # # 运行时打印值（只在 pid=0 时打印，避免刷屏）
        # if pid_batch == 0 and pid_head == 0:
        #     tl.device_print("block_idx: ", block_idx)
        #     tl.device_print("physical_idx: ", physical_idx)
        #     tl.device_print("base: ", base)
        #     tl.device_print("context_len: ", context_len)
        # k_block: (block_size, head_dim)
        k_block = tl.load(k_ptrs, mask = kv_token_mask, other=0.0)
        # k_block = tl.zeros((BLOCK_SIZE * HEAD_DIM,), dtype=tl.bfloat16)
        k_block = tl.reshape(k_block, (BLOCK_SIZE, HEAD_DIM))
        v_ptrs = v_cache + base + offs_kv
        v_block = tl.load(v_ptrs, mask = kv_token_mask, other=0.0)
        # v_block = tl.zeros((BLOCK_SIZE * HEAD_DIM,), dtype=tl.bfloat16)
        v_block = tl.reshape(v_block, (BLOCK_SIZE, HEAD_DIM))

        # # 编译期打印类型（不需要运行，编译时就输出）
        # tl.static_print("physical_idx dtype:", physical_idx.dtype)
        # tl.static_print("base dtype:", base.dtype)
        # tl.static_print("k_cache dtype:", k_cache.dtype)
        # tl.static_print("offs_kv dtype:", offs_kv.dtype)
        # tl.static_print("k_ptrs dtype:", k_ptrs.dtype)

        # mask 最后一个 block 的无效 token
        valid = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < context_len
        # s = tl.dot(k_block, q_vec)
        q_row = tl.reshape(q_vec, (1, HEAD_DIM))           # (1, HEAD_DIM)
        s = tl.sum(k_block * q_row, axis=1)                 # (BLOCK_SIZE,)
        s = s.to(tl.float32) * scale
        # s = tl.reshape(s, (BLOCK_SIZE, )) * scale
        s = tl.where(valid, s, float('-inf'))

        m_new = tl.maximum(m, tl.max(s))
        alpha = tl.exp(m - m_new)           # 旧的缩放因子
        p = tl.exp(s - m_new)          # 当前 block 的权重

        l = l * alpha + tl.sum(p)
        # o = o * alpha + tl.dot(tl.reshape(p, (1, BLOCK_SIZE)), tl.cast(v_block ,tl.float32))
        p_col = tl.reshape(p, (BLOCK_SIZE, 1))             # (BLOCK_SIZE, 1)
        o = o * alpha + tl.sum(p_col * v_block, axis=0)    # (HEAD_DIM,)
        m = m_new

    o = o / l  # (1, HEAD_DIM)
    o_casted = tl.cast(o, out.dtype.element_ty)
    
    tl.store(out + pid_batch * (HEAD_DIM * N_HEAD) + pid_head * (HEAD_DIM) + tl.arange(0, HEAD_DIM), o_casted)

def paged_decode_attention(q, k_cache, v_cache, block_table, context_lens,
                           MAX_BLOCKS_PER_SEQ, BLOCK_SIZE=16):
    B, _, N_HEAD, HEAD_DIM = q.shape
    N_KV_HEAD = k_cache.shape[1]
    scale = 1.0 / (HEAD_DIM ** 0.5)

    out = torch.empty(B, N_HEAD, 1, HEAD_DIM, device=q.device, dtype=q.dtype)

    # k_cache 的最大合法地址
    max_physical_id = block_table.max().item()
    max_offset = max_physical_id * N_KV_HEAD * BLOCK_SIZE * HEAD_DIM
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
        c.prefill_update(0, k_rand, v_rand)
        # decode 一步，写入新 token
        c.prepare_decode_step()
        k_new = torch.randn(1, cfg.num_key_value_heads, cfg.head_dim,
                            device=device, dtype=dtype)
        v_new = torch.randn(1, cfg.num_key_value_heads, cfg.head_dim,
                            device=device, dtype=dtype)
        c.update(0, k_new.squeeze(0), v_new.squeeze(0))
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