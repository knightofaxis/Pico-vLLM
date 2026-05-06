import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# test_prefill_paged_attention.py
import torch
import torch.nn.functional as F
from ops.triton.attention import paged_prefill_attention

BLOCK_SIZE = 16

def ref_attention(q, k, v, is_causal=True):
    """
    参考实现：标准 SDPA
    q: (M, n_heads, head_dim)
    k: (N, n_kv_heads, head_dim)
    v: (N, n_kv_heads, head_dim)
    返回: (M, n_heads, head_dim)
    
    GQA：n_kv_heads 可以小于 n_heads，K/V 需要 repeat
    Causal：Q[m] 只能看 K[:prefix_len + m + 1]，但这里 q 和 k 的长度可能不同
    """
    M, n_heads, head_dim = q.shape
    N, n_kv_heads, _ = k.shape
    kv_groups = n_heads // n_kv_heads
    
    # K, V repeat 到 n_heads
    k = k.repeat_interleave(kv_groups, dim=1)  # (N, n_heads, head_dim)
    v = v.repeat_interleave(kv_groups, dim=1)
    
    # (n_heads, M, head_dim), (n_heads, N, head_dim)
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    
    # (n_heads, M, N)
    scale = 1.0 / (head_dim ** 0.5)
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    
    if is_causal:
        # Q 的第 m 个对应全局位置 (N - M) + m，因为 Q 是序列的最后 M 个
        # K 的第 n 个对应全局位置 n
        # mask: (N - M) + m >= n  <=>  m >= n - (N - M)
        prefix_len = N - M
        q_pos = torch.arange(M, device=q.device) + prefix_len  # (M,)
        k_pos = torch.arange(N, device=q.device)                 # (N,)
        mask = q_pos[:, None] >= k_pos[None, :]                  # (M, N)
        s = s.masked_fill(~mask, float('-inf'))
    
    p = torch.softmax(s, dim=-1)
    out = torch.matmul(p, v)  # (n_heads, M, head_dim)
    return out.transpose(0, 1).contiguous()  # (M, n_heads, head_dim)


def allocate_blocks_and_store_kv(k_full, v_full, num_gpu_blocks, block_size):
    """
    把 (N, n_kv_heads, head_dim) 的完整 K/V 分散到 paged blocks。
    返回 k_cache, v_cache, block_table（给一个 batch 用）。
    """
    N, n_kv_heads, head_dim = k_full.shape
    num_needed = (N + block_size - 1) // block_size
    
    # 随机挑一些物理 block id
    torch.manual_seed(42)
    perm = torch.randperm(num_gpu_blocks)[:num_needed].tolist()
    
    k_cache = torch.zeros(num_gpu_blocks, n_kv_heads, block_size, head_dim,
                           dtype=k_full.dtype, device=k_full.device)
    v_cache = torch.zeros_like(k_cache)
    
    for i, phys_id in enumerate(perm):
        start = i * block_size
        end = min(start + block_size, N)
        length = end - start
        # k_full[start:end] shape (length, n_kv_heads, head_dim)
        # cache 布局 (n_kv_heads, block_size, head_dim)
        k_cache[phys_id, :, :length, :] = k_full[start:end].transpose(0, 1)
        v_cache[phys_id, :, :length, :] = v_full[start:end].transpose(0, 1)
    
    return k_cache, v_cache, perm


def test_case(name, B, new_lens, prefix_lens, n_heads=12, n_kv_heads=2, head_dim=128,
              num_gpu_blocks=200, MAX_BLOCKS_PER_SEQ=32, atol=1e-2, rtol=1e-2):
    """
    通用测试：构造 B 个 batch，每个有自己的 prefix_len 和 new_len，
    和参考实现对比。
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    print(f"\n=== {name} ===")
    print(f"  B={B}, new_lens={new_lens}, prefix_lens={prefix_lens}")
    
    torch.manual_seed(0)
    
    # 每个 batch 独立构造
    k_caches_list = []  # 占位，实际共用一个全局 k_cache
    
    # 全局 cache
    k_cache = torch.zeros(num_gpu_blocks, n_kv_heads, BLOCK_SIZE, head_dim,
                           dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    
    # block_table，B 个 batch
    block_table = torch.full((B, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32, device=device)
    
    # 所有 batch 的 Q 打平拼在一起
    q_all_parts = []
    q_start_loc_list = []
    ref_outs = []  # 每个 batch 的参考输出
    
    next_block_id = 0
    total_new = 0
    
    for b in range(B):
        new_len = new_lens[b]
        prefix_len = prefix_lens[b]
        total_len = prefix_len + new_len
        
        # 构造 batch b 的完整 Q / K / V
        q_full = torch.randn(new_len, n_heads, head_dim, dtype=dtype, device=device) * 0.1
        k_full = torch.randn(total_len, n_kv_heads, head_dim, dtype=dtype, device=device) * 0.1
        v_full = torch.randn(total_len, n_kv_heads, head_dim, dtype=dtype, device=device) * 0.1
        
        # 参考实现：Q 对完整 K/V 做 attention
        ref_out = ref_attention(q_full, k_full, v_full, is_causal=True)
        ref_outs.append(ref_out)
        
        # 分配 block 并把完整 K/V 写入 cache
        num_blocks_b = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        phys_ids = list(range(next_block_id, next_block_id + num_blocks_b))
        next_block_id += num_blocks_b
        
        for i, phys_id in enumerate(phys_ids):
            start = i * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, total_len)
            length = end - start
            k_cache[phys_id, :, :length, :] = k_full[start:end].transpose(0, 1)
            v_cache[phys_id, :, :length, :] = v_full[start:end].transpose(0, 1)
            block_table[b, i] = phys_id
        
        q_all_parts.append(q_full)
        q_start_loc_list.append(total_new)
        total_new += new_len
    
    # 打平 Q
    q_all = torch.cat(q_all_parts, dim=0)  # (total_new, n_heads, head_dim)
    
    context_lens = torch.tensor([prefix_lens[b] + new_lens[b] for b in range(B)],
                                 dtype=torch.int32, device=device)
    new_token_lens = torch.tensor(new_lens, dtype=torch.int32, device=device)
    q_start_loc = torch.tensor(q_start_loc_list, dtype=torch.int32, device=device)
    
    # kernel 调用
    out = paged_prefill_attention(
        q_all, k_cache, v_cache, block_table,
        context_lens, new_token_lens, q_start_loc,
        MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_M=BLOCK_SIZE,
    )
    
    # 对比每个 batch
    all_pass = True
    for b in range(B):
        start = q_start_loc_list[b]
        end = start + new_lens[b]
        out_b = out[start:end]
        ref_b = ref_outs[b]
        
        max_diff = (out_b.float() - ref_b.float()).abs().max().item()
        mean_diff = (out_b.float() - ref_b.float()).abs().mean().item()
        
        close = torch.allclose(out_b.float(), ref_b.float(), atol=atol, rtol=rtol)
        status = "✅" if close else "❌"
        print(f"  {status} batch {b}: new_len={new_lens[b]}, prefix_len={prefix_lens[b]}, "
              f"max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}")
        
        if not close:
            all_pass = False
    
    return all_pass


def test_1_no_prefix_single_batch():
    """完整 prefill，无 prefix，B=1，等价于传统 prefill"""
    return test_case("1. 无 prefix，B=1，new_len=32",
                     B=1, new_lens=[32], prefix_lens=[0])


def test_2_no_prefix_aligned():
    """无 prefix，new_len 对齐 block_size"""
    return test_case("2. 无 prefix，new_len=64 (对齐)",
                     B=1, new_lens=[64], prefix_lens=[0])


def test_3_no_prefix_unaligned():
    """无 prefix，new_len 不对齐 block_size"""
    return test_case("3. 无 prefix，new_len=37 (不对齐)",
                     B=1, new_lens=[37], prefix_lens=[0])


def test_4_with_prefix_aligned():
    """有 prefix，都对齐"""
    return test_case("4. prefix_len=32, new_len=16 (都对齐)",
                     B=1, new_lens=[16], prefix_lens=[32])


def test_5_with_prefix_unaligned_prefix():
    """prefix_len 不对齐"""
    return test_case("5. prefix_len=20, new_len=16 (prefix 不对齐)",
                     B=1, new_lens=[16], prefix_lens=[20])


def test_6_with_prefix_unaligned_both():
    """两个都不对齐"""
    return test_case("6. prefix_len=20, new_len=15 (都不对齐)",
                     B=1, new_lens=[15], prefix_lens=[20])


def test_7_long_prefix():
    """长 prefix"""
    return test_case("7. prefix_len=200, new_len=20",
                     B=1, new_lens=[20], prefix_lens=[200])


def test_8_single_new_token():
    """new_len=1，退化为 decode 的语义（但用 prefill kernel）"""
    return test_case("8. prefix_len=32, new_len=1",
                     B=1, new_lens=[1], prefix_lens=[32])


def test_9_batch_mixed():
    """多 batch 混合：不同 prefix_len 和 new_len"""
    return test_case("9. B=3, 混合长度",
                     B=3,
                     new_lens=[16, 32, 8],
                     prefix_lens=[0, 48, 20])


def test_10_batch_all_no_prefix():
    """多 batch 全部无 prefix（等价于普通 batch prefill）"""
    return test_case("10. B=4, 全无 prefix",
                     B=4,
                     new_lens=[16, 24, 32, 8],
                     prefix_lens=[0, 0, 0, 0])


def test_11_batch_all_with_prefix():
    """多 batch 全部有 prefix"""
    return test_case("11. B=4, 全部有 prefix",
                     B=4,
                     new_lens=[8, 16, 4, 12],
                     prefix_lens=[32, 48, 16, 64])


def test_12_large_scale():
    """大规模：长序列 + 大 batch"""
    return test_case("12. 大规模 B=4",
                     B=4,
                     new_lens=[64, 128, 48, 96],
                     prefix_lens=[100, 50, 200, 30])


if __name__ == "__main__":
    tests = [
        test_1_no_prefix_single_batch,
        test_2_no_prefix_aligned,
        test_3_no_prefix_unaligned,
        test_4_with_prefix_aligned,
        test_5_with_prefix_unaligned_prefix,
        test_6_with_prefix_unaligned_both,
        test_7_long_prefix,
        test_8_single_new_token,
        test_9_batch_mixed,
        test_10_batch_all_no_prefix,
        test_11_batch_all_with_prefix,
        test_12_large_scale,
    ]
    
    results = []
    for t in tests:
        try:
            passed = t()
            results.append((t.__name__, passed))
        except Exception as e:
            print(f"  ❌ {t.__name__} 抛出异常: {e}")
            results.append((t.__name__, False))
    
    print("\n" + "=" * 50)
    print("总结：")
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    all_pass = all(p for _, p in results)
    if all_pass:
        print("\n🎉 全部测试通过！")
    else:
        print(f"\n⚠️  {sum(1 for _, p in results if not p)} 个测试失败")
