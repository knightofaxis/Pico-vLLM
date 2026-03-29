# test_slot_mapping.py
import torch
from cache import BlockManager, PagedKVCache

device = 'cuda'
dtype = torch.bfloat16
BLOCK_SIZE = 16
N_LAYERS = 1
N_KV_HEADS = 2
HEAD_DIM = 4  # 小一点方便打印

bm = BlockManager(
    num_gpu_blocks=10, num_cpu_blocks=0,
    block_size=BLOCK_SIZE, num_layers=N_LAYERS,
    num_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, dtype=dtype,
)

cache = PagedKVCache(
    block_manager=bm, num_layers=N_LAYERS,
    max_seq_len=64, num_kv_heads=N_KV_HEADS,
    head_dim=HEAD_DIM, device=device, dtype=dtype,
)

# ============================================================
# 测试一：_allocate_for_prefill 后 cache_block_index 是否正确
# ============================================================
print("=== 测试一：block 分配 ===")
prefill_len = 5
cache._allocate_for_prefill(prefill_len)

print(f"allocated_cache_block_num: {cache.allocated_cache_block_num}")
print(f"cache_block_index[:1]: {cache.cache_block_index[:1]}")

logical_id = cache.cache_block_index[0]
if hasattr(logical_id, 'item'):
    logical_id = logical_id.item()
_, physical_id = bm.block_mapping[logical_id]
print(f"logical_id={logical_id}, physical_id={physical_id}")
# 预期：physical_id 是某个非负整数

# ============================================================
# 测试二：get_prefill_slot_mapping 返回的 slot 是否对应 physical_id
# ============================================================
print("\n=== 测试二：prefill slot_mapping ===")
slot_mapping = cache.get_prefill_slot_mapping(prefill_len)
print(f"slot_mapping: {slot_mapping.tolist()}")

# 验证：slot[i] // BLOCK_SIZE 应该等于 physical_id
for i, slot in enumerate(slot_mapping.tolist()):
    expected_block = physical_id  # 前5个token都在同一个block里
    expected_offset = i
    actual_block = slot // BLOCK_SIZE
    actual_offset = slot % BLOCK_SIZE
    ok = actual_block == expected_block and actual_offset == expected_offset
    print(f"  token {i}: slot={slot}, block={actual_block}, offset={actual_offset} "
          f"(expected block={expected_block}, offset={expected_offset}) {'✓' if ok else '✗'}")

# ============================================================
# 测试三：store_kvcache 写入后能否从正确位置读回
# ============================================================
print("\n=== 测试三：store_kvcache 写入验证 ===")
from store_kvcache import store_kvcache

# 构造已知 k/v
k = torch.arange(prefill_len * N_KV_HEADS * HEAD_DIM,
                 dtype=dtype, device=device).reshape(prefill_len, N_KV_HEADS, HEAD_DIM)
v = k * 2

store_kvcache(k, v, bm.gpu_kv_cache[0, 0], bm.gpu_kv_cache[1, 0], slot_mapping)

# 验证每个 token 写到了正确位置
for i, slot in enumerate(slot_mapping.tolist()):
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    k_readback = bm.gpu_kv_cache[0, 0, blk, :, off, :]  # (n_kv_heads, head_dim)
    k_expected = k[i]
    ok = torch.allclose(k_readback.float(), k_expected.float(), atol=1e-2)
    print(f"  token {i}: slot={slot} 写入一致={ok}")
    if not ok:
        print(f"    expected: {k_expected}")
        print(f"    got:      {k_readback}")

# ============================================================
# 测试四：手动更新 seq_len 后，get_decode_slot 是否指向下一个正确位置
# ============================================================
print("\n=== 测试四：decode slot ===")
cache._seq_len = prefill_len  # 模拟 prefill 完成
cache.prepare_decode_step()
decode_slot = cache.get_decode_slot()
print(f"decode_slot={decode_slot}")
print(f"expected block={decode_slot // BLOCK_SIZE}, offset={decode_slot % BLOCK_SIZE}")
# 预期：offset=5（紧接 prefill 的第6个位置），block 和 prefill 一样

# ============================================================
# 测试五：gather 从 cache 里读回 prefill 写入的值
# ============================================================
print("\n=== 测试五：gather 读回 prefill 数据 ===")
# 构造 block_table
block_table = torch.full((1, 64), -1, dtype=torch.int32, device=device)
bt = cache.get_block_table()
block_table[0, :len(bt)] = bt
context_lens = torch.tensor([prefill_len], dtype=torch.int32, device=device)

num_blocks = (prefill_len + BLOCK_SIZE - 1) // BLOCK_SIZE
phys = block_table[0, :num_blocks]
k_gathered = bm.gpu_kv_cache[0, 0][phys].permute(1, 0, 2, 3) \
              .reshape(N_KV_HEADS, -1, HEAD_DIM)[:, :prefill_len, :]

print(f"k original:\n{k}")
print(f"k gathered:\n{k_gathered.permute(1, 0, 2)}")  # (prefill_len, n_kv_heads, head_dim)
ok = torch.allclose(k.float(), k_gathered.permute(1, 0, 2).float(), atol=1e-2)
print(f"gather 和原始 k 一致: {'✓' if ok else '✗'}")

# 接着上面的测试继续

# ============================================================
# 测试六：第二个序列（physical_id != 0 的情况）
# ============================================================
print("\n=== 测试六：第二个序列 ===")

cache2 = PagedKVCache(
    block_manager=bm, num_layers=N_LAYERS,
    max_seq_len=64, num_kv_heads=N_KV_HEADS,
    head_dim=HEAD_DIM, device=device, dtype=dtype,
)

prefill_len2 = 5
cache2._allocate_for_prefill(prefill_len2)

logical_id2 = cache2.cache_block_index[0]
if hasattr(logical_id2, 'item'):
    logical_id2 = logical_id2.item()
_, physical_id2 = bm.block_mapping[logical_id2]
print(f"logical_id={logical_id2}, physical_id={physical_id2}")
# 预期：physical_id2 != 0，因为 physical_id=0 已被第一个序列占用

slot_mapping2 = cache2.get_prefill_slot_mapping(prefill_len2)
print(f"slot_mapping2: {slot_mapping2.tolist()}")

# 验证 slot 是否指向 physical_id2 而不是 0
for i, slot in enumerate(slot_mapping2.tolist()):
    actual_block = slot // BLOCK_SIZE
    actual_offset = slot % BLOCK_SIZE
    ok = actual_block == physical_id2 and actual_offset == i
    print(f"  token {i}: slot={slot}, block={actual_block}, offset={actual_offset} "
          f"(expected block={physical_id2}, offset={i}) {'✓' if ok else '✗'}")

# 写入
k2 = torch.ones(prefill_len2, N_KV_HEADS, HEAD_DIM, dtype=dtype, device=device) * 99
v2 = k2 * 2
store_kvcache(k2, v2, bm.gpu_kv_cache[0, 0], bm.gpu_kv_cache[1, 0], slot_mapping2)

# 验证写入位置正确（不能写到 physical_id=0 的位置）
for i, slot in enumerate(slot_mapping2.tolist()):
    blk = slot // BLOCK_SIZE
    off = slot % BLOCK_SIZE
    k_readback = bm.gpu_kv_cache[0, 0, blk, :, off, :]
    ok = torch.allclose(k_readback.float(), k2[i].float(), atol=1e-2)
    print(f"  store token {i}: slot={slot} 写入一致={ok}")

# 验证 decode slot
cache2._seq_len = prefill_len2
cache2.prepare_decode_step()
decode_slot2 = cache2.get_decode_slot()
print(f"\ndecode_slot2={decode_slot2}")
print(f"expected block={physical_id2}, offset={prefill_len2}")
ok = (decode_slot2 // BLOCK_SIZE == physical_id2 and 
      decode_slot2 % BLOCK_SIZE == prefill_len2)
print(f"decode slot 正确: {'✓' if ok else '✗'}")

# gather 验证
block_table2 = torch.full((1, 64), -1, dtype=torch.int32, device=device)
bt2 = cache2.get_block_table()
block_table2[0, :len(bt2)] = bt2
print(f"\nblock_table2 valid: {block_table2[0, :cache2.allocated_cache_block_num].tolist()}")

num_blocks2 = (prefill_len2 + BLOCK_SIZE - 1) // BLOCK_SIZE
phys2 = block_table2[0, :num_blocks2]
k_gathered2 = bm.gpu_kv_cache[0, 0][phys2].permute(1, 0, 2, 3) \
               .reshape(N_KV_HEADS, -1, HEAD_DIM)[:, :prefill_len2, :]
ok = torch.allclose(k2.float(), k_gathered2.permute(1, 0, 2).float(), atol=1e-2)
print(f"gather 和原始 k2 一致: {'✓' if ok else '✗'}")
if not ok:
    print(f"k2:\n{k2}")
    print(f"k_gathered2:\n{k_gathered2.permute(1, 0, 2)}")