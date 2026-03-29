# test_block_mapping_bug.py
import torch

# 模拟 block_mapping：key 是 Python int
block_mapping = {
    0: ('GPU', 10),
    1: ('GPU', 42),
    2: ('GPU', 7),
}

# 模拟 cache_block_index 是 tensor
cache_block_index = torch.tensor([1, 2, 0], dtype=torch.int32)

print("=== 用 tensor 查 dict（有 bug 的版本）===")
for lid in cache_block_index[:2]:
    print(f"lid type: {type(lid)}, value: {lid}")
    result = block_mapping.get(lid, None)
    print(f"block_mapping[{lid}] = {result}")

print()
print("=== 用 .item() 查 dict（正确版本）===")
for lid in cache_block_index[:2]:
    lid_int = lid.item()
    print(f"lid type: {type(lid_int)}, value: {lid_int}")
    result = block_mapping.get(lid_int, None)
    print(f"block_mapping[{lid_int}] = {result}")