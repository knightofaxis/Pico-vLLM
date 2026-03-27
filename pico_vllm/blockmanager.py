from collections import deque
from enum import Enum
import torch
from torch import dtype
from typing import Dict, List

class pagedblocktype(Enum):
    GPU = "gpu"
    CPU = "cpu"
    NONE = "none"  # 表示未分配
class BlockManager:
    def __init__(self, 
                 num_gpu_blocks: int, 
                 num_cpu_blocks: int, 
                 block_size: int, 
                 num_kv_heads: int, 
                 head_dim: int, 
                 dtype: dtype):
        ########## 物理块索引 ##########
        self.num_physical_gpu_blocks = num_gpu_blocks
        self.num_physical_cpu_blocks = num_cpu_blocks
        self.num_total_blocks = num_gpu_blocks + num_cpu_blocks
        self.block_size = block_size
        # 空闲块列表，初始全部空闲
        # 用 deque 比 list 的 pop(0) 快
        # GPU pool
        self.gpu_free_blocks: deque[int] = deque(range(num_gpu_blocks))
        self.gpu_kv_cache = torch.zeros(
            num_gpu_blocks, 2, num_kv_heads, block_size, head_dim,
            device='cuda', dtype=dtype
        )
        
        # CPU pool（offload 目标）
        self.cpu_free_blocks: deque[int] = deque(range(num_cpu_blocks))
        self.cpu_kv_cache = torch.zeros(
            num_cpu_blocks, 2, num_kv_heads, block_size, head_dim,
            device='cpu', dtype=dtype,
            pin_memory=True  # ← 关键：pin_memory 让 CPU→GPU 传输更快
        )
        
        ########## 逻辑块索引 ##########
        # 逻辑块表，记录每个逻辑块对应的物理块和类型
        # 最多逻辑块不会超过 num_total_blocks，因为每个逻辑块至少占一个物理块
        # 第i个初始逻辑块初始化为[NONE, -1]，表示未分配物理块
        self.block_mapping: List[tuple[pagedblocktype, int]] = [
            (pagedblocktype.NONE, -1) for i in range(self.num_total_blocks)
        ]
        self.logical_free_blocks: deque[int] = deque(range(self.num_total_blocks))  # 逻辑块索引，初始全部空闲
    
    def allocate(self, num_blocks: int) -> List[int]:
        # 分配 num_blocks 个物理块
        block_ids = []
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(f"Not enough free blocks to allocate {num_blocks} blocks")
        for _ in range(num_blocks):
            if self.gpu_free_blocks:
                physical_block_id = self.gpu_free_blocks.popleft()
                logical_block_id = self.logical_free_blocks.popleft()
                self.block_mapping[logical_block_id] = (pagedblocktype.GPU, physical_block_id)
                block_ids.append(logical_block_id)
            elif self.cpu_free_blocks:
                physical_block_id = self.cpu_free_blocks.popleft()
                logical_block_id = self.logical_free_blocks.popleft()
                self.block_mapping[logical_block_id] = (pagedblocktype.CPU, physical_block_id)
                block_ids.append(logical_block_id)
            else:
                raise RuntimeError("No free blocks available")

        return block_ids

    def free(self, block_ids: List[int]) -> None:
        # 回收物理块，加回 free_blocks
        # 回收逻辑块
        for block_id in block_ids:
            block_type, physical_block_id = self.block_mapping[block_id]
            if block_type == pagedblocktype.GPU:
                self.gpu_free_blocks.append(physical_block_id)
            elif block_type == pagedblocktype.CPU:
                self.cpu_free_blocks.append(physical_block_id)
            else:
                raise RuntimeError(f"Block {block_id} is not allocated")
            # 更新 block_mapping
            self.logical_free_blocks.append(block_id)
            self.block_mapping[block_id] = (pagedblocktype.NONE, -1)

    #####################################################
    # 非必须的方法，用于offload到CPU，swap_in/swap_out()
    #####################################################
    def swap_out(self, block_ids: List[int]) -> None:
        """
        把 GPU 上的 block 换出到 CPU
        """
        cpu_block_ids = []
        for block_id in block_ids:
            block_type, physical_block_id = self.block_mapping[block_id]
            if block_type != pagedblocktype.GPU:
                raise RuntimeError(f"Block {block_id} is not on GPU, cannot swap out")
            cpu_block_id = self.cpu_free_blocks.popleft()
            self.cpu_kv_cache[cpu_block_id].copy_(
                self.gpu_kv_cache[physical_block_id], non_blocking=True
            )
            self.gpu_free_blocks.append(physical_block_id)  # GPU 块释放
            self.block_mapping[block_id] = (pagedblocktype.CPU, cpu_block_id)  # 更新映射
            cpu_block_ids.append(cpu_block_id)

    def swap_in(self, block_ids: List[int]) -> None:
        """
        把 CPU 上的 block 换回 GPU
        """
        gpu_block_ids = []
        for block_id in block_ids:
            block_type, physical_block_id = self.block_mapping[block_id]
            if block_type != pagedblocktype.CPU:
                raise RuntimeError(f"Block {block_id} is not on CPU, cannot swap in")
            gpu_block_id = self.gpu_free_blocks.popleft()
            self.gpu_kv_cache[gpu_block_id].copy_(
                self.cpu_kv_cache[physical_block_id], non_blocking=True
            )
            self.cpu_free_blocks.append(physical_block_id)  # CPU 块释放
            self.block_mapping[block_id] = (pagedblocktype.GPU, gpu_block_id)  # 更新映射
            gpu_block_ids.append(gpu_block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.gpu_free_blocks) + len(self.cpu_free_blocks)

    def can_allocate_gpu(self, num_blocks: int) -> bool:
        # 查询是否有足够的GPU空闲块
        # scheduler 用这个决定是否接受新请求
        return len(self.gpu_free_blocks) >= num_blocks
    
if __name__ == "__main__":
    import torch
    from collections import deque
    
    print("=" * 60)
    print("BlockManager 单元测试")
    print("=" * 60)
    
    # 初始化：4个GPU块，2个CPU块，block_size=4
    bm = BlockManager(
        num_gpu_blocks=4,
        num_cpu_blocks=2,
        block_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )
    
    # ===== 测试1：基本状态 =====
    print("\n[Test 1] 初始状态")
    assert len(bm.gpu_free_blocks) == 4
    assert len(bm.cpu_free_blocks) == 2
    assert len(bm.logical_free_blocks) == 6
    assert bm.num_free_blocks == 6
    assert bm.can_allocate_gpu(4) == True
    assert bm.can_allocate_gpu(5) == False
    print("  初始状态正确 ✓")

    # ===== 测试2：基本分配 =====
    print("\n[Test 2] 基本分配")
    ids_a = bm.allocate(2)  # 分配 2 个块给请求 A
    assert len(ids_a) == 2
    assert len(bm.gpu_free_blocks) == 2
    assert len(bm.logical_free_blocks) == 4
    # 验证 block_mapping 正确
    for lid in ids_a:
        btype, pid = bm.block_mapping[lid]
        assert btype == pagedblocktype.GPU
        assert pid >= 0
    print(f"  分配 2 个块: logical_ids={ids_a} ✓")

    # ===== 测试3：写入数据，验证物理地址正确 =====
    print("\n[Test 3] 写入数据验证")
    lid = ids_a[0]
    btype, pid = bm.block_mapping[lid]
    # 写入一个可识别的值
    bm.gpu_kv_cache[pid, 0, 0, 0, 0] = 42.0
    assert bm.gpu_kv_cache[pid, 0, 0, 0, 0].item() == 42.0
    print(f"  逻辑块 {lid} → 物理GPU块 {pid}，写入42.0 ✓")

    # ===== 测试4：swap_out =====
    print("\n[Test 4] swap_out（GPU → CPU）")
    bm.swap_out(ids_a)
    assert len(bm.gpu_free_blocks) == 4
    # 验证 block_mapping 已更新为 CPU
    for lid in ids_a:
        btype, pid = bm.block_mapping[lid]
        assert btype == pagedblocktype.CPU, f"block {lid} should be CPU after swap_out"
    # 验证 GPU 块已释放
    assert len(bm.gpu_free_blocks) == 4
    assert len(bm.cpu_free_blocks) == 0
    # 验证数据被正确拷贝到 CPU
    lid = ids_a[0]
    _, cpu_pid = bm.block_mapping[lid]
    assert bm.cpu_kv_cache[cpu_pid, 0, 0, 0, 0].item() == 42.0
    print(f"  swap_out 后数据正确，GPU块已释放 ✓")

    # ===== 测试5：swap_in =====
    print("\n[Test 5] swap_in（CPU → GPU）")
    bm.swap_in(ids_a)
    # 验证 block_mapping 已更新为 GPU
    for lid in ids_a:
        btype, pid = bm.block_mapping[lid]
        assert btype == pagedblocktype.GPU, f"block {lid} should be GPU after swap_in"
    assert len(bm.cpu_free_blocks) == 2
    assert len(bm.gpu_free_blocks) == 2
    # 验证数据完整性
    lid = ids_a[0]
    _, gpu_pid = bm.block_mapping[lid]
    assert bm.gpu_kv_cache[gpu_pid, 0, 0, 0, 0].item() == 42.0
    print(f"  swap_in 后数据正确，CPU块已释放 ✓")

    # ===== 测试6：free =====
    print("\n[Test 6] free")
    bm.free(ids_a)
    for lid in ids_a:
        btype, _ = bm.block_mapping[lid]
        assert btype == pagedblocktype.NONE
    assert len(bm.gpu_free_blocks) == 4
    assert len(bm.logical_free_blocks) == 6  # 逻辑块也归还了
    print(f"  free 后状态正确，逻辑块已归还 ✓")

    # ===== 测试7：OOM =====
    print("\n[Test 7] OOM 检测")
    try:
        bm.allocate(7)  # 只有 6 个块
        assert False, "应该抛出 RuntimeError"
    except RuntimeError as e:
        print(f"  正确抛出 OOM: {e} ✓")

    # ===== 测试8：分配用尽 GPU 后自动用 CPU =====
    print("\n[Test 8] GPU 用尽后自动分配 CPU 块")
    all_ids = bm.allocate(6)  # 4 GPU + 2 CPU
    gpu_count = sum(1 for lid in all_ids if bm.block_mapping[lid][0] == pagedblocktype.GPU)
    cpu_count = sum(1 for lid in all_ids if bm.block_mapping[lid][0] == pagedblocktype.CPU)
    assert gpu_count == 4
    assert cpu_count == 2
    assert len(bm.gpu_free_blocks) == 0
    assert len(bm.cpu_free_blocks) == 0
    print(f"  分配 6 块: {gpu_count} GPU + {cpu_count} CPU ✓")
    bm.free(all_ids)

    # ===== 测试9：free 后重新分配（复用逻辑id）=====
    print("\n[Test 9] free 后重新分配")
    ids1 = bm.allocate(2)
    bm.free(ids1)
    ids2 = bm.allocate(2)
    # 逻辑 id 应该被复用
    assert set(ids2).issubset(set(range(bm.num_total_blocks)))
    bm.free(ids2)
    assert bm.num_free_blocks == 6
    print(f"  复用逻辑id正确 ✓")

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)