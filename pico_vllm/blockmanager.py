from collections import deque
from enum import Enum
import torch
from torch import dtype
from typing import Dict, List, Callable

class pagedblocktype(Enum):
    GPU = "gpu"
    CPU = "cpu"
    NONE = "none"  # 表示未分配
class BlockManager:
    """集中管理 KV cache 的物理显存/内存分页。

    - GPU / CPU 各持一块大的连续 KV 张量（pool），分成固定大小的 block。
    - 维护 "逻辑 block id → (设备类型, 物理 block id)" 的映射，同一逻辑 block 可以
      在 swap_in/swap_out 过程中在 GPU/CPU 之间迁移而保持对外 id 不变。
    - Prefix cache 的驱逐通过 `_evict_callback` 注入：当空闲不足时先触发回调尝试
      驱逐，再抛 OOM。
    - 引用计数 (`logical_ref_count`) 用于 prefix cache 的共享释放。
    """
    def __init__(self,
                 num_gpu_blocks: int, 
                 num_cpu_blocks: int, 
                 block_size: int, 
                 num_layers: int,
                 num_kv_heads: int, 
                 head_dim: int, 
                 dtype: dtype,
                 device='cuda'):
        self.dtype = dtype
        self.device = torch.device(device)
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
            2, num_layers, num_gpu_blocks, num_kv_heads, block_size, head_dim,
            device=self.device, dtype=dtype
        )
        
        # CPU pool（offload 目标）
        self.cpu_free_blocks: deque[int] = deque(range(num_cpu_blocks))
        self.cpu_kv_cache = torch.zeros(
            2, num_layers, num_cpu_blocks, num_kv_heads, block_size, head_dim,
            device='cpu', dtype=dtype,
            pin_memory=self.device.type == 'cuda'  # ← 关键：pin_memory 让 CPU→GPU 传输更快
        )
        
        ########## 逻辑块索引 ##########
        # 逻辑块表，记录每个逻辑块对应的物理块和类型
        # 最多逻辑块不会超过 num_total_blocks，因为每个逻辑块至少占一个物理块
        # 第i个初始逻辑块初始化为[NONE, -1]，表示未分配物理块
        self.block_mapping: List[tuple[pagedblocktype, int]] = [
            (pagedblocktype.NONE, -1) for i in range(self.num_total_blocks)
        ]
        self.logical_free_blocks: deque[int] = deque(range(self.num_total_blocks))  # 逻辑块索引，初始全部空闲


        ######### prefix caching 相关 #########
        self.logical_ref_count = [0] * self.num_total_blocks   # 逻辑块的 ref
        self._evict_callback = None   # type: Callable[[int], int] | None
    
    def set_evict_callback(self, callback):
        """注册驱逐回调。callback(num_needed) -> num_actually_evicted"""
        self._evict_callback = callback
    
    def allocate(self, num_blocks: int = 1) -> List[int]:
        while self.num_free_blocks < num_blocks:
            if self._evict_callback is None:
                break
            needed = num_blocks - self.num_free_blocks
            evicted_count = self._evict_callback(needed)
            if evicted_count == 0:
                break   # 驱逐不出来了，真 OOM
        
        if self.num_free_blocks < num_blocks:
            raise RuntimeError("Block Manager Out Of Memory.")

        # 分配 num_blocks 个物理块
        block_ids = []
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(f"Not enough free blocks to allocate {num_blocks} blocks")
        for _ in range(num_blocks):
            if self.gpu_free_blocks:
                physical_block_id = self.gpu_free_blocks.popleft()
                logical_block_id = self.logical_free_blocks.popleft()
                self.block_mapping[logical_block_id] = (pagedblocktype.GPU, physical_block_id)
                self.logical_ref_count[logical_block_id] = 1 # 增加1引用计数
                block_ids.append(logical_block_id)
            elif self.cpu_free_blocks:
                physical_block_id = self.cpu_free_blocks.popleft()
                logical_block_id = self.logical_free_blocks.popleft()
                self.block_mapping[logical_block_id] = (pagedblocktype.CPU, physical_block_id)
                self.logical_ref_count[logical_block_id] = 1 # 增加1引用计数
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
    # Prefix Caching 需要的引用技术相关操作
    #####################################################
    def inc_ref(self, logical_block_ids: list[int]):
        for bid in logical_block_ids:
            self.logical_ref_count[bid] += 1

    def dec_ref(self, logical_block_ids: list[int]):
        """引用降到 0 时 block 立即回收"""
        to_free_blocks:list[int] = []
        for bid in logical_block_ids:
            self.logical_ref_count[bid] -= 1
            assert self.logical_ref_count[bid] >= 0
            if self.logical_ref_count[bid] == 0:
                to_free_blocks.append(bid)
        
        self.free(to_free_blocks)
    
    #####################################################
    # 非必须的方法，用于offload到CPU，swap_in/swap_out()
    # 暂时还没有实际使用
    #####################################################
    def swap_out(self, block_ids: List[int]) -> None:
        """
        把 GPU 上的 block 换出到 CPU
        """
        for block_id in block_ids:
            block_type, physical_block_id = self.block_mapping[block_id]
            if block_type != pagedblocktype.GPU:
                raise RuntimeError(f"Block {block_id} is not on GPU, cannot swap out")
            cpu_block_id = self.cpu_free_blocks.popleft()
            self.cpu_kv_cache[:,:,cpu_block_id].copy_(
                self.gpu_kv_cache[:,:,physical_block_id], non_blocking=True
            )
            self.gpu_free_blocks.append(physical_block_id)  # GPU 块释放
            self.block_mapping[block_id] = (pagedblocktype.CPU, cpu_block_id)  # 更新映射

    def swap_in(self, block_ids: List[int]) -> None:
        """
        把 CPU 上的 block 换回 GPU
        """
        for block_id in block_ids:
            block_type, physical_block_id = self.block_mapping[block_id]
            if block_type != pagedblocktype.CPU:
                raise RuntimeError(f"Block {block_id} is not on CPU, cannot swap in")
            gpu_block_id = self.gpu_free_blocks.popleft()
            self.gpu_kv_cache[:,:,gpu_block_id].copy_(
                self.cpu_kv_cache[:,:,physical_block_id], non_blocking=True
            )
            self.cpu_free_blocks.append(physical_block_id)  # CPU 块释放
            self.block_mapping[block_id] = (pagedblocktype.GPU, gpu_block_id)  # 更新映射

    @property
    def num_free_blocks(self) -> int:
        return len(self.gpu_free_blocks) + len(self.cpu_free_blocks)

    def can_allocate_gpu(self, num_blocks: int) -> bool:
        # 查询是否有足够的GPU空闲块
        # scheduler 用这个决定是否接受新请求
        return len(self.gpu_free_blocks) >= num_blocks
