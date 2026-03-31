# 实现naive kv cache和paged kv cache两个版本，统一用kv cache类抽象
# attention 只调用这两个方法，不管内部怎么存
from abc import ABC, abstractmethod
import torch
from torch import Tensor, dtype
from blockmanager import BlockManager, pagedblocktype
from typing import List
import math

class KVCache(ABC):  # 抽象接口
    
    # ''' update: 更新指定 layer 的 KV，输入是当前 step 计算出的 K 和 V，shape (num_heads, head_dim)
    #     get: 获取指定 layer 的 KV，返回 shape (num_heads, head_dim) 的 K 和 V
    #     reset: 重置 KV cache，清空所有 KV'''
    # @abstractmethod
    # def update(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
    #     """
    #     将新的 k/v 写入 cache
    #     k, v: (1, num_kv_heads, head_dim)        ← decode 时
    #        或 (seq_len, num_kv_heads, head_dim)   ← prefill 时
    #     """
    #     ...
    # @abstractmethod
    # def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
    #     """
    #     返回当前层所有历史 token 的 k/v
    #     return: k, v 各 (seq_len, num_kv_heads, head_dim)
    #     """
    #     ...
    @abstractmethod
    def reset(self) -> None:
        """清空 cache，开始新的请求"""
        ...
    @property
    @abstractmethod
    def seq_len(self) -> int:
        """当前已缓存的 token 数量"""
        ...
    
# 朴素实现
class NaiveKVCache(KVCache):
    # 连续 tensor
    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_dim, device, dtype):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.cache = torch.zeros(
            num_layers, 2, max_seq_len, num_kv_heads, head_dim,
            device=device, dtype=dtype
        )
        self._seq_len = 0  # 当前已填入的长度

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        # k, v: (1, num_kv_heads, head_dim) 或 (seq_len, num_kv_heads, head_dim)
        # 通过读取kv的shape自动适配 prefill 和 decode 场景
        # 以后用 cuda kernel 的时候这个要怎么处理？可能需要分开 prefill 和 decode 的接口，或者额外传参？
        num_new_tokens = k.shape[0]
        start = self._seq_len
        end = self._seq_len + num_new_tokens

        self.cache[layer_idx, 0, start:end, :, :] = k
        self.cache[layer_idx, 1, start:end, :, :] = v
        if (layer_idx == self.num_layers - 1):  # 只在最后一层更新 seq_len，保证所有层的 seq_len 一致
            self._seq_len += num_new_tokens

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        return self.cache[layer_idx, 0, :self._seq_len, :, :], self.cache[layer_idx, 1, :self._seq_len, :, :]

    def reset(self) -> None:
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

# 分页的page Attention用的kv cache管理
class PagedKVCache():
    # block table + 物理块
    # max_block_len: int          # 最大的block数量
    def __init__(self, 
                 block_manager: BlockManager, 
                 num_layers: int, 
                 max_seq_len: int, 
                 num_kv_heads: int, 
                 head_dim: int, 
                 device, 
                 dtype: dtype):
        self.block_manager = block_manager
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.block_size = block_manager.block_size

        self.max_cache_block_num = math.ceil(max_seq_len / block_manager.block_size)
        self.cache_block_index : List[int] = [ (-1) for i in range(self.max_cache_block_num)]
        self.allocated_cache_block_num = 0
        self._seq_len = 0  # 当前已填入的长度
        self._updated_layer = 0 # 当前已填入的layer数量

        # 优化 1：直接缓存 physical_ids 的 Python List，加速标量查询
        self.physical_block_ids: List[int] = []
        self.logical_block_ids: List[int] = []
        
        # 优化 2：在 GPU 上预分配一条完整的 static block table
        self.gpu_block_table = torch.full(
            (self.max_cache_block_num,), -1, dtype=torch.int32, device=self.device
        )

    def _allocate_blocks(self, num_blocks: int) -> None:
        """内部统一分配物理块并同步更新 GPU Cache"""
        new_logical_ids = self.block_manager.allocate(num_blocks)
        new_physical_ids = [self.block_manager.block_mapping[lid][1] for lid in new_logical_ids]

        self.logical_block_ids.extend(new_logical_ids)   # 记录
        self.physical_block_ids.extend(new_physical_ids)
        
        start_idx = self.allocated_cache_block_num
        self.allocated_cache_block_num += num_blocks
        
        # 增量更新 GPU Tensor (只在分配新 block 时触发，16 个 token 才触发一次)
        new_phys_tensor = torch.tensor(new_physical_ids, dtype=torch.int32, device=self.device)
        self.gpu_block_table[start_idx : self.allocated_cache_block_num] = new_phys_tensor

    def prepare_decode_step(self) -> None:
        total_needed = math.ceil((self._seq_len + 1) / self.block_size)
        diff = total_needed - self.allocated_cache_block_num
        if diff > 0:
            self._allocate_blocks(diff)

    def _allocate_for_prefill(self, prefill_len: int) -> None:
        total_needed = math.ceil((self._seq_len + prefill_len) / self.block_size)
        diff = total_needed - self.allocated_cache_block_num
        if diff > 0:
            self._allocate_blocks(diff)

    def get_block_table(self) -> torch.Tensor:
        """
        直接返回 GPU Tensor 的View切片
        """
        return self.gpu_block_table[:self.allocated_cache_block_num]

    def get_decode_slot(self) -> int:
        """
        直接从 Python List 取物理 ID，避免从 GPU Tensor 取标量的隐式同步开销
        """
        token_pos = self._seq_len
        block_idx = token_pos // self.block_size
        offset = token_pos % self.block_size
        
        physical_id = self.physical_block_ids[block_idx]
        return physical_id * self.block_size + offset

    def get_prefill_slot_mapping(self, prefill_len: int) -> torch.Tensor:
        """
        去掉冗余的 block_mapping 字典查询
        """
        slots = []
        for i in range(prefill_len):
            token_pos = self._seq_len + i
            block_idx = token_pos // self.block_size
            offset = token_pos % self.block_size
            
            physical_id = self.physical_block_ids[block_idx]
            slots.append(physical_id * self.block_size + offset)
            
        return torch.tensor(slots, dtype=torch.int32, device=self.device)

    def reset(self) -> None:
        """
        释放所有 block，归还给 block_manager
        """
        self.block_manager.free(self.logical_block_ids)  # ← 用 logical id
        self._seq_len = 0
        self.allocated_cache_block_num = 0
        self.physical_block_ids = []
        self.logical_block_ids = []
        self.gpu_block_table.fill_(-1)

    @property
    def seq_len(self) -> int:
        return self._seq_len
    