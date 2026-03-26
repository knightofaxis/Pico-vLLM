# 实现naive kv cache和paged kv cache两个版本，统一用kv cache类抽象
# attention 只调用这两个方法，不管内部怎么存
from abc import ABC, abstractmethod
import torch
from torch import Tensor, dtype

class KVCache(ABC):  # 抽象接口

    
    ''' update: 更新指定 layer 的 KV，输入是当前 step 计算出的 K 和 V，shape (num_heads, head_dim)
        get: 获取指定 layer 的 KV，返回 shape (num_heads, head_dim) 的 K 和 V
        reset: 重置 KV cache，清空所有 KV'''
    @abstractmethod
    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        """
        将新的 k/v 写入 cache
        k, v: (1, num_kv_heads, head_dim)        ← decode 时
           或 (seq_len, num_kv_heads, head_dim)   ← prefill 时
        """
        ...
    @abstractmethod
    def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """
        返回当前层所有历史 token 的 k/v
        return: k, v 各 (seq_len, num_kv_heads, head_dim)
        """
        ...
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

# 后面替换
class PagedKVCache(KVCache):
    # block table + 物理块
    ...