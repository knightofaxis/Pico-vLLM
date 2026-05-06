from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class OpsBackend(ABC):
    """Backend-neutral operator surface used by the model.

    The interface is intentionally close to the current Triton wrappers so the
    model can be migrated one call site at a time.
    """

    name: str
    device_type: str
    supports_cuda_graph: bool = False

    @abstractmethod
    def create_rms_norm(self, hidden_size: int, eps: float = 1e-6) -> nn.Module:
        ...

    @abstractmethod
    def swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def store_kvcache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int = 16,
    ) -> None:
        ...

    @abstractmethod
    def decode_rope_and_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache_k: torch.Tensor,
        kv_cache_v: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def paged_decode_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        MAX_BLOCKS_PER_SEQ: int,
        BLOCK_SIZE: int = 16,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def paged_prefill_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        new_token_lens: torch.Tensor,
        q_start_loc: torch.Tensor,
        MAX_BLOCKS_PER_SEQ: int,
        BLOCK_SIZE: int = 16,
        BLOCK_M: int = 16,
    ) -> torch.Tensor:
        ...
