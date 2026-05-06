import torch
import torch.nn as nn

from ..base import OpsBackend


class _UnimplementedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Torch backend op is not implemented yet: rms_norm")


class TorchOps(OpsBackend):
    """CPU/Torch backend placeholder.

    This class defines the expected surface but intentionally does not implement
    kernels yet. Fill these methods in as CPU operator work lands.
    """

    name = "torch"
    device_type = "cpu"
    supports_cuda_graph = False

    def create_rms_norm(self, hidden_size: int, eps: float = 1e-6) -> nn.Module:
        return _UnimplementedRMSNorm(hidden_size, eps)

    def swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Torch backend op is not implemented yet: swiglu")

    def store_kvcache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int = 16,
    ) -> None:
        raise NotImplementedError("Torch backend op is not implemented yet: store_kvcache")

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
        raise NotImplementedError("Torch backend op is not implemented yet: decode_rope_and_cache")

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
        raise NotImplementedError("Torch backend op is not implemented yet: paged_decode_attention")

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
        raise NotImplementedError("Torch backend op is not implemented yet: paged_prefill_attention")
