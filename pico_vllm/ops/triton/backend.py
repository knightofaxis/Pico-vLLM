import torch
import torch.nn as nn

from ..base import OpsBackend


class TritonOps(OpsBackend):
    """Triton backend adapter over the existing kernel wrappers."""

    name = "triton"
    device_type = "cuda"
    supports_cuda_graph = True

    def create_rms_norm(self, hidden_size: int, eps: float = 1e-6) -> nn.Module:
        from .rms_norm import FastRMSNorm

        return FastRMSNorm(hidden_size, eps=eps)

    def swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        from .swiglu import fused_swiglu

        return fused_swiglu(gate_up)

    def store_kvcache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int = 16,
    ) -> None:
        from .store_kvcache import store_kvcache

        store_kvcache(k, v, k_cache, v_cache, slot_mapping, block_size=block_size)

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
        from .fused_rope_kvcache_store import fused_decode_rope_and_cache

        return fused_decode_rope_and_cache(
            q, k, v, cos, sin,
            kv_cache_k, kv_cache_v,
            slot_mapping,
            context_lens,
        )

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
        from .attention import paged_decode_attention

        return paged_decode_attention(
            q,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
            BLOCK_SIZE=BLOCK_SIZE,
        )

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
        from .attention import paged_prefill_attention

        return paged_prefill_attention(
            q,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            new_token_lens,
            q_start_loc,
            MAX_BLOCKS_PER_SEQ=MAX_BLOCKS_PER_SEQ,
            BLOCK_SIZE=BLOCK_SIZE,
            BLOCK_M=BLOCK_M,
        )
