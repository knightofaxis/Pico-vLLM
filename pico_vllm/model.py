# model.py
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from cache import KVCache, NaiveKVCache, PagedKVCache
from Attention import paged_decode_attention
from store_kvcache import store_kvcache
from RMSNorm import FastRMSNorm
from SwiGLU import fused_swiglu
from Fused_RoPE_KVcache_store import fused_decode_rope_and_cache

@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 8960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = True

    BLOCK_SIZE = 16 # 目前硬编码为固定值
    MAX_BLOCKS_PER_SEQ = max_position_embeddings // BLOCK_SIZE  # 固定值，启动时确定

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self):
        # 每个 KV 头服务几个 Q 头
        return self.num_attention_heads // self.num_key_value_heads

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    # @torch.compile(fullgraph=True)
    # @torch.compile(options={"epilogue_fusion": True})
    def forward(self, x):
        # x: (B, seq, hidden_size)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

class RoPE(nn.Module):
    cos_table: torch.Tensor  # 类型声明，告诉 Pylance 这个属性存在
    sin_table: torch.Tensor

    def __init__(self, head_dim: int, rope_theta: float = 1000000.0, 
                 max_seq_len: int = 131072):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        
        # 预计算全局 cos/sin 表，避免每次 forward 都算
        # freqs: (head_dim//2,)
        freqs = 1.0 / (rope_theta ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        ))
        # positions: (max_seq_len,)
        t = torch.arange(max_seq_len).float()
        # angles: (max_seq_len, head_dim//2)
        angles = torch.outer(t, freqs)
        # cos/sin 表: (max_seq_len, head_dim)
        # 拼接两份，对应 concat 式 RoPE
        # cos_table = torch.cat([angles.cos(), angles.cos()], dim=-1)
        # sin_table = torch.cat([angles.sin(), angles.sin()], dim=-1)
        
        # register_buffer: 跟随模型 device 移动，但不是参数
        self.register_buffer('cos_table', torch.cat([angles.cos(), angles.cos()], dim=-1), persistent=False)
        self.register_buffer('sin_table', torch.cat([angles.sin(), angles.sin()], dim=-1), persistent=False)

    def get_cos_sin(self, position_ids: torch.Tensor):
        """
        position_ids: (B, seq_len) 或 (seq_len,)
        return: cos, sin 各 (B, seq_len, head_dim) 或 (seq_len, head_dim)
        """
        return self.cos_table[position_ids], self.sin_table[position_ids]

    @staticmethod
    def apply_rope(q: torch.Tensor, k: torch.Tensor,
                   cos: torch.Tensor, sin: torch.Tensor):
        """
        q, k: (B, seq_len, n_heads, head_dim)
        cos, sin: (B, seq_len, head_dim) 或 broadcast 兼容的形状
        return: q_rot, k_rot，形状不变
        
        concat 式旋转：
          前半 head_dim//2 和后半 head_dim//2 配对旋转
          rotate_half(x) = [-x[..., head_dim//2:], x[..., :head_dim//2]]
        """
        # cos/sin 需要加 n_heads 维度以 broadcast
        # (B, seq_len, head_dim) -> (B, seq_len, 1, head_dim)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        
        def rotate_half(x):
            half = x.shape[-1] // 2
            x1 = x[..., :half]   # 前半
            x2 = x[..., half:]   # 后半
            return torch.cat([-x2, x1], dim=-1)
        
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot

class GQAAttention(nn.Module):
    layer_idx: Tensor
    def __init__(self, cfg: ModelConfig, layer_idx):
        super().__init__()
        self.cfg = cfg
        # self.layer_idx = Tensor(layer_idx)
        self.register_buffer(
            'layer_idx', 
            torch.tensor(layer_idx, dtype=torch.int32), 
            persistent=False
        )
        # self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        # # k_proj: (B, seq_len, hidden_size) -> (B, seq_len, num_kv_heads * head_dim)
        # self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        # self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim          # q
            + cfg.num_key_value_heads * cfg.head_dim        # k  
            + cfg.num_key_value_heads * cfg.head_dim,       # v
            bias=True
        )

    def _prefill_attention(self, q: Tensor, k: Tensor, v: Tensor):
        
        # SDPA（B=1，标准 causal attention）
        # q/k/v: (B, seq, n_heads, head_dim) → 转成 SDPA 期望的格式
        q = q.transpose(1, 2)  # (B, n_heads, seq, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.repeat_interleave(self.cfg.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.cfg.num_kv_groups, dim=1)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # (B, n_heads, seq, head_dim) → (B, seq, hidden)
        return out.transpose(1, 2).contiguous().view(q.shape[0], -1, self.cfg.hidden_size)

    def _decode_attention(self, q, kv_cache_k, kv_cache_v, block_table, context_lens):
        # 直接从 cache 读，不再需要 k/v 参数
        B = q.shape[0]
        q_triton = q.transpose(1, 2)
        
        out = paged_decode_attention(
            q_triton, kv_cache_k, kv_cache_v,# type:ignore
            block_table, context_lens,
            MAX_BLOCKS_PER_SEQ=block_table.shape[1],
        )

        return out.transpose(1, 2).contiguous().view(B, 1, self.cfg.hidden_size)
    
    def forward(self,
                x: Tensor,                    # (B, seq_len, hidden_size)
                cos: Tensor, sin: Tensor, 
                kv_cache_k: Tensor,
                kv_cache_v: Tensor,
                slot_mapping: Tensor,
                is_prefill: bool,
                block_table: Tensor | None = None,   # (B, max_blocks) int32，物理块id
                context_lens: Tensor | None = None,  # (B,) int32，每个请求当前长度
                ) -> Tensor:                  # (B, seq_len, hidden_size)
        B, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, seq, q_size + k_size + v_size)

        q_size = self.cfg.num_attention_heads * self.cfg.head_dim
        k_size = self.cfg.num_key_value_heads * self.cfg.head_dim

        q, k, v = qkv.split([q_size, k_size, k_size], dim=-1)
        q = q.view(B, seq_len, self.cfg.num_attention_heads, self.cfg.head_dim)
        k = k.view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)
        v = v.view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)
        
        # RoPE（两条路径共用）
        q, k = RoPE.apply_rope(q, k, cos, sin)
        # print(v.is_contiguous())
        # k/v reshape 成 (total_tokens, n_kv_heads, head_dim) 给 store kernel
        k_flat = k.reshape(-1, self.cfg.num_key_value_heads, self.cfg.head_dim)
        v_flat = v.reshape(-1, self.cfg.num_key_value_heads, self.cfg.head_dim)

        # 统一写入 KV cache（prefill 和 decode 都用这个 kernel）
        store_kvcache(k_flat, v_flat, kv_cache_k, kv_cache_v, slot_mapping)# type:ignore

        if is_prefill:
            output = self._prefill_attention(q, k, v)
        else:
            output = self._decode_attention(q, kv_cache_k, kv_cache_v, block_table, context_lens)

        return self.o_proj(output)

    def forward_prefill(self,
                x: Tensor,                    # (B, seq_len, hidden_size)
                cos: Tensor, sin: Tensor, 
                kv_cache_k: Tensor,
                kv_cache_v: Tensor,
                slot_mapping: Tensor,
                block_table: Tensor | None = None,   # (B, max_blocks) int32，物理块id
                context_lens: Tensor | None = None,  # (B,) int32，每个请求当前长度
                ) -> Tensor:                  # (B, seq_len, hidden_size)
        B, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, seq, q_size + k_size + v_size)

        q_size = self.cfg.num_attention_heads * self.cfg.head_dim
        k_size = self.cfg.num_key_value_heads * self.cfg.head_dim

        q, k, v = qkv.split([q_size, k_size, k_size], dim=-1)
        q = q.view(B, seq_len, self.cfg.num_attention_heads, self.cfg.head_dim)
        k = k.view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)
        v = v.view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)
        
        # RoPE（两条路径共用）
        q, k = RoPE.apply_rope(q, k, cos, sin)
        # print(v.is_contiguous())
        # k/v reshape 成 (total_tokens, n_kv_heads, head_dim) 给 store kernel
        k_flat = k.reshape(-1, self.cfg.num_key_value_heads, self.cfg.head_dim)
        v_flat = v.reshape(-1, self.cfg.num_key_value_heads, self.cfg.head_dim)

        # 统一写入 KV cache（prefill 和 decode 都用这个 kernel）
        store_kvcache(k_flat, v_flat, kv_cache_k, kv_cache_v, slot_mapping)# type:ignore

        output = self._prefill_attention(q, k, v)

        return self.o_proj(output)
    
    def forward_decode(self,
                x: Tensor,                    # (B, seq_len, hidden_size)
                cos: Tensor, sin: Tensor, 
                kv_cache_k: Tensor,
                kv_cache_v: Tensor,
                slot_mapping: Tensor,
                block_table: Tensor,   # (B, max_blocks) int32，物理块id
                context_lens: Tensor,  # (B,) int32，每个请求当前长度
                ) -> Tensor:                  # (B, seq_len, hidden_size)
        B, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)  # (B, seq, q_size + k_size + v_size)

        q_size = self.cfg.num_attention_heads * self.cfg.head_dim
        k_size = self.cfg.num_key_value_heads * self.cfg.head_dim

        q, k, v = qkv.split([q_size, k_size, k_size], dim=-1)
        q = q.view(B, seq_len, self.cfg.num_attention_heads, self.cfg.head_dim)
        k = k.view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)
        v = v.view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)
        
        # RoPE（两条路径共用）
        # q, k = RoPE.apply_rope(q, k, cos, sin)
        # # print(v.is_contiguous())
        # # k/v reshape 成 (total_tokens, n_kv_heads, head_dim) 给 store kernel
        # k_flat = k.reshape(-1, self.cfg.num_key_value_heads, self.cfg.head_dim)
        # v_flat = v.reshape(-1, self.cfg.num_key_value_heads, self.cfg.head_dim)

        # # 统一写入 KV cache（prefill 和 decode 都用这个 kernel）
        # store_kvcache(k_flat, v_flat, kv_cache_k, kv_cache_v, slot_mapping)# type:ignore
        q_rot = fused_decode_rope_and_cache(
            q, k, v, cos, sin, 
            kv_cache_k, kv_cache_v, 
            slot_mapping,
            context_lens
        )

        output = self._decode_attention(q_rot, kv_cache_k, kv_cache_v, block_table, context_lens)

        return self.o_proj(output)
    
class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.gate_up_proj = nn.Linear(
            cfg.hidden_size, cfg.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        # x: (B, seq_len, hidden_size)
        # gate = self.gate_proj(x)  # (B, seq_len, intermediate_size)
        # up = self.up_proj(x)  # (B, seq_len, intermediate_size)
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.split(self.cfg.intermediate_size, dim=-1)
        return self.down_proj(F.silu(gate) * up)  # (B, seq_len, hidden_size)
    
    def forward_decode(self, x):
            # x: (B, 1, hidden_size)  -- 天然支持 B > 1
            # 1. 跑一次大矩阵乘法 (GEMV/GEMM)，算出 gate_up
            gate_up = self.gate_up_proj(x)
            
            # 2. 调用手写 Triton 算子，零碎片、零中间变量完成 SwiGLU
            activated = fused_swiglu(gate_up)
            
            # 3. 再跑一次下采样矩阵乘法
            return self.down_proj(activated)

#############################################################
# Transformer block
#############################################################
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.attn = GQAAttention(cfg, layer_idx=layer_idx)
        self.ffn = SwiGLUFFN(cfg)
        self.norm1 = FastRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm2 = FastRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin,
                kv_cache_k: Tensor,   # 当前层的 slice: (num_blocks, n_kv_heads, block_size, head_dim)
                kv_cache_v: Tensor,
                slot_mapping: Tensor,
                is_prefill: bool,
                block_table: Tensor | None = None,
                context_lens: Tensor | None = None,
                ):
        attn_out = self.attn(
            self.norm1(x), cos, sin,
            kv_cache_k=kv_cache_k,
            kv_cache_v=kv_cache_v,
            slot_mapping=slot_mapping,
            is_prefill=is_prefill,
            block_table=block_table,
            context_lens=context_lens,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_decode(self, x, cos, sin,
                kv_cache_k: Tensor,   # 当前层的 slice: (num_blocks, n_kv_heads, block_size, head_dim)
                kv_cache_v: Tensor,
                slot_mapping: Tensor,
                block_table: Tensor,
                context_lens: Tensor,
                ):
        attn_out = self.attn.forward_decode(
            self.norm1(x), cos, sin,
            kv_cache_k=kv_cache_k,
            kv_cache_v=kv_cache_v,
            slot_mapping=slot_mapping,
            block_table=block_table,
            context_lens=context_lens,
        )
        x = x + attn_out
        x = x + self.ffn.forward_decode(self.norm2(x))
        return x
#############################################################
# Model Assembly to Qwen2.5-1.5B
#############################################################
class Qwen25_15B(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = RoPE(cfg.head_dim, cfg.rope_theta)
        self.layers = nn.ModuleList([TransformerBlock(cfg, layer_idx=i) for i in range(cfg.num_hidden_layers)])
        self.norm = FastRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None  # 后面 forward 里会用 embed_tokens.weight 来做输出投影
    
    def forward(self,
            input_ids: Tensor,
            kv_cache_k: Tensor,       # (num_layers, num_blocks, n_kv_heads, block_size, head_dim)
            kv_cache_v: Tensor,
            position_ids: Tensor,     # prefill: (1, seq_len)  decode: (B, 1)
            slot_mapping: Tensor,     # (total_tokens,) int32
            is_prefill: bool,
            block_table: Tensor | None = None,   # decode: (B, MAX_BLOCKS)
            context_lens: Tensor | None = None,  # decode: (B,)
            ) -> Tensor:
        
        x = self.embed_tokens(input_ids)

        # position_ids 直接用传入的，不再从 kv_caches 计算
        cos, sin = self.rope.get_cos_sin(position_ids)

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x, cos, sin,
                kv_cache_k=kv_cache_k[layer_idx],  # 每层取自己的 slice
                kv_cache_v=kv_cache_v[layer_idx],
                slot_mapping=slot_mapping,
                is_prefill=is_prefill,
                block_table=block_table,
                context_lens=context_lens,
            )
        x = self.norm(x)
        return F.linear(x, self.embed_tokens.weight)

    def forward_decode(self,
            input_ids: Tensor,
            kv_cache_k: Tensor,       # (num_layers, num_blocks, n_kv_heads, block_size, head_dim)
            kv_cache_v: Tensor,
            position_ids: Tensor,     # prefill: (1, seq_len)  decode: (B, 1)
            slot_mapping: Tensor,     # (total_tokens,) int32
            block_table: Tensor,   # decode: (B, MAX_BLOCKS)
            context_lens: Tensor,  # decode: (B,)
            ) -> Tensor:
        
        x = self.embed_tokens(input_ids)

        # position_ids 直接用传入的，不再从 kv_caches 计算
        cos, sin = self.rope.get_cos_sin(position_ids)

        for layer_idx, layer in enumerate(self.layers):
            x = layer.forward_decode(# type:ignore
                x, cos, sin,
                kv_cache_k=kv_cache_k[layer_idx],  # 每层取自己的 slice
                kv_cache_v=kv_cache_v[layer_idx],
                slot_mapping=slot_mapping,
                block_table=block_table,
                context_lens=context_lens,
            )
        x = self.norm(x)
        return F.linear(x, self.embed_tokens.weight)
