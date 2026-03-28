# model.py
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from cache import KVCache, NaiveKVCache, PagedKVCache
from Attention import paged_decode_attention

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
    def __init__(self, cfg: ModelConfig, layer_idx):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        # k_proj: (B, seq_len, hidden_size) -> (B, seq_len, num_kv_heads * head_dim)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def _prefill_attention(self, q: Tensor, k: Tensor, v: Tensor, kv_cache: PagedKVCache):
        # 写入 KV cache
        kv_cache.prefill_update(self.layer_idx, k.squeeze(0), v.squeeze(0))
        
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

    def _decode_attention(self,
                      q: Tensor,  # (B, 1, n_heads, head_dim)
                      k: Tensor,  # (B, 1, n_kv_heads, head_dim)
                      v: Tensor,  # (B, 1, n_kv_heads, head_dim)
                      kv_caches: list[PagedKVCache],  # 每个请求的 cache
                      block_table: Tensor,  # (B, max_blocks)
                      context_lens: Tensor, # (B,)
                      ) -> Tensor:  # (B, 1, hidden)
        # TODO: 换成 Triton kernel
            # 写入 KV cache
        for i, cache in enumerate(kv_caches):
            cache.update(self.layer_idx, k[i].squeeze(0), v[i].squeeze(0))

        bm = kv_caches[0].block_manager
        k_cache = bm.gpu_kv_cache[0, self.layer_idx]  # 或 bm.gpu_kv_cache[0, self.layer_idx]
        v_cache = bm.gpu_kv_cache[1, self.layer_idx]
        
        out = paged_decode_attention(
            q, k_cache, v_cache,
            block_table, context_lens,
            MAX_BLOCKS_PER_SEQ=block_table.shape[1],
        )
        # out: (B, N_HEAD, 1, HEAD_DIM) → (B, 1, hidden_size)
        B = q.shape[0]
        return out.transpose(1, 2).contiguous().view(B, 1, self.cfg.hidden_size)
        # return self._gather_decode_attention(q, k, v, kv_caches, block_table, context_lens)

    def _gather_decode_attention(self, q, k, v, kv_caches: list[PagedKVCache], block_table: Tensor, context_lens: Tensor):
        # 临时 gather 实现，后面换 Triton
        # q/k/v: (B, 1, n_heads/n_kv_heads, head_dim)
        B = q.shape[0]
        block_size = kv_caches[0].block_manager.block_size
        
        # 写入 KV cache（每个请求单独写）
        for i, cache in enumerate(kv_caches):
            # k[i]: (1, 1, n_kv_heads, head_dim)
            cache.update(self.layer_idx, k[i].squeeze(0), v[i].squeeze(0))
        
        # 从 block_manager 的全局 kv_cache 里按 block_table gather
        bm = kv_caches[0].block_manager
        # gpu_kv_cache: (k/v, num_layers, num_blocks, n_kv_heads, block_size, head_dim)
        # k_global: (num_blocks, block_size, n_kv_heads, head_dim)
        k_global = bm.gpu_kv_cache[0, self.layer_idx]  # (num_blocks, n_kv_heads, block_size, head_dim)
        v_global = bm.gpu_kv_cache[1, self.layer_idx]
        
        outputs = []
        for i in range(B):
            seq_len = context_lens[i].item()
            num_blocks = (seq_len + block_size - 1) // block_size
            phys_blocks = block_table[i, :num_blocks]  # (num_blocks,) 物理 block id
            
            # # gather: (num_blocks, n_kv_heads, block_size, head_dim)
            # # k_global[phys_blocks]: (n_kv_heads, block_size, head_dim)
            # k_i = k_global[phys_blocks].transpose(0, 1).reshape(-1, self.cfg.num_key_value_heads, self.cfg.head_dim)[:seq_len]
            # v_i = v_global[phys_blocks].transpose(0, 1).reshape(-1, self.cfg.num_key_value_heads, self.cfg.head_dim)[:seq_len]
            
            # # GQA expand
            # # (seq_len, n_kv_heads, head_dim)->(seq_len, n_heads, head_dim)
            # k_i = k_i.repeat_interleave(self.cfg.num_kv_groups, dim=1)  # (seq_len, n_heads, head_dim)
            # v_i = v_i.repeat_interleave(self.cfg.num_kv_groups, dim=1)
            
            # # SDPA: (1, n_heads, 1, head_dim) x (1, n_heads, seq_len, head_dim)
            # # q[i]: (1, n_heads, head_dim)
            # q_i = q[i].transpose(0, 1).unsqueeze(0)  # (1, n_heads, 1, head_dim)
            # k_i = k_i.transpose(0, 1).unsqueeze(0)   # (1, n_heads, seq_len, head_dim)
            # v_i = v_i.transpose(0, 1).unsqueeze(0)
            
            # out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False)

            # k_global[phys_blocks]: (num_blocks_i, n_kv_heads, block_size, head_dim)

            # 目标：(1, n_heads, seq_len, head_dim) 供 SDPA 使用

            # 合并 num_blocks_i 和 block_size 两个维度，截断到 seq_len
            # (num_blocks_i, n_kv_heads, block_size, head_dim)
            # → (n_kv_heads, num_blocks_i * block_size, head_dim)  permute
            # → (n_kv_heads, seq_len, head_dim)                    截断
            # → (n_heads, seq_len, head_dim)                       GQA expand
            # → (1, n_heads, seq_len, head_dim)                    unsqueeze

            k_i = k_global[phys_blocks].permute(1, 0, 2, 3) \
                .reshape(self.cfg.num_key_value_heads, -1, self.cfg.head_dim)[:, :seq_len, :] \
                .repeat_interleave(self.cfg.num_kv_groups, dim=0) \
                .unsqueeze(0)

            v_i = v_global[phys_blocks].permute(1, 0, 2, 3) \
                .reshape(self.cfg.num_key_value_heads, -1, self.cfg.head_dim)[:, :seq_len, :] \
                .repeat_interleave(self.cfg.num_kv_groups, dim=0) \
                .unsqueeze(0)

            # q[i]: (1, n_heads, head_dim) → (1, n_heads, 1, head_dim)
            q_i = q[i].unsqueeze(2)  # unsqueeze seq 维度

            out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=False)

            # decode 不需要 causal mask，新 token attend to 所有历史
            outputs.append(out_i)  # (1, n_heads, 1, head_dim)
        
        out = torch.cat(outputs, dim=0)  # (B, n_heads, 1, head_dim)
        return out.transpose(1, 2).contiguous().view(B, 1, self.cfg.hidden_size)
    
    # def _gather_decode_attention(self, q, k, v, kv_caches, block_table, context_lens):
    #     B = q.shape[0]
    #     block_size = kv_caches[0].block_manager.block_size
    #     bm = kv_caches[0].block_manager
    #     n_kv_heads = self.cfg.num_key_value_heads
    #     head_dim = self.cfg.head_dim

    #     # 1. 写入 KV cache（仍然逐个，因为每个请求的 cache 对象独立）
    #     for i, cache in enumerate(kv_caches):
    #         cache.update(self.layer_idx, k[i].squeeze(0), v[i].squeeze(0))

    #     # 2. 确定 padding 目标长度
    #     max_seq_len = context_lens.max().item()
    #     max_blocks = (max_seq_len + block_size - 1) // block_size

    #     # 3. 向量化 gather
    #     # block_table: (B, max_blocks_total)，取前 max_blocks 列
    #     # -1 表示未分配，clamp 到 0 避免越界，后面用 mask 屏蔽
    #     safe_blocks = block_table[:, :max_blocks].clamp(min=0)  # (B, max_blocks)

    #     k_global = bm.gpu_kv_cache[0, self.layer_idx]  # (num_blocks, block_size, n_kv_heads, head_dim)
    #     v_global = bm.gpu_kv_cache[1, self.layer_idx]

    #     # (B, max_blocks, block_size, n_kv_heads, head_dim)
    #     k_gathered = k_global[safe_blocks]
    #     v_gathered = v_global[safe_blocks]

    #     # (B, max_blocks * block_size, n_kv_heads, head_dim) → 截到 max_seq_len
    #     k_gathered = k_gathered.reshape(B, -1, n_kv_heads, head_dim)[:, :max_seq_len]
    #     v_gathered = v_gathered.reshape(B, -1, n_kv_heads, head_dim)[:, :max_seq_len]

    #     # 4. GQA expand
    #     # (B, max_seq_len, n_heads, head_dim)
    #     k_gathered = k_gathered.repeat_interleave(self.cfg.num_kv_groups, dim=2)
    #     v_gathered = v_gathered.repeat_interleave(self.cfg.num_kv_groups, dim=2)

    #     # 5. 转置成 SDPA 期望格式
    #     # (B, n_heads, 1, head_dim)
    #     q_t = q.transpose(1, 2)
    #     # (B, n_heads, max_seq_len, head_dim)
    #     k_t = k_gathered.transpose(1, 2)
    #     v_t = v_gathered.transpose(1, 2)

    #     # 6. 构造 padding mask
    #     # positions: (1, max_seq_len)
    #     positions = torch.arange(max_seq_len, device=q.device).unsqueeze(0)
    #     # pad_mask: (B, max_seq_len)，True 表示需要屏蔽（超出 context_lens 的位置）
    #     pad_mask = positions >= context_lens.unsqueeze(1)
    #     # attn_bias: (B, 1, 1, max_seq_len)
    #     attn_bias = torch.zeros(B, 1, 1, max_seq_len, dtype=q.dtype, device=q.device)
    #     attn_bias.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

    #     # 7. 单次 batched SDPA
    #     out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_bias)
    #     # (B, n_heads, 1, head_dim) → (B, 1, hidden_size)
    #     return out.transpose(1, 2).contiguous().view(B, 1, self.cfg.hidden_size)
    
    def _triton_decode_attention(self, q, k, v, k_cache, v_cache, block_table, context_lens):
        """
        q:            (B, 1, n_heads, head_dim)
        k_cache:      (num_blocks, block_size, n_kv_heads, head_dim)  ← 全局，当前层的 view
        v_cache:      (num_blocks, block_size, n_kv_heads, head_dim)
        block_table:  (B, max_blocks) int32
        context_lens: (B,) int32
        return:       (B, 1, hidden_size)
        """
        # return paged_attention_triton(q, k_cache, v_cache, block_table, context_lens)

    def forward(self,
                x: Tensor,                    # (B, seq_len, hidden_size)
                cos: Tensor, sin: Tensor, 
                kv_caches: list[PagedKVCache],
                is_prefill: bool,
                # decode 阶段额外需要的信息，从外部传入
                block_table: Tensor | None = None,   # (B, max_blocks) int32，物理块id
                context_lens: Tensor | None = None,  # (B,) int32，每个请求当前长度
                attention_mask = None,
                ) -> Tensor:                  # (B, seq_len, hidden_size)
        B, seq_len, _ = x.shape
        
        # QKV projection（两条路径共用）
        q = self.q_proj(x).view(B, seq_len, self.cfg.num_attention_heads, self.cfg.head_dim)
        k = self.k_proj(x).view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)
        v = self.v_proj(x).view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)
        
        # RoPE（两条路径共用）
        q, k = RoPE.apply_rope(q, k, cos, sin)
        
        if is_prefill:
            output = self._prefill_attention(q, k, v, kv_caches[0])
        else:
            assert block_table is not None
            assert context_lens is not None
            output = self._decode_attention(q, k, v, kv_caches, block_table, context_lens)
        
        return self.o_proj(output)

class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        # x: (B, seq_len, hidden_size)
        gate = self.gate_proj(x)  # (B, seq_len, intermediate_size)
        up = self.up_proj(x)  # (B, seq_len, intermediate_size)
        return self.down_proj(F.silu(gate) * up)  # (B, seq_len, hidden_size)


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
        self.norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, 
                x, 
                cos: Tensor, sin: Tensor, 
                kv_caches: list[PagedKVCache], 
                is_prefill: bool, 
                block_table: Tensor | None = None,   # (B, max_blocks) int32，物理块id
                context_lens: Tensor | None = None,  # (B,) int32，每个请求当前长度, attention_mask=None):
                attention_mask = None):
        # x: (B, seq_len, hidden_size)
        attn_out = self.attn(self.norm1(x), cos, sin, kv_caches=kv_caches, is_prefill=is_prefill, block_table=block_table, context_lens=context_lens, attention_mask=attention_mask)  # (B, seq_len, hidden_size)
        x = x + attn_out  # 残差连接
        ffn_out = self.ffn(self.norm2(x))  # (B, seq_len, hidden_size)
        x = x + ffn_out  # 残差连接
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
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if not cfg.tie_word_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None  # 后面 forward 里会用 embed_tokens.weight 来做输出投影
    
    def forward(self, input_ids, kv_caches: list[PagedKVCache], is_prefill: bool=False, attention_mask=None):
        x = self.embed_tokens(input_ids)  # (B, seq_len, hidden_size)
        B, seq_len, _ = x.shape

        if is_prefill:
            # prefill: B=1，连续位置从 start_pos 开始
            start_pos = kv_caches[0].seq_len
            position_ids = torch.arange(
                start_pos, start_pos + seq_len,
                dtype=torch.long, device=x.device
            ).unsqueeze(0)  # (1, seq_len)
            block_table = None
            context_lens = None
        else:
            # decode: 每个请求位置不同，各自的 seq_len 就是当前 position
            position_ids = torch.tensor(
                [[c.seq_len] for c in kv_caches],
                dtype=torch.long, device=x.device
            )  # (B, 1)

            # 之后再预分配
            # 预分配，确保 block_table 包含新 token 的 block
            for c in kv_caches:
                c.prepare_decode_step()
                
            # 构造 block_table
            max_blocks = max(c.allocated_cache_block_num for c in kv_caches)
            # block_table = torch.full(
            #     (B, max_blocks), -1, dtype=torch.int32, device=x.device
            # )
            block_table = torch.full(
                (B, self.cfg.MAX_BLOCKS_PER_SEQ), -1,
                dtype=torch.int32, device=x.device
            )
            for i, c in enumerate(kv_caches):
                bt = c.get_block_table()
                block_table[i, :len(bt)] = bt
            for i, c in enumerate(kv_caches):
                bt = c.get_block_table()  # (allocated_blocks,)
                block_table[i, :len(bt)] = bt

            context_lens = torch.tensor(
                [c.seq_len + 1 for c in kv_caches],
                dtype=torch.int32, device=x.device
            )  # (B,)

        # 统一查表，得到 cos/sin
        cos, sin = self.rope.get_cos_sin(position_ids)
        # prefill:  cos/sin (1, seq_len, head_dim)
        # decode:   cos/sin (B, 1, head_dim)

        for layer in self.layers:
            x = layer(x, cos, sin, is_prefill=is_prefill,
                    kv_caches=kv_caches,
                    block_table=block_table,
                    context_lens=context_lens,
                    attention_mask=attention_mask)

        x = self.norm(x)
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = x @ self.embed_tokens.weight.t()
        return logits

