# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from cache import KVCache, NaiveKVCache

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
    def __init__(self, head_dim=128, rope_theta=1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        
    def precompute_freqs_cis(self, seq_len):
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(seq_len, dtype=torch.float)
        freqs_cis = torch.outer(t, freqs).float().view(seq_len, -1)
        freqs_cis = torch.polar(torch.ones_like(freqs_cis), freqs_cis)  # 转成复数表示
        return freqs_cis
    
    def apply_rope(self, x, freqs_cis):
        # x: (B, seq_len, num_heads, head_dim)
        # freqs_cis: (seq_len, head_dim//2) 复数表示
        # apply_rope 里加这一行，让意图更清晰
        freqs_cis = freqs_cis[None, :, None, :]  # (1, seq_len, 1, head_dim//2)
        x1 = x[..., :self.head_dim//2]  # 前半
        x2 = x[..., self.head_dim//2:self.head_dim]  # 后半
        x1_out = x1 * freqs_cis.real - x2 * freqs_cis.imag
        x2_out = x1 * freqs_cis.imag + x2 * freqs_cis.real
        # out: (B, seq_len, num_heads, head_dim)
        out = torch.empty_like(x)
        out[..., :self.head_dim//2] = x1_out
        out[..., self.head_dim//2:self.head_dim] = x2_out
        return out
    
    def forward(self, x):
        # x: (B, seq_len, num_heads, head_dim)
        seq_len = x.shape[1]
        freqs_cis = self.precompute_freqs_cis(seq_len).to(x.device)
        return self.apply_rope(x, freqs_cis)

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
        self.rope = RoPE(cfg.head_dim, cfg.rope_theta)

    def forward(self, x, freqs_cis, kv_cache: KVCache, attention_mask=None):
        assert kv_cache is NaiveKVCache
        # x: (B, seq_len, hidden_size)
        # 如果使用KV cache：
        # prefill: kv_cache.seq_len == 0，处理整个 prompt，kv_cache 还没有内容
        # decode:  kv_cache.seq_len != 0，只处理一个新 token，kv_cache有kv_cache.seq_len的内容
        B, seq_len, _ = x.shape
        # q_proj: (B, seq_len, hidden_size) -> (B, seq_len, num_heads, head_dim)
        # k_proj, v_proj: (B, seq_len, hidden_size) -> (B, seq_len, num_kv_heads, head_dim)
        q = self.q_proj(x).view(B, seq_len, self.cfg.num_attention_heads, self.cfg.head_dim)
        k = self.k_proj(x).view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)
        v = self.v_proj(x).view(B, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim)

        # RoPE
        # 如果使用 KV cache，位置编码起始位置由 KV cache 的 seq_len 决定
        # prefill: seq_len > 1，位置编码从 0 开始
        # decode: seq_len = 1，位置编码从当前已缓存的 seq_len 开始
        # 在外部传入的时候自动处理了所以不再需要分支了
        q = self.rope.apply_rope(q, freqs_cis)
        k = self.rope.apply_rope(k, freqs_cis)
        # 此时的shape为(B, seq_len, num_kv_heads, head_dim),恰好适合后续写入 KV cache
        # 取出历史 KV
        # k_history, v_history: (seq_len_cache, num_kv_heads, head_dim)
        k_history, v_history = kv_cache.get(self.layer_idx)
        # (seq_len_cache, num_kv_heads, head_dim) -> (1, seq_len_cache, num_kv_heads, head_dim)
        k_history = k_history.unsqueeze(0) 
        v_history = v_history.unsqueeze(0)
        # 目前只处理B=1的情况，所以直接 squeeze 掉 batch 维度
        k_to_store = k.squeeze(0)
        v_to_store = v.squeeze(0)
        kv_cache.update(self.layer_idx, k_to_store, v_to_store)
        # 拼接：历史 + 新的
        # (1, seq_len_cache, num_kv_heads, head_dim) + (1, seq_len_new, num_kv_heads, head_dim) -> (1, seq_len_total, num_kv_heads, head_dim)
        k = torch.cat([k_history, k], dim=1)
        v = torch.cat([v_history, v], dim=1)
        # q: (B, seq_len, num_heads, head_dim) -> (B, num_heads, seq_len, head_dim)
        # k: (B, seq_len, num_kv_heads, head_dim) -> (B, num_kv_heads, seq_len, head_dim)
        # v: (B, seq_len, num_kv_heads, head_dim) -> (B, num_kv_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # broadcast k 和 v 以匹配 q 的 num_heads
        # k, v: (B, num_kv_heads, seq_len, head_dim) -> (B, num_heads, seq_len, head_dim)
        k = k.repeat_interleave(self.cfg.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.cfg.num_kv_groups, dim=1)
        # attn_scores: (B, num_heads, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.cfg.head_dim ** 0.5)
        # 应用 attention_mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        # 否则应用 causal mask
        else:
            # 当有Prefix Cache的时候，q_len和k_len即使是在prefill阶段也不相等，q_len < k_len
            q_len = q.shape[2]    # transpose 之后
            k_len = k.shape[2]
            if q_len > 1:
                causal_mask = torch.full((q_len, k_len), float('-inf'), device=x.device, dtype=x.dtype)
                causal_mask = torch.triu(causal_mask, diagonal=k_len - q_len + 1)
                attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
            # if seq_len > 1:  # seq_len=1 时不需要 mask
            #     causal_mask = torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=x.dtype)
            #     causal_mask = torch.triu(causal_mask, diagonal=1)  # 上三角为 -inf，对角线及以下为 0
            #     attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        # attn_probs: (B, num_heads, seq_len, seq_len)
        attn_probs = F.softmax(attn_scores, dim=-1)
        # attn_output: (B, num_heads, seq_len, head_dim)
        attn_output = attn_probs @ v
        # attn_output: (B, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.cfg.hidden_size)
        # out_proj: (B, seq_len, hidden_size)
        # output: (B, seq_len, hidden_size)
        output = self.o_proj(attn_output)
        return output

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

    def forward(self, x, freqs_cis, kv_cache, attention_mask=None):
        # x: (B, seq_len, hidden_size)
        attn_out = self.attn(self.norm1(x), freqs_cis, kv_cache=kv_cache, attention_mask=attention_mask)  # (B, seq_len, hidden_size)
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

    def forward(self, input_ids, kv_cache, attention_mask=None):
        # input_ids: (B, seq_len)
        x = self.embed_tokens(input_ids)  # (B, seq_len, hidden_size)
        start_pos = kv_cache.seq_len if kv_cache is not None else 0
        total_len = start_pos + x.shape[1]
        freqs_cis = self.rope.precompute_freqs_cis(total_len)[start_pos:total_len].to(x.device)
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, kv_cache=kv_cache, attention_mask=attention_mask)
        x = self.norm(x)  # (B, seq_len, hidden_size)
        if self.lm_head is not None:
            logits = self.lm_head(x)  # (B, seq_len, vocab_size)
        else:
            logits = x @ self.embed_tokens.weight.t()  # (B, seq_len, vocab_size) 使用词嵌入权重做输出投影
        return logits

if __name__ == "__main__":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # from model import Qwen25_15B, ModelConfig

    torch.manual_seed(42)
    cfg = ModelConfig()
    hf_model = AutoModelForCausalLM.from_pretrained("./weights", torch_dtype=torch.float32)
    hf_model.eval()

    my_model = Qwen25_15B(cfg)

    # print(type(my_model.layers[0]))
    # print(type(my_model.layers))

    # 复制所有权重
    my_model.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.float()
    my_model.norm.weight.data = hf_model.model.norm.weight.data.float()

    for i in range(cfg.num_hidden_layers):
        hf_layer = hf_model.model.layers[i]
        my_layer = my_model.layers[i]

        assert isinstance(my_layer, TransformerBlock)

        my_layer.attn.q_proj.weight.data = hf_layer.self_attn.q_proj.weight.data.float()
        my_layer.attn.q_proj.bias.data   = hf_layer.self_attn.q_proj.bias.data.float()
        my_layer.attn.k_proj.weight.data = hf_layer.self_attn.k_proj.weight.data.float()
        my_layer.attn.k_proj.bias.data   = hf_layer.self_attn.k_proj.bias.data.float()
        my_layer.attn.v_proj.weight.data = hf_layer.self_attn.v_proj.weight.data.float()
        my_layer.attn.v_proj.bias.data   = hf_layer.self_attn.v_proj.bias.data.float()
        my_layer.attn.o_proj.weight.data = hf_layer.self_attn.o_proj.weight.data.float()

        my_layer.ffn.gate_proj.weight.data = hf_layer.mlp.gate_proj.weight.data.float()
        my_layer.ffn.up_proj.weight.data   = hf_layer.mlp.up_proj.weight.data.float()
        my_layer.ffn.down_proj.weight.data = hf_layer.mlp.down_proj.weight.data.float()

        my_layer.norm1.weight.data = hf_layer.input_layernorm.weight.data.float()
        my_layer.norm2.weight.data = hf_layer.post_attention_layernorm.weight.data.float()

    my_model.eval()

    # 用真实 token 测试
    tokenizer = AutoTokenizer.from_pretrained("./weights")
    text = "The capital of France is"
    input_ids = tokenizer(text, return_tensors="pt").input_ids  # (1, seq_len)

    with torch.no_grad():
        my_logits = my_model(input_ids)
        hf_logits = hf_model(input_ids).logits.float()

    diff = (my_logits - hf_logits).abs()
    print(f"最大误差: {diff.max().item():.2e}")
    print(f"平均误差: {diff.mean().item():.2e}")

    # greedy decoding 对比
    my_next = my_logits[0, -1].argmax().item()
    hf_next = hf_logits[0, -1].argmax().item()
    print(f"我的下一个 token: {tokenizer.decode([my_next])!r}")
    print(f"HF 的下一个 token: {tokenizer.decode([hf_next])!r}")
    print("通过!" if my_next == hf_next else "❌ token 不一致")