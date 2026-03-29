# weights.py
import torch
from safetensors import safe_open
import glob
import os

def load_weights(model, weight_dir: str):
    """从 safetensors 文件加载权重到 model"""
    
    # 读取所有权重
    safetensor_files = sorted(glob.glob(os.path.join(weight_dir, "*.safetensors")))
    state_dict = {}
    for f in safetensor_files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for key in st.keys():
                state_dict[key] = st.get_tensor(key).float()

    # embed_tokens
    model.embed_tokens.weight.data = state_dict["model.embed_tokens.weight"]
    model.norm.weight.data = state_dict["model.norm.weight"]

    # 28 个 layer
    for i in range(model.cfg.num_hidden_layers):
        layer = model.layers[i]
        p = f"model.layers.{i}"

        # layer.attn.q_proj.weight.data = state_dict[f"{p}.self_attn.q_proj.weight"]
        # layer.attn.q_proj.bias.data   = state_dict[f"{p}.self_attn.q_proj.bias"]
        # layer.attn.k_proj.weight.data = state_dict[f"{p}.self_attn.k_proj.weight"]
        # layer.attn.k_proj.bias.data   = state_dict[f"{p}.self_attn.k_proj.bias"]
        # layer.attn.v_proj.weight.data = state_dict[f"{p}.self_attn.v_proj.weight"]
        # layer.attn.v_proj.bias.data   = state_dict[f"{p}.self_attn.v_proj.bias"]
        layer.attn.o_proj.weight.data = state_dict[f"{p}.self_attn.o_proj.weight"]
        # weights.py 里对每一层
        q_w = state_dict[f"{p}.self_attn.q_proj.weight"]  # (q_size, hidden)
        k_w = state_dict[f"{p}.self_attn.k_proj.weight"]  # (k_size, hidden)
        v_w = state_dict[f"{p}.self_attn.v_proj.weight"]  # (v_size, hidden)
        layer.attn.qkv_proj.weight.data = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = state_dict[f"{p}.self_attn.q_proj.bias"]
        k_b = state_dict[f"{p}.self_attn.k_proj.bias"]
        v_b = state_dict[f"{p}.self_attn.v_proj.bias"]
        layer.attn.qkv_proj.bias.data = torch.cat([q_b, k_b, v_b], dim=0)

        # layer.ffn.gate_proj.weight.data = state_dict[f"{p}.mlp.gate_proj.weight"]
        # layer.ffn.up_proj.weight.data   = state_dict[f"{p}.mlp.up_proj.weight"]
        gate_w = state_dict[f"{p}.mlp.gate_proj.weight"]  # (q_size, hidden)
        up_w = state_dict[f"{p}.mlp.up_proj.weight"]  # (k_size, hidden)
        layer.ffn.gate_up_proj.weight.data = torch.cat([gate_w, up_w], dim=0)
        layer.ffn.down_proj.weight.data = state_dict[f"{p}.mlp.down_proj.weight"]

        layer.norm1.weight.data = state_dict[f"{p}.input_layernorm.weight"]
        layer.norm2.weight.data = state_dict[f"{p}.post_attention_layernorm.weight"]

    print(f"权重加载完成，共 {len(state_dict)} 个 tensor")
    return model

# 在 weights.py 末尾测试
if __name__ == "__main__":
    from model import Qwen25_15B, ModelConfig
    cfg = ModelConfig()
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights")
    print("加载成功")