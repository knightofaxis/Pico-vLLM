import torch

def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, vocab_size)
    return: (B,)
    """
    return torch.argmax(logits, dim=-1)


def sample_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logits: (B, vocab_size)
    return: (B,)
    """
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """
    logits: (B, vocab_size)  — 同一组内 temperature 和 top_p 相同
    return: (B,)
    """
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 把超过 top_p 的部分 mask 掉（保留第一个超过的位置）
    mask = cumulative_probs - sorted_probs >= top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    # 在 sorted 空间采样，再映射回原始 vocab index
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1)
    token_ids = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)
    return token_ids.squeeze(-1)


def sample_batch(
    logits: torch.Tensor,
    temperatures: list[float],
    top_ps: list[float],
) -> list[int]:
    """
    批量采样入口。对 (B, vocab_size) 的 logits 按策略分组处理。

    logits: (B, vocab_size)
    temperatures: 长度 B 的列表
    top_ps: 长度 B 的列表
    return: 长度 B 的 int 列表
    """
    B = logits.shape[0]
    results = [0] * B

    # 按策略分组：greedy / temperature-only / top_p
    greedy_idx = []
    temp_groups: dict[float, list[int]] = {}      # temperature -> [indices]
    topp_groups: dict[tuple, list[int]] = {}       # (temperature, top_p) -> [indices]

    for i in range(B):
        if temperatures[i] == 0:
            greedy_idx.append(i)
        elif top_ps[i] >= 1.0:
            temp_groups.setdefault(temperatures[i], []).append(i)
        else:
            key = (temperatures[i], top_ps[i])
            topp_groups.setdefault(key, []).append(i)

    # greedy：一次 argmax
    if greedy_idx:
        ids = sample_greedy(logits[greedy_idx])
        for j, idx in enumerate(greedy_idx):
            results[idx] = ids[j].item()

    # temperature sampling：按 temperature 值分组
    for temp, indices in temp_groups.items():
        ids = sample_temperature(logits[indices], temp)
        for j, idx in enumerate(indices):
            results[idx] = ids[j].item()

    # top_p sampling：按 (temperature, top_p) 分组
    for (temp, tp), indices in topp_groups.items():
        ids = sample_top_p(logits[indices], temp, tp)
        for j, idx in enumerate(indices):
            results[idx] = ids[j].item()

    return results


# 保留单请求接口（向后兼容，prefill 阶段之前使用的函数）
def sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    """
    统一入口：
    - temperature=0 → greedy
    - top_p=1.0     → pure temperature sampling
    - 其他          → top-p sampling
    logits: (vocab_size,)
    return: () 标量
    """
    logits = logits.squeeze()
    if temperature == 0:
        return torch.argmax(logits)
    elif top_p >= 1.0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze()
    else:
        return sample_top_p(logits.unsqueeze(0), temperature, top_p).squeeze()