# test_kv_transfer.py
import torch
from model import ModelConfig
from cache import PagedKVCache, BlockManager
from kv_transfer import SyncKVTransfer
from scheduler import Request

def test_gather_scatter_roundtrip():
    """验证 gather → scatter round-trip：block 内容完全一致"""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cfg = ModelConfig()
    BLOCK_SIZE = 16

    # 两个独立 BlockManager，模拟 P 卡和 D 卡
    bm_p = BlockManager(
        num_gpu_blocks=100, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, dtype=dtype,
    )
    bm_d = BlockManager(
        num_gpu_blocks=100, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, dtype=dtype,
    )

    cache_kwargs_p = dict(
        block_manager=bm_p, num_layers=cfg.num_hidden_layers,
        max_seq_len=512, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, device=device, dtype=dtype,
    )
    cache_kwargs_d = dict(
        block_manager=bm_d, num_layers=cfg.num_hidden_layers,
        max_seq_len=512, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, device=device, dtype=dtype,
    )

    transfer_p = SyncKVTransfer(
        local_rank=0, peer_rank=1, device=device,
        block_manager=bm_p, model_cfg=cfg
    )
    transfer_d = SyncKVTransfer(
        local_rank=1, peer_rank=0, device=device,
        block_manager=bm_d, model_cfg=cfg
    )

    # 模拟 prefill：分配 block 并填入随机数据
    seq_len = 37  # 故意不对齐 block_size
    cache_p = PagedKVCache(**cache_kwargs_p)
    cache_p._allocate_for_prefill(seq_len)
    cache_p._seq_len = seq_len

    # 往 prefill 侧的 block 里写随机数据
    block_table_p = cache_p.get_block_table()
    num_blocks_alloc = len(block_table_p)
    print(f"seq_len={seq_len}, block_size={BLOCK_SIZE}, num_blocks_alloc={num_blocks_alloc}")

    for phys_id in block_table_p.tolist():
        bm_p.gpu_kv_cache[0, :, phys_id] = torch.randn(
            cfg.num_hidden_layers, cfg.num_key_value_heads, BLOCK_SIZE, cfg.head_dim,
            dtype=dtype, device=device
        )
        bm_p.gpu_kv_cache[1, :, phys_id] = torch.randn(
            cfg.num_hidden_layers, cfg.num_key_value_heads, BLOCK_SIZE, cfg.head_dim,
            dtype=dtype, device=device
        )

    # 构造 fake request
    request = Request(
        request_id=0, input_ids=[1]*seq_len,
        max_new_tokens=20, temperature=0, top_p=1.0,
        kv_cache=cache_p,
    )

    # === 测试 gather ===
    kv_data = transfer_p._gather_kv_cache(request)
    expected_shape = (2, cfg.num_hidden_layers, num_blocks_alloc, cfg.num_key_value_heads, BLOCK_SIZE, cfg.head_dim)
    assert kv_data.shape == expected_shape, f"gather shape 错误: {kv_data.shape} != {expected_shape}"
    print(f"✅ gather shape 正确: {kv_data.shape}")

    # === 测试 scatter ===
    cache_d = transfer_d._scatter_kv_cache(kv_data, seq_len)
    assert cache_d.seq_len == seq_len, f"scatter seq_len 错误: {cache_d.seq_len} != {seq_len}"
    assert cache_d.allocated_cache_block_num == num_blocks_alloc, \
        f"scatter block 数错误: {cache_d.allocated_cache_block_num} != {num_blocks_alloc}"
    print(f"✅ scatter seq_len={cache_d.seq_len}, blocks={cache_d.allocated_cache_block_num}")

    # === 逐 block 对比 ===
    block_table_d = cache_d.get_block_table()
    max_diff = 0.0

    for i in range(num_blocks_alloc):
        phys_p = block_table_p[i].item()
        phys_d = block_table_d[i].item()

        # 对比所有层的 K 和 V
        diff_k = (bm_p.gpu_kv_cache[0, :, phys_p] - bm_d.gpu_kv_cache[0, :, phys_d]).abs().max().item()
        diff_v = (bm_p.gpu_kv_cache[1, :, phys_p] - bm_d.gpu_kv_cache[1, :, phys_d]).abs().max().item()
        max_diff = max(max_diff, diff_k, diff_v)

    print(f"Max diff: {max_diff}")
    if max_diff == 0.0:
        print("✅ gather → scatter round-trip 完全一致")
    else:
        print("❌ 数据不一致！")

    cache_p.reset()
    cache_d.reset()


if __name__ == "__main__":
    test_gather_scatter_roundtrip()