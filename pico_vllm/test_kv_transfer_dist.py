# test_kv_transfer_dist.py
import torch
import torch.distributed as dist
import os
from model import ModelConfig
from cache import PagedKVCache, BlockManager
from kv_transfer import SyncKVTransfer
from scheduler import Request

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dtype = torch.bfloat16
    cfg = ModelConfig()
    BLOCK_SIZE = 16

    # 每卡各自的 BlockManager
    bm = BlockManager(
        num_gpu_blocks=100, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, dtype=dtype,
    )
    cache_kwargs = dict(
        block_manager=bm, num_layers=cfg.num_hidden_layers,
        max_seq_len=512, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, device=device, dtype=dtype,
    )

    peer_rank = 1 if rank == 0 else 0
    transfer = SyncKVTransfer(
        local_rank=rank, peer_rank=peer_rank, device=device,
        block_manager=bm, model_cfg=cfg,cache_kwargs=cache_kwargs,
    )

    seq_len = 37

    if rank == 0:
        # === Prefill 侧：创建 cache，填数据，发送 ===
        cache = PagedKVCache(**cache_kwargs)
        cache._allocate_for_prefill(seq_len)
        cache._seq_len = seq_len

        # 填入随机数据
        block_table = cache.get_block_table()
        for phys_id in block_table.tolist():
            bm.gpu_kv_cache[0, :, phys_id] = torch.randn(
                cfg.num_hidden_layers, cfg.num_key_value_heads, BLOCK_SIZE, cfg.head_dim,
                dtype=dtype, device=device,
            )
            bm.gpu_kv_cache[1, :, phys_id] = torch.randn(
                cfg.num_hidden_layers, cfg.num_key_value_heads, BLOCK_SIZE, cfg.head_dim,
                dtype=dtype, device=device,
            )

        # 记录原始数据用于后续对比
        original_k = bm.gpu_kv_cache[0, :, block_table].clone()
        original_v = bm.gpu_kv_cache[1, :, block_table].clone()

        request = Request(
            request_id=42, input_ids=[1]*seq_len,
            max_new_tokens=20, temperature=0.8, top_p=0.9,
            kv_cache=cache,
        )
        request.generated_ids = [100, 200]

        print(f"[Rank 0] Sending request: id={request.request_id}, seq_len={seq_len}, generated_ids={request.generated_ids}")
        transfer.send_request(request)
        print(f"[Rank 0] Send complete")

        # 把原始数据发给 rank 1 用于验证
        dist.send(original_k, dst=1)
        dist.send(original_v, dst=1)

    else:
        # === Decode 侧：接收 ===
        print(f"[Rank 1] Waiting for request...")
        recv_request = transfer.try_recv_request()
        print(f"[Rank 1] Received request: id={recv_request.request_id}")

        # 验证元数据
        assert recv_request.request_id == 42, f"request_id 错误: {recv_request.request_id}"
        assert recv_request.generated_ids == [100, 200], f"generated_ids 错误: {recv_request.generated_ids}"
        assert recv_request.temperature == 0.8, f"temperature 错误"
        assert recv_request.top_p == 0.9, f"top_p 错误"
        assert recv_request.kv_cache.seq_len == seq_len, f"seq_len 错误: {recv_request.kv_cache.seq_len}"
        print(f"✅ 元数据正确")

        # 接收 rank 0 的原始数据用于对比
        num_blocks_alloc = recv_request.kv_cache.allocated_cache_block_num
        original_k = torch.empty(
            cfg.num_hidden_layers, num_blocks_alloc, cfg.num_key_value_heads, BLOCK_SIZE, cfg.head_dim,
            dtype=dtype, device=device,
        )
        original_v = torch.empty_like(original_k)
        dist.recv(original_k, src=0)
        dist.recv(original_v, src=0)

        # 对比 KV Cache 数据
        block_table_d = recv_request.kv_cache.get_block_table()
        max_diff = 0.0
        for i in range(num_blocks_alloc):
            phys_d = block_table_d[i].item()
            diff_k = (original_k[:, i] - bm.gpu_kv_cache[0, :, phys_d]).abs().max().item()
            diff_v = (original_v[:, i] - bm.gpu_kv_cache[1, :, phys_d]).abs().max().item()
            max_diff = max(max_diff, diff_k, diff_v)

        print(f"KV Cache max diff: {max_diff}")
        if max_diff == 0.0:
            print("✅ send → recv KV Cache 完全一致")
        else:
            print("❌ KV Cache 数据不一致！")

        recv_request.kv_cache.reset()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()