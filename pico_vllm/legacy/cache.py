# 实现naive kv cache和paged kv cache两个版本，统一用kv cache类抽象
# attention 只调用这两个方法，不管内部怎么存
from abc import ABC, abstractmethod
import torch
from torch import Tensor, dtype
from blockmanager import BlockManager, pagedblocktype
from typing import List
import math

class KVCache(ABC):  # 抽象接口
    
    # ''' update: 更新指定 layer 的 KV，输入是当前 step 计算出的 K 和 V，shape (num_heads, head_dim)
    #     get: 获取指定 layer 的 KV，返回 shape (num_heads, head_dim) 的 K 和 V
    #     reset: 重置 KV cache，清空所有 KV'''
    # @abstractmethod
    # def update(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
    #     """
    #     将新的 k/v 写入 cache
    #     k, v: (1, num_kv_heads, head_dim)        ← decode 时
    #        或 (seq_len, num_kv_heads, head_dim)   ← prefill 时
    #     """
    #     ...
    # @abstractmethod
    # def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
    #     """
    #     返回当前层所有历史 token 的 k/v
    #     return: k, v 各 (seq_len, num_kv_heads, head_dim)
    #     """
    #     ...
    @abstractmethod
    def reset(self) -> None:
        """清空 cache，开始新的请求"""
        ...
    @property
    @abstractmethod
    def seq_len(self) -> int:
        """当前已缓存的 token 数量"""
        ...
    
# 朴素实现
class NaiveKVCache(KVCache):
    # 连续 tensor
    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_dim, device, dtype):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.cache = torch.zeros(
            num_layers, 2, max_seq_len, num_kv_heads, head_dim,
            device=device, dtype=dtype
        )
        self._seq_len = 0  # 当前已填入的长度

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        # k, v: (1, num_kv_heads, head_dim) 或 (seq_len, num_kv_heads, head_dim)
        # 通过读取kv的shape自动适配 prefill 和 decode 场景
        # 以后用 cuda kernel 的时候这个要怎么处理？可能需要分开 prefill 和 decode 的接口，或者额外传参？
        num_new_tokens = k.shape[0]
        start = self._seq_len
        end = self._seq_len + num_new_tokens

        self.cache[layer_idx, 0, start:end, :, :] = k
        self.cache[layer_idx, 1, start:end, :, :] = v
        if (layer_idx == self.num_layers - 1):  # 只在最后一层更新 seq_len，保证所有层的 seq_len 一致
            self._seq_len += num_new_tokens

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        return self.cache[layer_idx, 0, :self._seq_len, :, :], self.cache[layer_idx, 1, :self._seq_len, :, :]

    def reset(self) -> None:
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

# 分页的page Attention用的kv cache管理
class PagedKVCache():
    # block table + 物理块
    # max_block_len: int          # 最大的block数量
    def __init__(self, 
                 block_manager: BlockManager, 
                 num_layers: int, 
                 max_seq_len: int, 
                 num_kv_heads: int, 
                 head_dim: int, 
                 device, 
                 dtype: dtype):
        self.block_manager = block_manager
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device

        self.max_cache_block_num = math.ceil(max_seq_len / block_manager.block_size)
        self.cache_block_index : List[int] = [ (-1) for i in range(self.max_cache_block_num)]
        self.allocated_cache_block_num = 0
        self._seq_len = 0  # 当前已填入的长度
        self._updated_layer = 0 # 当前已填入的layer数量

        # 优化 1：直接缓存 physical_ids 的 Python List，加速标量查询
        self.physical_block_ids: List[int] = []
        
        # 优化 2：在 GPU 上预分配一条完整的 static block table
        self.gpu_block_table = torch.full(
            (self.max_cache_block_num,), -1, dtype=torch.int32, device=self.device
        )

    def prefill_update(self, layer_idx: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        一次性把 Prefill 阶段生成的全部 k/v 写入 block_manager
        k, v: (seq_len, num_kv_heads, head_dim)
        """
        prefill_len = k.shape[0]
        
        # 1. 如果是第一层 (layer_idx == 0)，统一结算并申请整个序列需要的块
        if self._updated_layer == 0:
            total_slots_needed = (self._seq_len + prefill_len)
            need_cache_block_num = math.ceil(total_slots_needed / self.block_manager.block_size)
            
            diff = need_cache_block_num - self.allocated_cache_block_num
            if diff > 0:
                new_blocks = self.block_manager.allocate(diff)
                for i, logical_id in enumerate(new_blocks):
                    self.cache_block_index[self.allocated_cache_block_num + i] = logical_id
                self.allocated_cache_block_num += diff

        # 2. 计算这批 token 对应的全局索引、逻辑块和偏移量
        # token_indices: [0, 1, 2, ..., prefill_len - 1]
        token_indices = torch.arange(prefill_len, device=self.device)
        global_slot_idxs = (self._seq_len + token_indices)
        
        block_idxs = global_slot_idxs // self.block_manager.block_size
        block_offsets = global_slot_idxs % self.block_manager.block_size
        
        # 3. 查表获取物理块 ID 和类型
        # 因为跨越多个 token，我们用列表收集映射结果
        physical_ids = []
        block_types = []
        for b_idx in block_idxs.tolist():
            logical_block_id = self.cache_block_index[b_idx]
            b_type, p_id = self.block_manager.block_mapping[logical_block_id]
            physical_ids.append(p_id)
            block_types.append(b_type)

        # 4. 分离 GPU 和 CPU 的索引，进行向量化批量写入
        # 这样做可以避免跨设备互相等待，且单次操作就能写完同设备的全部数据
        gpu_indices = [i for i, t in enumerate(block_types) if t == pagedblocktype.GPU]
        cpu_indices = [i for i, t in enumerate(block_types) if t == pagedblocktype.CPU]

        if gpu_indices:
            # 转换为 Tensor 以便进行 PyTorch 高级索引
            idx_t = torch.tensor(gpu_indices, dtype=torch.long, device=self.device)
            phys_t = torch.tensor([physical_ids[i] for i in gpu_indices], dtype=torch.long, device=self.device)
            offset_t = block_offsets[idx_t]

            # 批量写入 GPU
            self.block_manager.gpu_kv_cache[0, layer_idx, phys_t, :, offset_t, :] = k[idx_t]
            self.block_manager.gpu_kv_cache[1, layer_idx, phys_t, :, offset_t, :] = v[idx_t]

        if cpu_indices:
            # 转换为 Tensor，注意放在 CPU 上
            idx_t_cpu = torch.tensor(cpu_indices, dtype=torch.long, device='cpu')
            phys_t_cpu = torch.tensor([physical_ids[i] for i in cpu_indices], dtype=torch.long, device='cpu')
            offset_t_cpu = block_offsets[idx_t_cpu].cpu()

            # 批量写入 CPU (附带 to('cpu') 数据转移)
            self.block_manager.cpu_kv_cache[0, layer_idx, phys_t_cpu, :, offset_t_cpu, :] = k[idx_t_cpu].to('cpu')
            self.block_manager.cpu_kv_cache[1, layer_idx, phys_t_cpu, :, offset_t_cpu, :] = v[idx_t_cpu].to('cpu')

        # 如果出现未知类型，抛出异常
        if len(gpu_indices) + len(cpu_indices) != prefill_len:
            raise RuntimeError("尝试写入未分配物理空间（NONE）的逻辑块")

        # 5. 状态更新
        self._updated_layer += 1
        if self._updated_layer == self.num_layers:
            self._seq_len += prefill_len  # 只有所有层都写入完毕，序列长度才增加 prefill_len
            self._updated_layer = 0
    
    def prepare_decode_step(self) -> None:
        """
        在 decode forward 之前调用，预分配下一个 token 需要的 block
        这样 block_table 构造时就能包含新 block
        """
        total_needed = math.ceil((self._seq_len + 1) / self.block_manager.block_size)
        diff = total_needed - self.allocated_cache_block_num
        if diff > 0:
            new_blocks = self.block_manager.allocate(diff)
            for i, logical_id in enumerate(new_blocks):
                self.cache_block_index[self.allocated_cache_block_num + i] = logical_id
            self.allocated_cache_block_num += diff

    def update(self, layer_idx: Tensor, k: Tensor, v: Tensor) -> None:
        """
        把新的 k/v 写入 block_manager 的物理 KV cache
        k, v: (seq_len, num_kv_heads, head_dim)
        
        逻辑：
        1. 根据当前 _seq_len 计算写入位置
        block_idx = token_pos // block_size
        offset    = token_pos % block_size
        2. 查 block_table[block_idx] 得到逻辑 block_id
        3. 查 block_manager.block_mapping 得到物理 block_id
        4. 写入 block_manager.gpu_kv_cache[physical_id, k_or_v, head, offset, dim]
        """
        # 一个paged kv cache block的shape：(physical_block_idx, 2, num_kv_heads, block_size, head_dim)
        # 如果计算发现多出接下来一个token的所有层之后kv cache预留的数量不够，新申请直到足够
        if (self._updated_layer == 0):
            # 计算到当前 token 为止，总共需要的 slot 数量
            total_slots_needed = (self._seq_len + 1)
            # 计算总共需要的 block 数量
            need_cache_block_num = math.ceil(total_slots_needed / self.block_manager.block_size)
            
            # 如果已分配的块不够，向 block_manager 申请
            diff = need_cache_block_num - self.allocated_cache_block_num
            if diff > 0:
                new_blocks = self.block_manager.allocate(diff)
                for i, physical_id in enumerate(new_blocks):
                    self.cache_block_index[self.allocated_cache_block_num + i] = physical_id
                self.allocated_cache_block_num += diff
            
        # 逻辑：将 KV Cache 看作一维展平的空间，按 (token_pos, layer_idx) 顺序排列
        block_idx = self._seq_len // self.block_manager.block_size
        block_offset = self._seq_len % self.block_manager.block_size
        logical_block_id = self.cache_block_index[block_idx]
        # physical_block_id = self.block_manager.block_mapping[logical_block_id][1]
        # self.block_manager.gpu_kv_cache[physical_block_id, 0, :, block_offset, :] = k
        # self.block_manager.gpu_kv_cache[physical_block_id, 1, :, block_offset, :] = v
        block_type, physical_id = self.block_manager.block_mapping[logical_block_id]

        # 写入（确保此时是在 GPU）
        if block_type == pagedblocktype.GPU:
            # 使用 .view() 或 .squeeze() 确保维度对齐: (num_heads, head_dim)
            # 物理 cache 维度: [2, num_layers, num_blocks, block_size, num_heads, head_dim]
            self.block_manager.gpu_kv_cache[0, layer_idx, physical_id, :, block_offset, :] = k.squeeze(0)
            self.block_manager.gpu_kv_cache[1, layer_idx, physical_id, :, block_offset, :] = v.squeeze(0)
        elif block_type == pagedblocktype.CPU:
            # 如果逻辑块在 CPU，这里会产生严重的同步开销，通常做法是：
            # 在 update 之前由 BlockManager 确保活跃块已被 swap-in 到 GPU
            self.block_manager.cpu_kv_cache[0, layer_idx, physical_id, :, block_offset, :] = k.squeeze(0).to('cpu')
            self.block_manager.cpu_kv_cache[1, layer_idx, physical_id, :, block_offset, :] = v.squeeze(0).to('cpu')
        else:
            raise RuntimeError(f"尝试写入未分配物理空间的逻辑块: {logical_block_id}")
        
        self._updated_layer += 1
        if (self._updated_layer == self.num_layers):
            self._seq_len += 1
            self._updated_layer = 0

    def get_block_table(self) -> torch.Tensor:
        """
        读出当前所有已写入的 k/v
        return: 存储的所有对应index的物理index
        """
        physical_ids = []
        for lid in self.cache_block_index[:self.allocated_cache_block_num]:
            _, pid = self.block_manager.block_mapping[lid]
            physical_ids.append(pid)
        return torch.tensor(physical_ids, dtype=torch.int32, device=self.device)
        
    def get_prefill_slot_mapping(self, prefill_len: int) -> torch.Tensor:
        """
        计算本次 prefill 的 slot_mapping
        prefill_len: 本次 prefill 的 token 数
        """
        slots = []
        for i in range(prefill_len):
            token_pos = self._seq_len + i
            block_idx = token_pos // self.block_manager.block_size
            offset = token_pos % self.block_manager.block_size
            logical_id = self.cache_block_index[block_idx]
            _, physical_id = self.block_manager.block_mapping[logical_id]
            slot = physical_id * self.block_manager.block_size + offset
            slots.append(slot)
        return torch.tensor(slots, dtype=torch.int32, device=self.device)
    
    def _allocate_for_prefill(self, prefill_len: int) -> None:
        """提前分配 prefill 需要的所有 block"""
        total_needed = math.ceil((self._seq_len + prefill_len) / self.block_manager.block_size)
        diff = total_needed - self.allocated_cache_block_num
        if diff > 0:
            new_blocks = self.block_manager.allocate(diff)
            # 写入 cache_block_index（根据你用 list 还是 tensor 选对应写法）
            for i, logical_id in enumerate(new_blocks):
                self.cache_block_index[self.allocated_cache_block_num + i] = logical_id
            self.allocated_cache_block_num += diff
            
    def get_decode_slot(self) -> int:
        """
        返回 decode 新 token 的物理 slot（已经过 prepare_decode_step）
        """
        token_pos = self._seq_len  # prepare_decode_step 之后，block 已分配
        block_idx = token_pos // self.block_manager.block_size
        offset = token_pos % self.block_manager.block_size
        logical_id = self.cache_block_index[block_idx]
        _, physical_id = self.block_manager.block_mapping[logical_id]
        return physical_id * self.block_manager.block_size + offset
    
    def reset(self) -> None:
        """
        释放所有 block，归还给 block_manager
        """
        # 调用block manager并且释放非-1部分的idx
        self.block_manager.free(self.cache_block_index[:self.allocated_cache_block_num])
        # 重置kv cache状态
        self._seq_len = 0
        self._updated_layer = 0
        self.allocated_cache_block_num = 0
        self.cache_block_index : List[int] = [ (-1) for i in range(self.max_cache_block_num)]

    @property
    def seq_len(self) -> int:
        return self._seq_len
    

if __name__ == "__main__":
    import torch
    import math

    print("=" * 60)
    print("PagedKVCache 单元测试")
    print("=" * 60)

    # 小参数方便手算验证
    NUM_LAYERS = 2
    NUM_KV_HEADS = 2
    HEAD_DIM = 4
    BLOCK_SIZE = 4
    NUM_GPU_BLOCKS = 8
    NUM_CPU_BLOCKS = 4
    MAX_SEQ_LEN = 20
    DEVICE = 'cuda'
    DTYPE = torch.float16

    bm = BlockManager(
        num_gpu_blocks=NUM_GPU_BLOCKS,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        block_size=BLOCK_SIZE,
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
    )

    def make_cache():
        return PagedKVCache(
            block_manager=bm,
            num_layers=NUM_LAYERS,
            max_seq_len=MAX_SEQ_LEN,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            device=DEVICE,
            dtype=DTYPE,
        )

    # ===== 测试1：初始状态 =====
    print("\n[Test 1] 初始状态")
    cache = make_cache()
    assert cache.seq_len == 0
    assert cache.allocated_cache_block_num == 0
    assert cache._updated_layer == 0
    assert len(cache.cache_block_index) == math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)
    assert all(x == -1 for x in cache.cache_block_index)
    print("  ✓")

    # ===== 测试2：prefill_update 写入，验证数据 =====
    print("\n[Test 2] prefill_update 写入数据")
    PREFILL_LEN = 6  # 跨越两个 block（block_size=4）

    # 构造可识别的 k/v
    k_prefill = torch.arange(
        PREFILL_LEN * NUM_KV_HEADS * HEAD_DIM,
        dtype=DTYPE, device=DEVICE
    ).view(PREFILL_LEN, NUM_KV_HEADS, HEAD_DIM)
    v_prefill = k_prefill * 2

    # 写入所有层
    for layer in range(NUM_LAYERS):
        cache.prefill_update(torch.tensor(layer), k_prefill, v_prefill)

    assert cache.seq_len == PREFILL_LEN
    assert cache._updated_layer == 0
    # 6 tokens / block_size 4 → 需要 2 个 block
    assert cache.allocated_cache_block_num == 2
    # BlockManager 应该分配了 2 个逻辑块
    assert len(bm.gpu_free_blocks) == NUM_GPU_BLOCKS - 2

    # 验证数据写入正确：检查 token 0 的 layer 0
    lid = cache.cache_block_index[0]
    _, pid = bm.block_mapping[lid]
    k_read = bm.gpu_kv_cache[0, 0, pid, 0, :, :]  # layer=0, offset=0
    assert torch.allclose(k_read, k_prefill[0].to(DTYPE))
    # 检查 token 5（第二个 block 的 offset=1）
    lid2 = cache.cache_block_index[1]
    _, pid2 = bm.block_mapping[lid2]
    k_read2 = bm.gpu_kv_cache[0, 0, pid2, 1, :, :]  # offset = 5 % 4 = 1
    assert torch.allclose(k_read2, k_prefill[5].to(DTYPE))
    print(f"  prefill {PREFILL_LEN} tokens，写入正确 ✓")

    # ===== 测试3：decode update 写入单个 token =====
    print("\n[Test 3] decode update 写入")
    k_decode = torch.ones(1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 99
    v_decode = torch.ones(1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 88

    for layer in range(NUM_LAYERS):
        cache.update(torch.tensor(layer), k_decode, v_decode)

    assert cache.seq_len == PREFILL_LEN + 1  # 7
    # token 6 → block_idx=1, offset=2
    lid = cache.cache_block_index[1]
    _, pid = bm.block_mapping[lid]
    k_read = bm.gpu_kv_cache[0, 0, pid, 2, :, :]
    assert torch.allclose(k_read, k_decode.squeeze(0))
    print(f"  decode token {PREFILL_LEN}，写入正确 ✓")

    # ===== 测试4：decode 触发新 block 分配 =====
    print("\n[Test 4] decode 触发新 block 分配")
    blocks_before = cache.allocated_cache_block_num

    # 写到 token 7（还在 block 1 内），token 8 会进入 block 2
    for _ in range(8 - cache.seq_len):  # 补齐到 token 7
        for layer in range(NUM_LAYERS):
            cache.update(torch.tensor(layer), k_decode, v_decode)

    assert cache.allocated_cache_block_num == 2  # 还是 2 块

    # 写 token 8，触发第 3 块分配
    for layer in range(NUM_LAYERS):
        cache.update(torch.tensor(layer), k_decode, v_decode)

    assert cache.allocated_cache_block_num == 3
    print(f"  token 8 触发新 block 分配，allocated={cache.allocated_cache_block_num} ✓")

    # ===== 测试5：get_block_table =====
    print("\n[Test 5] get_block_table")
    bt = cache.get_block_table()
    assert bt.dtype == torch.int32
    assert bt.shape[0] == cache.allocated_cache_block_num
    assert bt.device.type == 'cuda'
    # 验证全是有效的物理 block id
    assert (bt >= 0).all()
    print(f"  block_table shape={bt.shape}, values={bt.tolist()} ✓")

    # ===== 测试6：reset =====
    print("\n[Test 6] reset")
    gpu_free_before = len(bm.gpu_free_blocks)
    allocated = cache.allocated_cache_block_num

    cache.reset()

    assert cache.seq_len == 0
    assert cache.allocated_cache_block_num == 0
    assert cache._updated_layer == 0
    assert all(x == -1 for x in cache.cache_block_index)
    # block 归还给 BlockManager
    assert len(bm.gpu_free_blocks) == gpu_free_before + allocated
    print(f"  reset 后状态正确，归还 {allocated} 个块 ✓")

    # ===== 测试7：reset 后可以复用 =====
    print("\n[Test 7] reset 后重新写入")
    k2 = torch.ones(3, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 42
    v2 = k2.clone()
    for layer in range(NUM_LAYERS):
        cache.prefill_update(torch.tensor(layer), k2, v2)
    assert cache.seq_len == 3
    print(f"  reset 后重新写入正确 ✓")

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)