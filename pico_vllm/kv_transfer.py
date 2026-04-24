# kv_transfer.py
"""
KV Cache 传输层，负责 PD 分离时的 KV Cache 和请求元数据的跨卡传输。
设计为可替换：Phase 1 同步实现，Phase 2 替换为异步实现，Engine 不感知。
"""

from enum import Enum
import torch
import torch.distributed as dist
import pickle
from abc import ABC, abstractmethod

from blockmanager import BlockManager
from cache import KVCache, PagedKVCache
from scheduler import Request
from model import ModelConfig

class RecvState(Enum):
    """一个正在接收的请求的状态机"""
    IDLE = 0           # 没在接收
    WAIT_SIZE = 1      # 等 size_tensor 完成
    WAIT_META = 2      # 等 meta 完成
    WAIT_KV = 3        # 等 kv_data 完成
    # DONE = 4           # 全部完成

class KVTransferBase(ABC):
    """传输层接口，Engine 只依赖这个"""

    @abstractmethod
    def send_request(self, request):
        """Prefill 侧：发送请求元数据 + KV Cache"""
        ...

    @abstractmethod
    def try_recv_request(self) -> Request | None:
        """Decode 侧：尝试接收请求，返回 生成了一个token，可以直接用于Decode的Request 或 None"""
        ...

    @abstractmethod
    def poll(self):
        """每步调用，处理异步状态更新（Phase 1 里是 no-op）"""
        ...


class NoOpKVTransfer(KVTransferBase):
    """role='pd' 时使用，所有操作都是 no-op，永远不会被实际调用"""
    recv_done = False  # decode 侧：P 是否已经发完所有请求
    pending_sends = []

    def send_request(self, request):
        raise RuntimeError("NoOpKVTransfer.send_request should never be called (role=pd)")

    def try_recv_request(self) -> Request | None:
        return None

    def send_done(self):
        """告诉 decode 侧没有更多请求了"""
        pass
    
    def poll(self):
        pass

class SyncKVTransfer(KVTransferBase):
    """Phase 1：同步阻塞传输"""
    def __init__(self, local_rank, peer_rank, device,
                 block_manager:BlockManager, model_cfg:ModelConfig, cache_kwargs:dict):
        self.local_rank = local_rank
        self.peer_rank = peer_rank
        self.device = device
        self.block_manager = block_manager
        self.cfg = model_cfg
        self.cache_kwargs = cache_kwargs
        self.dtype = block_manager.dtype
        self.recv_done = False  # decode 侧：P 是否已经发完所有请求

    def _gather_kv_cache(self, request: Request) -> torch.Tensor:
        block_table = request.kv_cache.get_block_table()
        # 按块传递而非按seq_len传递，省去转置的开销
        # 一个block大约448KB
        return self.block_manager.gpu_kv_cache[:, :, block_table].contiguous()
        # shape: (2, num_layers, num_blocks_alloc, num_kv_heads, block_size, head_dim)

    def _scatter_kv_cache(self, kv_data: torch.Tensor, seq_len: int) -> PagedKVCache:
        num_blocks_alloc = kv_data.shape[2]
        cache = PagedKVCache(
            block_manager=self.block_manager,
            num_layers=self.cfg.num_hidden_layers,
            max_seq_len=self.cache_kwargs['max_seq_len'],
            num_kv_heads=self.cfg.local_num_key_value_heads,
            head_dim=self.cfg.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        cache._allocate_for_prefill(num_blocks_alloc * self.block_manager.block_size)
        # 需要 seq_len 个slot，分配则上取整到块
        block_table = cache.get_block_table()
        self.block_manager.gpu_kv_cache[:, :, block_table] = kv_data
        cache._seq_len = seq_len
        return cache

    def send_request(self, request):
        """阻塞发送：先发元数据，再发 KV Cache"""
        # gather KV cache数据，以准备发送
        kv_data = self._gather_kv_cache(request)

        # 生成元数据
        meta = {
            'request_id': request.request_id,
            'input_ids': request.input_ids,
            'generated_ids': request.generated_ids,
            'max_new_tokens': request.max_new_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'seq_len': request.kv_cache.seq_len,
            'kv_shape': list(kv_data.shape),
        }

        # 发送元数据长度 + 元数据
        data = pickle.dumps(meta)
        size_tensor = torch.tensor([len(data)], dtype=torch.long, device=self.device)
        dist.send(size_tensor, dst=self.peer_rank)

        data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(self.device)
        dist.send(data_tensor, dst=self.peer_rank)

        # 发送 KV Cache
        dist.send(kv_data, dst=self.peer_rank)

    def try_recv_request(self) -> Request | None:
        """阻塞接收：先收元数据，再收 KV Cache"""
        # 已经结束，则直接返回
        if self.recv_done:
            return None
        # 接收元数据长度
        size_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
        dist.recv(size_tensor, src=self.peer_rank)
        size = (int)(size_tensor.item())

        if size == 0:
            self.recv_done = True  # 标记：不再尝试接收
            return None

        # 接收元数据
        data_tensor = torch.zeros(size, dtype=torch.uint8, device=self.device)
        dist.recv(data_tensor, src=self.peer_rank)
        meta = pickle.loads(data_tensor.cpu().numpy().tobytes())

        # 接收 KV Cache
        kv_shape = meta['kv_shape']
        kv_data = torch.empty(
            kv_shape,
            dtype=self.dtype,
            device=self.device,
        )
        dist.recv(kv_data, src=self.peer_rank)

        # 创建 Request
        request = Request(
            request_id=meta['request_id'],
            input_ids=meta['input_ids'],
            generated_ids=meta['generated_ids'],
            max_new_tokens=meta['max_new_tokens'],
            temperature=meta['temperature'],
            top_p=meta['top_p'],
            kv_cache=self._scatter_kv_cache(kv_data, meta['seq_len'])
        )

        return request
    
    def send_done(self):
        """告诉 decode 侧没有更多请求了"""
        size_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.send(size_tensor, dst=self.peer_rank)

    def poll(self):
        """同步模式下无需 poll"""
        pass


class AsyncKVTransfer(KVTransferBase):
    """Phase 2：异步非阻塞传输"""

    def __init__(self, local_rank, 
                 device, 
                 block_manager, 
                 model_cfg,
                 peer_ranks: list[int],       # 要发送/接受的对侧rank
                 local_tp_size: int = 1,      # 本侧并行度
                 remote_tp_size: int = 1,     # 对侧并行度
                 is_primary: bool = True,     # 是否负责meta传输
                 role:str='pd',
                 dtype=torch.bfloat16, 
                 cache_cls:type[PagedKVCache]=PagedKVCache, 
                 cache_kwargs: dict|None = None, 
                 ):
        self.local_rank = local_rank
        self.peer_ranks = peer_ranks
        self.local_tp_size = local_tp_size
        self.remote_tp_size = remote_tp_size
        self.is_primary = is_primary

        self.device = device
        self.block_manager = block_manager
        self.cfg = model_cfg
        self.dtype = dtype
        self.cache_cls = cache_cls
        self.cache_kwargs = cache_kwargs
        self.recv_done = False
        if role == "p":
            self.recv_done = True

        # === P 侧状态 ===
        # 每个元素: (kv_data_tensor, [handle1, handle2, handle3])
        # 需要持有 tensor 引用直到 isend 完成
        self.pending_sends = []
        self._done_signal_sent = False

        # === D 侧状态 ===
        self._state = RecvState.IDLE
        self._size_buf = torch.zeros(1, dtype=torch.long, device=device)
        self._size_handle = None
        self._meta_buf = None
        self._meta_handle = None
        self._kv_buf = None
        self._kv_handle = None
        self._current_meta = None
        self.completed_requests = []

    # =========================================================
    # gather / scatter 复用，和 SyncKVTransfer 完全一样
    # =========================================================
    def _gather_kv_cache(self, request):
        block_table = request.kv_cache.get_block_table()
        return self.block_manager.gpu_kv_cache[:, :, block_table].contiguous()

    def _scatter_kv_cache(self, kv_data, seq_len):
        num_blocks_alloc = kv_data.shape[2]
        cache = self.cache_cls(**self.cache_kwargs) #type:ignore
        cache._allocate_for_prefill(num_blocks_alloc * self.block_manager.block_size)
        block_table = cache.get_block_table()
        self.block_manager.gpu_kv_cache[:, :, block_table] = kv_data
        cache._seq_len = seq_len
        return cache

    # =========================================================
    # P 侧：异步发送
    # =========================================================
    def send_request(self, request):
        kv_data = self._gather_kv_cache(request)
        # kv_data shape: (2, num_layers, num_blocks, local_kv_heads, block_size, head_dim)

        # 一对多时，split 沿 head 维度
        if len(self.peer_ranks) > 1:
            kv_chunks = kv_data.chunk(len(self.peer_ranks), dim=3)
        else:
            kv_chunks = [kv_data]

        for i, (peer, chunk) in enumerate(zip(self.peer_ranks, kv_chunks)):
            chunk = chunk.contiguous()

            if self.is_primary:
                # 只有 primary 向 peer 发 meta
                meta = {
                    'request_id': request.request_id,
                    'input_ids': request.input_ids,
                    'generated_ids': list(request.generated_ids),
                    'max_new_tokens': request.max_new_tokens,
                    'temperature': request.temperature,
                    'top_p': request.top_p,
                    'seq_len': request.kv_cache.seq_len,
                    'kv_shape': list(chunk.shape),
                }
                data = pickle.dumps(meta)
                size_tensor = torch.tensor([len(data)], dtype=torch.long, device=self.device)
                data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(self.device)

                h_size = dist.isend(size_tensor, dst=peer)
                h_meta = dist.isend(data_tensor, dst=peer)
                h_kv = dist.isend(chunk, dst=peer)
                self.pending_sends.append({
                    'handles': [h_size, h_meta, h_kv],
                    'tensors': [size_tensor, data_tensor, chunk],
                })
            else:
                # 非 primary （多对一时的副 P rank）：只发 kv
                h_kv = dist.isend(chunk, dst=peer)
                self.pending_sends.append({
                    'handles': [h_kv],
                    'tensors': [chunk],
                })

    def send_done(self):
        if not self.is_primary:
            self._done_signal_sent = True
            return

        for peer in self.peer_ranks:
            size_tensor = torch.tensor([0], dtype=torch.long, device=self.device)
            h = dist.isend(size_tensor, dst=peer)
            self.pending_sends.append({
                'handles': [h],
                'tensors': [size_tensor],
            })
        self._done_signal_sent = True

    # =========================================================
    # D 侧：状态机驱动的异步接收
    # =========================================================
    def try_recv_request(self):
        """非阻塞：从已完成队列取，没有就返回 None"""
        if self.completed_requests:
            return self.completed_requests.pop(0)
        return None

    # =========================================================
    # poll：两侧都调用，推进各自的异步状态
    # =========================================================
    def poll(self):
        self._poll_sends()
        self._poll_recvs()

    def _poll_sends(self):
        """P 侧：检查哪些发送完成了，释放 tensor 引用"""
        still_pending = []
        for entry in self.pending_sends:
            all_done = all(h.is_completed() for h in entry['handles'])
            if not all_done:
                still_pending.append(entry)
            # all_done 时不再持有引用，tensor 可以被 GC
        self.pending_sends = still_pending

    def _poll_recvs(self):
        """D 侧：状态机，每次 poll 取出所有已经完成的传输"""
        if self.recv_done:
            return

        while True:
            progress = False

            # 阶段 0 → 1：从 primary peer 收 size
            if self._state == RecvState.IDLE:
                self._size_buf.zero_()
                self._size_handle = dist.irecv(self._size_buf, src=self.peer_ranks[0])
                self._state = RecvState.WAIT_SIZE
                break

            # 阶段 1 → 2：从 primary peer 收 meta
            if self._state == RecvState.WAIT_SIZE:
                if not self._size_handle.is_completed():
                    break
                size = int(self._size_buf.item())
                if size == 0:
                    self.recv_done = True
                    self._state = RecvState.IDLE
                    return
                self._meta_buf = torch.zeros(size, dtype=torch.uint8, device=self.device)
                self._meta_handle = dist.irecv(self._meta_buf, src=self.peer_ranks[0])
                self._state = RecvState.WAIT_META
                progress = True

            # 阶段 2 → 3：解析 meta，然后挂 ALL peers 的 kv irecv
            if self._state == RecvState.WAIT_META:
                if not self._meta_handle.is_completed():
                    break
                self._current_meta = pickle.loads(self._meta_buf.cpu().numpy().tobytes())
                kv_shape = self._current_meta['kv_shape']

                # 为每个 peer 分配一个 kv buffer 并挂 irecv
                self._kv_bufs = []
                self._kv_handles = []
                for peer in self.peer_ranks:
                    buf = torch.empty(kv_shape, dtype=self.dtype, device=self.device)
                    handle = dist.irecv(buf, src=peer)
                    self._kv_bufs.append(buf)
                    self._kv_handles.append(handle)

                self._state = RecvState.WAIT_KV
                progress = True

            # 阶段 3 → 0：等所有 peer 的 kv 都到齐，cat，scatter
            if self._state == RecvState.WAIT_KV:
                if not all(h.is_completed() for h in self._kv_handles):
                    break

                # 单 peer：直接用
                if len(self._kv_bufs) == 1:
                    kv_data = self._kv_bufs[0]
                else:
                    # 多 peer：沿 head 维度 cat
                    kv_data = torch.cat(self._kv_bufs, dim=3)

                cache = self._scatter_kv_cache(kv_data, self._current_meta['seq_len'])
                request = Request(
                    request_id=self._current_meta['request_id'],
                    input_ids=self._current_meta['input_ids'],
                    max_new_tokens=self._current_meta['max_new_tokens'],
                    temperature=self._current_meta['temperature'],
                    top_p=self._current_meta['top_p'],
                    kv_cache=cache,
                )
                request.generated_ids = self._current_meta['generated_ids']
                self.completed_requests.append(request)

                self._meta_buf = None
                self._kv_bufs = None
                self._kv_handles = None
                self._current_meta = None
                self._state = RecvState.IDLE
                progress = True

            if not progress:
                break