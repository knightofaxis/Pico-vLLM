from enum import Enum
from typing import List
from cache import PagedKVCache
from radix_tree import KVCacheRadixTreeNode


class RequestStatus(Enum):
    WAITING  = "waiting"   # 在 waiting 队列，还没做 prefill
    PREFILL  = "prefill"   # 本步正在做 prefill
    DECODING = "decoding"  # 已经 prefill 完，正在 decode
    FINISHED = "finished"  # 已完成，未释放资源
    CLOSED = "closed"  # 已完成，已在radix tree层面释放资源

class Request:
    """一个生成请求的所有状态。

    字段含义：
    - request_id: 请求 ID，唯一标识一个请求。
    - input_ids: 整个 prompt 的 token ids。
    - generated_ids: 已生成的 token ids，初始为空。
    - max_new_tokens / temperature / top_p: 生成与采样参数。
    - kv_cache: 每个请求独享一个 KV cache 实例。
    - request_status: 调度状态机中的位置（waiting / prefill / decoding / finished / closed）。
    - has_finished_notification: 由 engine 通知 scheduler 该请求已停止；状态迁移由 scheduler 负责。
    - matched_blocks / matched_len / last_node: prefix cache 命中信息，prefill 时用。
    """
    request_id: int
    input_ids: List[int]  # (1, init_seq_len)，包含整个 prompt
    generated_ids: List[int]  # (1, )，包含已经生成的 token ids，初始为 空
    max_new_tokens: int
    temperature: float
    top_p: float
    kv_cache: PagedKVCache  # 每个请求独享一个 KV cache 实例
    request_status: RequestStatus
    has_eos_token: bool  # 是否已经生成 eos_token，scheduler 不直接接触 tokenizer 和 eos_token_id，这个由 engine 在 decode_step 后更新
    # radix_ext_blocks: int
    last_node : KVCacheRadixTreeNode
    
    def __init__(self, request_id: int, input_ids: List[int], max_new_tokens: int, temperature: float, top_p: float, kv_cache: PagedKVCache, generated_ids: List[int] | None = None):
        self.request_id = request_id
        self.input_ids = input_ids
        self.generated_ids = generated_ids if generated_ids is not None else []
        self.request_status = RequestStatus.WAITING
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kv_cache = kv_cache
        self.has_finished_notification = False # engine改变这个状态，scheduler根据这个状态改变 request_status和移出队列
        # self.radix_ext_blocks: int = 0

        ###### Prefix Caching 相关字段 ######
        self.matched_blocks: list[int] = []   # prefix match 命中的 block
        self.matched_len: int = 0             # 命中的 token 数

    def is_max_len_finished(self) -> bool:
        return len(self.generated_ids) >= self.max_new_tokens

    @property
    def prompt_len(self) -> int:
        return len(self.input_ids)
    
    @property  
    def total_len(self) -> int:
        return len(self.input_ids) + len(self.generated_ids)

class Scheduler:
    """请求调度器，用 waiting / prefilling / decoding / finished 四个队列做状态机。

    `schedule()` 每次被 engine 调用一次，把已收到 finished notification 的请求移出
    decoding，把完成 prefill 的整批搬到 decoding，并从 waiting 尽可能补齐到
    `max_num_seqs` 的并发度（先到先服务）。engine 负责实际的 prefill / decode 计算
    和对 `has_finished_notification` 的写入。
    """
    _next_request_id: int  # 用于生成唯一的 request_id
    max_num_seqs: int  # 同时处理的最大请求数，超过这个数的新请求会排队等待
    kv_cache_cls : type[PagedKVCache]  # KVCache 的类，用于创建请求的 KV cache 实例
    kv_cache_kwargs = {}  # 创建 KV cache 实例的参数字典
    no_more_requests = False  # 不会再有新请求加入

    waiting: List[Request]    # 还没开始处理的请求
    prefilling: List[Request]   # 正在 prefill 的请求
    decoding: List[Request]     # 正在 decode 的请求
    finished: List[Request]   # 已完成的请求
    def __init__(self, kv_cache_cls: type[PagedKVCache], kv_cache_kwargs={}, _next_request_id: int = 0, max_num_seqs: int = 4):
        self._next_request_id = _next_request_id
        self.max_num_seqs = max_num_seqs
        self.kv_cache_cls = kv_cache_cls
        self.kv_cache_kwargs = kv_cache_kwargs
        self.waiting = []
        self.prefilling = []
        self.decoding = []
        self.finished = []

    def is_all_done(self) -> bool:
        """所有请求都处理完了，且不会再有新的"""
        return (self.no_more_requests 
                and len(self.waiting) == 0 
                and len(self.prefilling) == 0 
                and len(self.decoding) == 0)
    
    ''' 插入新请求
    - input_ids: 输入的 token ids，shape (1, seq_len)，包含整个 prompt
    - max_new_tokens: 最多生成多少个 token
    - temperature: 采样温度，传递给 sampler
    - top_p: top-p 截断，传递给 sampler
    - cache_cls: KVCache 的类，用于创建请求的 KV cache 实例
    - cache_kwargs: 创建 KV cache 实例的参数字典
    return: request_id，唯一标识一个请求'''
    def insert_request(self, 
                       input_ids: List[int], 
                       max_new_tokens: int, 
                       temperature: float, 
                       top_p: float, ) -> int:
        request = self.create_request(input_ids, max_new_tokens, temperature, top_p, self.kv_cache_cls, self.kv_cache_kwargs)
        self.add_request(request)
        return request.request_id

    '''' 创建新的请求对象
    - input_ids: 输入的 token ids，shape (1, seq_len)，包含整个 prompt
    - max_new_tokens: 最多生成多少个 token
    - temperature: 采样温度，传递给 sampler
    - top_p: top-p 截断，传递给 sampler
    - cache_cls: KVCache 的类，用于创建请求的 KV cache 实例
    - cache_kwargs: 创建 KV cache 实例的参数字典
    return: request_id，唯一标识一个请求'''
    def create_request(self, 
            input_ids: List[int], 
            max_new_tokens: int, 
            temperature: float, 
            top_p: float, 
            cache_cls, 
            cache_kwargs) -> Request:
        request_id = self._next_request_id
        self._next_request_id += 1
        kv_cache = cache_cls(**cache_kwargs)
        return Request(request_id, input_ids, max_new_tokens, temperature, top_p, kv_cache)

    '''' 添加新请求到等待队列
    - request: 新的请求对象
    '''
    def add_request(self, request: Request, type:RequestStatus=RequestStatus.WAITING):
        if type == RequestStatus.WAITING:
            self.waiting.append(request)
        elif type == RequestStatus.DECODING:
            self.decoding.append(request)

    ''' 调度器主循环
    return: 2个列表，分别是当前处于 prefilling、decoding 状态的请求列表
    - prefilling: 正在 prefill 的请求
    - decoding: 正在 decode 的请求
    '''
    def schedule(self) -> tuple[List[Request], List[Request]]:

        # 1. 对 decoding 队列中的请求进行 decode，完成后移动到 finished 队列
        # has_finished_notification 由 engine 在 decode_step 后更新，scheduler 不直接
        # 接触 tokenizer 和 eos_token_id；用列表推导一次 O(n) 重建代替 list.remove 的 O(n^2)。
        still_decoding: List[Request] = []
        for request in self.decoding:
            if request.has_finished_notification:
                request.request_status = RequestStatus.FINISHED
                self.finished.append(request)
            else:
                still_decoding.append(request)
        self.decoding = still_decoding

        # 2. prefilling 整批搬到 decoding —— 一次 extend + clear 比逐个 remove 高效
        for request in self.prefilling:
            request.request_status = RequestStatus.DECODING
        self.decoding.extend(self.prefilling)
        self.prefilling.clear()

        # 3. 从 waiting 队列中取出请求，放入 prefilling 队列，直到达到 max_batch_size
        # 不进行kv_cache的创建和初始化
        while self.waiting and self.num_in_progress < self.max_num_seqs:
            # 先到先服务策略
            request = self.waiting.pop(0)
            request.request_status = RequestStatus.PREFILL
            self.prefilling.append(request)

        return self.prefilling, self.decoding

    '''' 清空 finished 队列，释放资源'''
    def clear_finished(self):
        self.finished.clear()

    def get_running_requests(self) -> List[Request]:
        return self.prefilling + self.decoding
    
    @property
    def num_waiting(self) -> int:
        return len(self.waiting)
    @property
    def num_prefilling(self) -> int:
        return len(self.prefilling)
    @property
    def num_decoding(self) -> int:
        return len(self.decoding)
    @property
    def num_in_progress(self) -> int:
        return len(self.prefilling) + len(self.decoding)
    @property
    def num_finished(self) -> int:
        return len(self.finished)
