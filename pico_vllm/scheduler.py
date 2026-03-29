from enum import Enum
import torch
from typing import List, Optional
from dataclasses import dataclass
from cache import KVCache, NaiveKVCache, PagedKVCache


class RequestStatus(Enum):
    WAITING  = "waiting"   # 在 waiting 队列，还没做 prefill
    PREFILL  = "prefill"   # 本步正在做 prefill
    DECODING = "decoding"  # 已经 prefill 完，正在 decode
    FINISHED = "finished"  # 已完成

''' 请求对象，包含请求的所有信息和状态
- request_id: 请求 ID，唯一标识一个请求
- input_ids: 输入的 token ids，shape (1, init_seq_len)，包含整个 prompt
- generated_ids: 已经生成的 token ids，shape (1, )，初始为 空
- max_new_tokens: 最多生成多少个 token
- temperature: 采样温度，传递给 sampler
- top_p: top-p 截断，传递给 sampler
- kv_cache: 每个请求独享一个 KV cache 实例，存储生成过程中的 KV 状态
'''
class Request:
    request_id: int
    input_ids: List[int]  # (1, init_seq_len)，包含整个 prompt
    generated_ids: List[int]  # (1, )，包含已经生成的 token ids，初始为 空
    max_new_tokens: int
    temperature: float
    top_p: float
    kv_cache: PagedKVCache  # 每个请求独享一个 KV cache 实例
    request_status: RequestStatus
    has_eos_token: bool  # 是否已经生成 eos_token，scheduler 不直接接触 tokenizer 和 eos_token_id，这个由 engine 在 decode_step 后更新

    def __init__(self, request_id: int, input_ids: List[int], max_new_tokens: int, temperature: float, top_p: float, kv_cache: PagedKVCache):
        self.request_id = request_id
        self.input_ids = input_ids
        self.generated_ids = []  # 初始为 空
        self.request_status = RequestStatus.WAITING
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kv_cache = kv_cache
        self.has_finished_notification = False # engine改变这个状态，scheduler根据这个状态改变 request_status和移出队列

    def is_max_len_finished(self) -> bool:
        return len(self.generated_ids) >= self.max_new_tokens

    @property
    def prompt_len(self) -> int:
        return len(self.input_ids)
    
    @property  
    def total_len(self) -> int:
        return len(self.input_ids) + len(self.generated_ids)

class Scheduler:
    _next_request_id: int  # 用于生成唯一的 request_id
    max_num_seqs: int  # 同时处理的最大请求数，超过这个数的新请求会排队等待
    kv_cache_cls : type[PagedKVCache]  # KVCache 的类，用于创建请求的 KV cache 实例
    kv_cache_kwargs = {}  # 创建 KV cache 实例的参数字典

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
    def add_request(self, request: Request):
        self.waiting.append(request)

    ''' 调度器主循环
    return: 2个列表，分别是当前处于 prefilling、decoding 状态的请求列表
    - prefilling: 正在 prefill 的请求
    - decoding: 正在 decode 的请求
    '''
    def schedule(self) -> tuple[List[Request], List[Request]]:
        
        # 1. 对 decoding 队列中的请求进行 decode，完成后移动到 finished 队列
        for request in self.decoding[:]:
            # is_max_len_finished 由 scheduler判断
            # has_finished_notification 由 engine 在 decode_step 后更新，解耦两者的逻辑，scheduler 不直接接触 tokenizer 和 eos_token_id
            if request.has_finished_notification == True:
                request.request_status = RequestStatus.FINISHED
                self.decoding.remove(request)
                self.finished.append(request)

        # 2. 对 prefilling 队列中的请求进行 prefill，完成后移动到 decoding 队列
        for request in self.prefilling[:]:
            # prefill 完成后：
            request.request_status = RequestStatus.DECODING
            self.prefilling.remove(request)
            self.decoding.append(request)
        
        # 3. 从 waiting 队列中取出请求，放入 prefilling 队列，直到达到 max_batch_size
        # 不进行kv_cache的创建和初始化
        while self.waiting and self.num_in_progress < self.max_num_seqs:
            # 先到先服务策略
            request = self.waiting.pop(0)
            request.request_status = RequestStatus.PREFILL
            self.prefilling.append(request)
            # kv cache必须定长，否则报错
            # request.kv_cache = self.kv_cache_cls(**self.kv_cache_kwargs)

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


# if __name__ == "__main__":
#     from scheduler import Scheduler, Request, RequestStatus
#     from cache import NaiveKVCache
    
#     # KV cache 配置，所有请求共用
#     kv_cache_kwargs = dict(
#         num_layers=4,       # 用小值，不需要真实模型
#         max_seq_len=256,
#         num_kv_heads=2,
#         head_dim=64,
#         device='cpu',
#         dtype=torch.float32,
#     )
    
#     scheduler = Scheduler(
#         kv_cache_cls=NaiveKVCache,
#         kv_cache_kwargs=kv_cache_kwargs,
#         max_num_seqs=3,
#     )
    
#     # 插入 5 个请求，超过 max_num_seqs
#     for i in range(5):
#         scheduler.insert_request(
#             input_ids=list(range(i + 3)),  # 不同长度的 prompt
#             max_new_tokens=10,
#             temperature=0.0,
#             top_p=1.0,
#         )
    
#     print(f"初始状态: waiting={scheduler.num_waiting}, "
#           f"prefilling={scheduler.num_prefilling}, "
#           f"decoding={scheduler.num_decoding}")
#     assert scheduler.num_waiting == 5
    
#     # Step 1：前 3 个进入 prefilling
#     prefilling, decoding = scheduler.schedule()
#     print(f"\nStep 1 schedule 后:")
#     print(f"  waiting={scheduler.num_waiting}, "
#           f"prefilling={scheduler.num_prefilling}, "
#           f"decoding={scheduler.num_decoding}")
#     print(f"  prefilling ids: {[r.request_id for r in prefilling]}")
#     print(f"  decoding ids:   {[r.request_id for r in decoding]}")
#     assert scheduler.num_waiting == 2
#     assert scheduler.num_prefilling == 3
#     assert scheduler.num_decoding == 0
    
#     # 验证 KV cache 已创建
#     for r in prefilling:
#         assert r.kv_cache is not None, f"request {r.request_id} 没有 kv_cache"
#         assert r.request_status == RequestStatus.PREFILL
#     print("  KV cache 创建正确 ✓")
    
#     # 模拟 engine 完成 prefill：更新 generated_ids
#     for r in prefilling:
#         r.generated_ids.append(100)  # 假装生成了一个 token
    
#     # Step 2：prefilling → decoding，waiting 里补充一个进来
#     prefilling, decoding = scheduler.schedule()
#     print(f"\nStep 2 schedule 后:")
#     print(f"  waiting={scheduler.num_waiting}, "
#           f"prefilling={scheduler.num_prefilling}, "
#           f"decoding={scheduler.num_decoding}")
#     print(f"  prefilling ids: {[r.request_id for r in prefilling]}")
#     print(f"  decoding ids:   {[r.request_id for r in decoding]}")
#     assert scheduler.num_waiting == 2   # 还剩 1 个等待
#     assert scheduler.num_prefilling == 0  # 新进来 1 个
#     assert scheduler.num_decoding == 3    # 上一步的 3 个
    
#     # Step 3：模拟其中一个请求遇到 EOS
#     decoding[0].has_eos_token = True
    
#     # 模拟 engine 完成 decode
#     for r in decoding:
#         r.generated_ids.append(101)
    
#     prefilling, decoding = scheduler.schedule()
#     print(f"\nStep 3 schedule 后（一个请求 EOS）:")
#     print(f"  waiting={scheduler.num_waiting}, "
#           f"prefilling={scheduler.num_prefilling}, "
#           f"decoding={scheduler.num_decoding}, "
#           f"finished={scheduler.num_finished}")
#     assert scheduler.num_finished == 1
#     assert scheduler.num_decoding == 2   # EOS 的那个移出了
#     print(f"  finished ids: {[r.request_id for r in scheduler.finished]}")
    
#     # Step 4：模拟一个请求达到 max_new_tokens
#     for r in decoding:
#         r.generated_ids = list(range(10))  # 填满 max_new_tokens=10
    
#     prefilling, decoding = scheduler.schedule()
#     print(f"\nStep 4 schedule 后（max_new_tokens 耗尽）:")
#     print(f"  waiting={scheduler.num_waiting}, "
#           f"prefilling={scheduler.num_prefilling}, "
#           f"decoding={scheduler.num_decoding}, "
#           f"finished={scheduler.num_finished}")
    
#     print("\n所有断言通过 ✓")