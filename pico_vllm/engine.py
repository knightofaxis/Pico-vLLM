import torch
import transformers
from cache import KVCache, NaiveKVCache
import sampler
from scheduler import RequestStatus, Scheduler, Request

''' Engine 负责管理模型和采样器，提供统一接口供外部调用
- 第一阶段的计划是支持单卡、单模型、单batch，无KV cache，单步采样
'''
class Engine:
    def __init__(self, model, tokenizer, cache_cls : type[KVCache] = NaiveKVCache, cache_kwargs: dict | None = None, device='cuda'):
        self.model = model.to(device)
        # self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache_cls = cache_cls
        # KV cache 的配置参数，后续可以改成动态的
        self.kv_cache_kwargs = cache_kwargs if cache_kwargs is not None else dict(
            num_layers=model.cfg.num_hidden_layers,
            max_seq_len=4096,  # 先按硬编码实现，模型的 max_position_embeddings 会OOM
            num_kv_heads=model.cfg.num_key_value_heads,
            head_dim=model.cfg.hidden_size // model.cfg.num_attention_heads,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        self.eos_token_id = tokenizer.eos_token_id

        self.scheduler = Scheduler(kv_cache_cls=cache_cls, kv_cache_kwargs=self.kv_cache_kwargs)

        self.model.eval()
    

    ###########################################
    # 非Batch 版本的接口，包括prefill(), decode_step(), generate()
    # generate() 是单请求的接口，内部调用 prefill() 和 decode_step() 实现生成逻辑
    ###########################################

    # batch情况下的prefill也可以用它
    ''' 一次性前向，输入完整的 prompt，返回最后一个 token 的 logits
    - input_ids: 当前的 token ids，shape (1, seq_len)，包含整个 prompt
    return: 最后一个 token 的 logits，shape (vocab_size,)
    '''
    def prefill(self, input_ids: torch.Tensor, kvcache: KVCache) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids, kv_cache=kvcache)
            logits = outputs[:, -1, :]  # 取最后一个 token 的 logits，shape (vocab_size,)
        return logits

    # 只有非batch情况下的decode会调用它
    ''' decode 接口，输入当前的 token ids 和 KV cache，返回下一个 token 的 logits
    - input_ids: 当前的 token ids，shape (1, seq_len)，只包含当前 step 的 token
    - kv_cache: 当前的 KV cache，包含历史 token 的 KV
    return: 下一个 token 的 logits，shape (vocab_size,)
    '''
    def decode_step(self, input_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids, kv_cache=kv_cache)
            logits = outputs[:, -1, :]  # 取最后一个 token 的 logits，shape (vocab_size,)
        return logits
    
    #这个只能处理单请求，另外实现多请求的版本，名字叫step()
    ''' 生成接口，输入 prompt 和采样参数，返回生成的字符串
    - prompt: 输入文本
    - max_new_tokens: 最多生成多少个 token
    - temperature: 采样温度，0 表示 greedy
    - top_p: top-p 截断，1.0 表示不截断
    return: 生成的完整字符串（含 Prompt）'''
    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 100, 
                 temperature: float = 1.0, 
                 top_p: float = 1.0) -> str:
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        # kv_cache = self.kv_cache_cls(
        #     num_layers=self.model.cfg.num_hidden_layers,
        #     max_seq_len=4096,  # 先按硬编码实现，模型的 max_position_embeddings 会OOM
        #     num_kv_heads=self.model.cfg.num_key_value_heads,
        #     head_dim=self.model.cfg.hidden_size // self.model.cfg.num_attention_heads,
        #     device=self.device,
        #     dtype=next(self.model.parameters()).dtype)
        kv_cache = self.kv_cache_cls(
            **self.kv_cache_kwargs
        )
        
        # prefill
        logits = self.prefill(input_ids, kv_cache)
        
        # 自回归的decode loop
        generated_ids = []
        num_new_tokens = 0
        while not (generated_ids and generated_ids[-1] == self.tokenizer.eos_token_id) and num_new_tokens < max_new_tokens:
            next_token_id = sampler.sample(logits, temperature, top_p)
            generated_ids.append(next_token_id.item())
            num_new_tokens += 1
            
            # decode step
            logits = self.decode_step(next_token_id.unsqueeze(0).unsqueeze(0), kv_cache)
        
        full_ids = input_ids[0].tolist() + generated_ids
        return self.tokenizer.decode(full_ids)
    

    ###########################################
    # Batch 版本的接口，包括submit(),step()
    ###########################################

    def submit(self, 
            prompt: str, 
            max_new_tokens: int, 
            temperature: float, 
            top_p: float) -> int:
        """提交生成请求，返回 request_id"""
        # 转成 list 存储
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device).tolist()[0]
        request = self.scheduler.create_request(input_ids, max_new_tokens, temperature, top_p, self.kv_cache_cls, self.kv_cache_kwargs)
        self.scheduler.add_request(request)
        return request.request_id
    
    def step(self) -> list[tuple[int, str]]:
        """调度器主循环，处理所有请求的 prefill 和 decode
        return: list of (request_id, generated_text)，包含刚刚完成的请求的id和生成结果"""
        prefilling, decoding = self.scheduler.schedule()
        # 这里的 prefilling 和 decoding 是 Request 对象列表，包含了每个请求的状态和参数
        # 需要对它们进行分组，构造成适合模型输入的 batch，然后调用模型进行前向计算，最后将结果写回对应的 Request 对象中
        # 在实现paged KV cache之前，先实现一个简单的版本，直接对每个请求调用 prefill() 和 decode_step()，不进行真正的 batch 处理
        # prefill 按照单个请求处理，decode 按多个请求一起处理
        
        completed_requests = []
        # 1. 处理 prefill 队列，逐个请求进行前向计算
        for request in prefilling:
            input_ids = torch.tensor(request.input_ids).unsqueeze(0).to(self.device)  # (1, seq_len)
            kv_cache = self.kv_cache_cls(**self.kv_cache_kwargs)
            logits = self.prefill(input_ids, kv_cache)
            next_token_id = sampler.sample(logits, request.temperature, request.top_p)
            request.generated_ids.append(int(next_token_id.item()))
            request.kv_cache = kv_cache  # 将 prefill 时的 KV cache 存储到请求对象中，供后续 decode 使用
            
            # 判断是否结束
            if next_token_id.item() == self.eos_token_id or request.is_max_len_finished():
                request.has_finished_notification = True  # 通知 scheduler 这个请求已经完成，scheduler 会在下一轮调度时将其移出 decoding 队列
                completed_requests.append((request.request_id, self.tokenizer.decode(request.input_ids + request.generated_ids)))
        
        # 2. 处理 decode 队列，构造 batch 输入进行前向计算
        for request in decoding:
            input_ids = torch.tensor([request.generated_ids[-1]]).unsqueeze(0).to(self.device)  # (1, 1)，只输入当前 step 的 token
            kv_cache = request.kv_cache  # 从请求对象中获取 KV cache
            # 理论上这里不可能为none，但类型检查会报警告，所以加个判断
            if kv_cache is None:
                kv_cache = self.kv_cache_cls(**self.kv_cache_kwargs)  # 如果没有 KV cache，创建一个新的（理论上不应该发生，因为 prefill 时会创建）
            logits = self.decode_step(input_ids, kv_cache)
            next_token_id = sampler.sample(logits, request.temperature, request.top_p)
            request.generated_ids.append(int(next_token_id.item()))
            # 更新 KV cache，假设模型的 decode_step 已经在内部更新了 KV cache
            # 判断是否结束
            if next_token_id.item() == self.eos_token_id or request.is_max_len_finished():
                request.has_finished_notification = True  # 通知 scheduler 这个请求已经完成，scheduler 会在下一轮调度时将其移出 decoding 队列
                completed_requests.append((request.request_id, self.tokenizer.decode(request.input_ids + request.generated_ids)))
        
        return completed_requests