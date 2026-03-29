import torch
import transformers
from cache import KVCache, NaiveKVCache, PagedKVCache, BlockManager
import sampler
from scheduler import RequestStatus, Scheduler, Request

''' Engine 负责管理模型和采样器，提供统一接口供外部调用
- 第一阶段的计划是支持单卡、单模型、单batch，无KV cache，单步采样
'''
class Engine:
    def __init__(self, model, tokenizer, block_manager: BlockManager, cache_cls : type[PagedKVCache], cache_kwargs: dict|None = None, device='cuda'):
        self.model = model.to(device)
        # self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache_cls = cache_cls
        # KV cache 的配置参数，后续可以改成动态的
        self.kv_cache_kwargs = cache_kwargs if cache_kwargs is not None else dict(
            block_manager=block_manager,   # ← 加这个
            num_layers=model.cfg.num_hidden_layers,
            max_seq_len=4096,
            num_kv_heads=model.cfg.num_key_value_heads,
            head_dim=model.cfg.head_dim,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        self.block_manager = block_manager
        self.eos_token_id = tokenizer.eos_token_id

        self.scheduler = Scheduler(kv_cache_cls=cache_cls, kv_cache_kwargs=self.kv_cache_kwargs)

        self.model.eval()
    
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
        # prefill：逐个请求，slot_mapping 长度各不同
        for request in prefilling:
            input_ids = torch.tensor(request.input_ids).unsqueeze(0).to(self.device)
            kv_cache = request.kv_cache
            kv_cache._allocate_for_prefill(len(request.input_ids))
            slot_mapping = kv_cache.get_prefill_slot_mapping(len(request.input_ids))

            # 新增：构造 position_ids
            start_pos = kv_cache.seq_len  # prefill 前是 0
            position_ids = torch.arange(
                start_pos, start_pos + len(request.input_ids),
                dtype=torch.long, device=self.device
            ).unsqueeze(0)  # (1, seq_len)

            with torch.no_grad():
                logits = self.model(
                    input_ids,
                    kv_cache_k=self.block_manager.gpu_kv_cache[0],  # 传 tensor
                    kv_cache_v=self.block_manager.gpu_kv_cache[1],
                    position_ids=position_ids,
                    slot_mapping=slot_mapping,
                    is_prefill=True,
                )

            # 新增：forward 之后手动更新 seq_len（原来在 prefill_update 里做）
            kv_cache._seq_len += len(request.input_ids) + 1
            
            next_token_id = sampler.sample(logits[:, -1, :], request.temperature, request.top_p)
            request.generated_ids.append(int(next_token_id.item()))
            request.kv_cache = kv_cache

            if next_token_id.item() == self.eos_token_id or request.is_max_len_finished():
                request.has_finished_notification = True
                completed_requests.append((
                    request.request_id,
                    self.tokenizer.decode(request.input_ids + request.generated_ids)
                ))

        # decode：batch，slot_mapping shape = (B,)
        if decoding:
            B = len(decoding)
            kv_caches = [r.kv_cache for r in decoding]

            for request in decoding:
                request.kv_cache.prepare_decode_step()

            slots = [c.get_decode_slot() for c in kv_caches]
            slot_mapping = torch.tensor(slots, dtype=torch.int32, device=self.device)

            input_ids = torch.tensor(
                [[r.generated_ids[-1]] for r in decoding],
                dtype=torch.long, device=self.device
            )

            # 新增：position_ids，每个请求当前的位置
            position_ids = torch.tensor(
                [[c.seq_len] for c in kv_caches],
                dtype=torch.long, device=self.device
            )  # (B, 1)

            block_table = torch.full(
                (B, self.model.cfg.MAX_BLOCKS_PER_SEQ), -1,
                dtype=torch.int32, device=self.device
            )
            for i, c in enumerate(kv_caches):
                bt = c.get_block_table()
                block_table[i, :len(bt)] = bt

            context_lens = torch.tensor(
                [c.seq_len + 1 for c in kv_caches],
                dtype=torch.int32, device=self.device
            )

            with torch.no_grad():
                logits = self.model(
                    input_ids,
                    kv_cache_k=self.block_manager.gpu_kv_cache[0],  # 传 tensor
                    kv_cache_v=self.block_manager.gpu_kv_cache[1],
                    position_ids=position_ids,
                    slot_mapping=slot_mapping,
                    is_prefill=False,
                    block_table=block_table,
                    context_lens=context_lens,
                )

            # 新增：forward 之后手动更新 seq_len
            for c in kv_caches:
                c._seq_len += 1
            
            for i, request in enumerate(decoding):
                next_token_id = sampler.sample(logits[i, -1, :], request.temperature, request.top_p)
                request.generated_ids.append(int(next_token_id.item()))
                if next_token_id.item() == self.eos_token_id or request.is_max_len_finished():
                    request.has_finished_notification = True
                    completed_requests.append((
                        request.request_id,
                        self.tokenizer.decode(request.input_ids + request.generated_ids)
                    ))

        return completed_requests