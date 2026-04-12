import torch
import transformers
from cache import KVCache, NaiveKVCache, PagedKVCache, BlockManager
import sampler
from scheduler import RequestStatus, Scheduler, Request
from kv_transfer import KVTransferBase, SyncKVTransfer, NoOpKVTransfer

''' Engine 负责管理模型和采样器，提供统一接口供外部调用
'''
class Engine:
    def __init__(self, 
                 model, 
                 tokenizer, 
                 block_manager: BlockManager, 
                 cache_cls : type[PagedKVCache], 
                 cache_kwargs: dict|None = None, device='cuda',
                 use_cuda_graph=True,   # 是否启用 CUDA Graph，启用后要求 batch size 固定为 max_batch_size
                 max_batch_size=8,      # ← CUDA Graph 需要固定 batch size,
                 tp_size=1, 
                 rank=0,
                 role:str="pd",
                 manual_seed=42
                 ):
        self.model = model.to(device)
        # self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache_cls = cache_cls
        # KV cache 的配置参数，后续可以改成动态的
        self.kv_cache_kwargs = cache_kwargs if cache_kwargs is not None else dict(
            block_manager=block_manager,
            num_layers=model.cfg.num_hidden_layers,
            max_seq_len=4096,
            num_kv_heads=model.cfg.local_num_key_value_heads,
            head_dim=model.cfg.head_dim,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        self.block_manager = block_manager
        self.eos_token_id = tokenizer.eos_token_id

        ### tp 并行相关设置 ###
        self.tp_size = tp_size
        self.rank = rank
        if tp_size > 1:
            torch.manual_seed(manual_seed)
            torch.cuda.manual_seed(manual_seed)
        
        ### PD 分离相关设置 ###
        self.role = role
        if self.role == "p":
            self.transfer = SyncKVTransfer(local_rank=rank, peer_rank=1, device=device,
                                            block_manager=block_manager, model_cfg=model.cfg,cache_kwargs=self.kv_cache_kwargs,)
        elif self.role == "d":
            self.transfer = SyncKVTransfer(local_rank=rank, peer_rank=0, device=device,
                                            block_manager=block_manager, model_cfg=model.cfg,cache_kwargs=self.kv_cache_kwargs,)
        else:
            # self.transfer = None  # role="pd" 不需要传输层
            self.transfer = NoOpKVTransfer()  # 类型安全，poll() 和 try_recv 都是 no-op

        self.scheduler = Scheduler(kv_cache_cls=cache_cls, kv_cache_kwargs=self.kv_cache_kwargs)
        self.no_more_requests = False # 不会再有更多请求进入
        self._done_sent = False # 已经全部发送完

        self.use_cuda_graph = use_cuda_graph
        self.max_batch_size = max_batch_size
        
        # CUDA Graph 只在有 decode 职责时 build
        if use_cuda_graph and role in ("d", "pd"):
            self._build_cuda_graph()

        self.model.eval()
        
    def _build_cuda_graph(self):
        """预分配静态 buffer 并 capture graph，只做一次"""
        # 静态 buffer
        self.static_input_ids    = torch.zeros(self.max_batch_size, 1, dtype=torch.long, device=self.device)
        self.static_slot_mapping = torch.zeros(self.max_batch_size, dtype=torch.int32, device=self.device)
        self.static_position_ids = torch.zeros(self.max_batch_size, 1, dtype=torch.long, device=self.device)
        self.static_block_table  = torch.full(
            (self.max_batch_size, self.model.cfg.MAX_BLOCKS_PER_SEQ), -1,
            dtype=torch.int32, device=self.device
        )
        self.static_context_lens = torch.zeros(self.max_batch_size, dtype=torch.int32, device=self.device)

        # 预热（触发 Triton autotune）
        for _ in range(3):
            with torch.no_grad():
                _ = self.model.forward_decode(
                    self.static_input_ids,
                    kv_cache_k=self.block_manager.gpu_kv_cache[0],
                    kv_cache_v=self.block_manager.gpu_kv_cache[1],
                    position_ids=self.static_position_ids,
                    slot_mapping=self.static_slot_mapping,
                    block_table=self.static_block_table,
                    context_lens=self.static_context_lens,
                )
        torch.cuda.synchronize()
        
        # capture
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.static_output = self.model.forward_decode(
                self.static_input_ids,
                kv_cache_k=self.block_manager.gpu_kv_cache[0],
                kv_cache_v=self.block_manager.gpu_kv_cache[1],
                position_ids=self.static_position_ids,
                slot_mapping=self.static_slot_mapping,
                block_table=self.static_block_table,
                context_lens=self.static_context_lens,
            )
        
        print(f"CUDA Graph captured (batch_size={self.max_batch_size})")
    
    def _decode_step_graph(self, decoding, kv_caches) -> torch.Tensor:
        """CUDA Graph replay 路径"""
        B = len(decoding)

        # =================================================================
        # 1. CPU侧
        # =================================================================
        input_ids_cpu = []
        slot_mapping_cpu = []
        position_ids_cpu = []
        context_lens_cpu = []

        for request, c in zip(decoding, kv_caches):
            c.prepare_decode_step()
            input_ids_cpu.append([request.generated_ids[-1]])
            slot_mapping_cpu.append(c.get_decode_slot())
            position_ids_cpu.append([c._seq_len])
            context_lens_cpu.append(c._seq_len + 1)

        # 补齐 Padding (幽灵请求)
        if B < self.max_batch_size:
            pad_len = self.max_batch_size - B
            input_ids_cpu.extend([[0]] * pad_len)
            slot_mapping_cpu.extend([-1] * pad_len)
            position_ids_cpu.extend([[0]] * pad_len)
            context_lens_cpu.extend([0] * pad_len)

        # =================================================================
        # 2. DMA 异步拷贝
        # =================================================================
        self.static_input_ids.copy_(torch.tensor(input_ids_cpu, dtype=torch.long), non_blocking=True)
        self.static_slot_mapping.copy_(torch.tensor(slot_mapping_cpu, dtype=torch.int32), non_blocking=True)
        self.static_position_ids.copy_(torch.tensor(position_ids_cpu, dtype=torch.long), non_blocking=True)
        self.static_context_lens.copy_(torch.tensor(context_lens_cpu, dtype=torch.int32), non_blocking=True)

        # =================================================================
        # 3. Block Table
        # =================================================================
        self.static_block_table.fill_(-1)

        for i, c in enumerate(kv_caches):
            bt = c.get_block_table()
            self.static_block_table[i, :bt.shape[0]].copy_(bt, non_blocking=True)

        # =================================================================
        # 4. CUDA Graph重放
        # =================================================================
        self.cuda_graph.replay()

        for c in kv_caches:
            c._seq_len += 1

        return self.static_output[:B, -1, :]  # (B, vocab_size)
    
    def _decode_step_eager(self, decoding, kv_caches, B) -> torch.Tensor:
        """普通 eager 路径，用于 B > max_batch_size 或 use_cuda_graph=False"""
        for c in kv_caches:
            c.prepare_decode_step()

        slots = [c.get_decode_slot() for c in kv_caches]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device=self.device)

        input_ids = torch.tensor(
            [[r.generated_ids[-1]] for r in decoding],
            dtype=torch.long, device=self.device
        )
        position_ids = torch.tensor(
            [[c.seq_len] for c in kv_caches],
            dtype=torch.long, device=self.device
        )
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
                kv_cache_k=self.block_manager.gpu_kv_cache[0],
                kv_cache_v=self.block_manager.gpu_kv_cache[1],
                position_ids=position_ids,
                slot_mapping=slot_mapping,
                is_prefill=False,
                block_table=block_table,
                context_lens=context_lens,
            )

        for c in kv_caches:
            c._seq_len += 1

        return logits[:, -1, :]  # (B, vocab_size)

    ###########################################
    # Batch 版本的接口，包括submit(),step(),is_done()
    ###########################################

    def submit(self, 
            prompt: str, 
            max_new_tokens: int, 
            temperature: float, 
            top_p: float) -> int:
            """提交生成请求，返回 request_id"""
            # 如果已经完成，返回-1
            if self.no_more_requests:
                return -1
            # 转成 list 存储
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device).tolist()[0]
            request = self.scheduler.create_request(input_ids, max_new_tokens, temperature, top_p, self.kv_cache_cls, self.kv_cache_kwargs)
            self.scheduler.add_request(request)
            return request.request_id
    
    def mark_finished(self):
        """告诉 Engine 不会再有新请求提交"""
        self.no_more_requests = True
    
    def step(self) -> list[tuple[int, str]]:
        # ── D 侧：先检查有没有新到的请求，在 schedule 之前 ──
        if self.role == "d" and not self.transfer.recv_done:
            new_request = self.transfer.try_recv_request()
            if new_request:
                # print(f'[Rank 1] new request received id {new_request.request_id}')
                self.scheduler.add_request(new_request, RequestStatus.DECODING)

        prefilling, decoding = self.scheduler.schedule()
        completed_requests = []

        # ── prefill：逐个请求，动态形状，不走 graph ──
        for request in prefilling:
            input_ids = torch.tensor(request.input_ids).unsqueeze(0).to(self.device)
            kv_cache = request.kv_cache
            kv_cache._allocate_for_prefill(len(request.input_ids))
            slot_mapping = kv_cache.get_prefill_slot_mapping(len(request.input_ids))

            start_pos = kv_cache.seq_len
            position_ids = torch.arange(
                start_pos, start_pos + len(request.input_ids),
                dtype=torch.long, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(
                    input_ids,
                    kv_cache_k=self.block_manager.gpu_kv_cache[0],
                    kv_cache_v=self.block_manager.gpu_kv_cache[1],
                    position_ids=position_ids,
                    slot_mapping=slot_mapping,
                    is_prefill=True,
                )

            kv_cache._seq_len += len(request.input_ids)

            # debug：打印每个请求分配到的物理 block
            # print(f"[DEBUG] req {request.request_id}: "
            #     f"seq_len={kv_cache.seq_len}, "
            #     f"blocks={kv_cache.get_block_table().tolist()}, "
            #     f"allocated={kv_cache.allocated_cache_block_num}")
            next_token_id = sampler.sample(logits[:, -1, :], request.temperature, request.top_p)
            request.generated_ids.append(int(next_token_id.item()))
            # print(next_token_id.item())

            if self.role == "p":
                self.transfer.send_request(request)
                # print(f'[Rank 0] sent request id {request.request_id}')
                request.kv_cache.reset()
                request.has_finished_notification = True
            elif next_token_id.item() == self.eos_token_id or request.is_max_len_finished():
                request.has_finished_notification = True
                completed_requests.append((
                    request.request_id,
                    self.tokenizer.decode(request.input_ids + request.generated_ids)
                ))

        # ── P 侧：所有 prefill 做完且用户标记 no_more_requests，发终止信号 ──
        if (self.role == "p"
            and self.no_more_requests
            and len(self.scheduler.waiting) == 0
            and len(self.scheduler.prefilling) == 0
            and not self._done_sent):
            self.transfer.send_done()
            self._done_sent = True
            # print("[Rank 0] sent done flag set")

        # ── decode：batch，走 CUDA Graph 或 eager ──
        if decoding and self.role in ("d", "pd"):
            B = len(decoding)
            # print(B)
            kv_caches = [r.kv_cache for r in decoding]

            # # debug：打印 decode 时每个请求的 block table
            # for i, (r, c) in enumerate(zip(decoding, kv_caches)):
            #     print(f"[DEBUG decode] req {r.request_id}: "
            #         f"seq_len={c.seq_len}, "
            #         f"blocks={c.get_block_table().tolist()}")
                
            if self.use_cuda_graph and B <= self.max_batch_size:
                logits_batch = self._decode_step_graph(decoding, kv_caches)
            else:
                logits_batch = self._decode_step_eager(decoding, kv_caches, B)

            for i, request in enumerate(decoding):
                next_token_id = sampler.sample(logits_batch[i], request.temperature, request.top_p)
                request.generated_ids.append(int(next_token_id.item()))
                if next_token_id.item() == self.eos_token_id or request.is_max_len_finished():
                    request.has_finished_notification = True
                    completed_requests.append((
                        request.request_id,
                        self.tokenizer.decode(request.input_ids + request.generated_ids)
                    ))

        return completed_requests
    
    def is_done(self) -> bool:
        if self.role == "p":
            return self._done_sent
        elif self.role == "d":
            return (self.transfer.recv_done
                    and len(self.scheduler.decoding) == 0)
        else:  # "pd"
            return (self.no_more_requests
                    and len(self.scheduler.waiting) == 0
                    and len(self.scheduler.prefilling) == 0
                    and len(self.scheduler.decoding) == 0)