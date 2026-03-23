import torch
import sampler

''' Engine 负责管理模型和采样器，提供统一接口供外部调用
- 第一阶段的计划是支持单卡、单模型、单batch，无KV cache，单步采样
'''
class Engine:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        # self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = device
    
    ''' 生成接口，输入 prompt 和采样参数，返回生成的字符串
    - prompt: 输入文本
    - max_new_tokens: 最多生成多少个 token
    - temperature: 采样温度，0 表示 greedy
    - top_p: top-p 截断，1.0 表示不截断
    return: 生成的完整字符串（含 Prompt）
    '''
    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_p: float = 1.0) -> str:
        # 返回生成的完整字符串（含Prompt）
        # 把模型切换到 eval 模式，关闭 dropout 等训练时机制
        self.model.eval()
        # output_ids: (1, seq_len)，初始为 prompt 的 token ids，后续不断 append 新生成的 token id
        output_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        num_new_tokens = 0
        while not (output_ids[0, -1].item() == self.tokenizer.eos_token_id or num_new_tokens >= max_new_tokens):
            logits = self.step(output_ids, temperature, top_p)
            next_token_id = sampler.sample(logits, temperature, top_p)
            output_ids = torch.cat([output_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            num_new_tokens += 1
        return self.tokenizer.decode(output_ids[0])

    ''' 模型单步前向，输入当前的 token ids，返回下一个 token 的 logits和已经生成的 token nums
    - input_ids: 当前的 token ids，shape (1, seq_len)
    - temperature: 采样温度，传递给 sampler
    - top_p: top-p 截断，传递给 sampler
    return: 下一个 token 的 logits，shape (vocab_size,)
    '''
    def step(self, input_ids: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs[0, -1, :]  # 取最后一个 token 的 logits，shape (vocab_size,)
        return logits

