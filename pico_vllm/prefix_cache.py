from blockmanager import BlockManager
from radix_tree import KVCacheRadixTree, KVCacheRadixTreeNode

class PrefixCache:
    """
    协调 RadixTree 和 BlockManager，对外提供 match / insert / release 接口。
    """
    def __init__(self, radix_tree:KVCacheRadixTree, block_manager:BlockManager):
        self.radix_tree = radix_tree
        self.block_manager = block_manager
        self.stats = {
            'match_calls': 0,
            'hit_tokens': 0,
            'miss_tokens': 0,
        }

    def match(self, tokens: list[int]) -> tuple[list[int], int, 'KVCacheRadixTreeNode']:
        """
        请求开始，沿 tokens 路径 inc_ref。
        查找 prefix 匹配。命中的 block 自动 inc_ref（防止被驱逐）。
        返回 (matched_block_ids, matched_len)
        """
        blocks, length, last_node = self.radix_tree.match_prefix(tokens)
        if blocks:
            self.block_manager.inc_ref(blocks)       # block 层保护
            self.radix_tree.inc_lock_ref(last_node)  # radix 层保护
        return blocks, length, last_node             # 返回 node 指针

    def insert(self, tokens: list[int], block_ids: list[int]):
        """
        请求 prefill 完成后，把新产生的 KV block 加入 cache。
        tokens 和 block_ids 必须对齐到 block_size。
        """
        newly_held_by_tree = self.radix_tree.insert(tokens, block_ids)
        if newly_held_by_tree:
            self.block_manager.inc_ref(newly_held_by_tree)
        return newly_held_by_tree

    def try_evict(self, num_blocks_needed: int) -> list[int]:
        """
        BlockManager 显存不足时调用，从 radix tree 驱逐可驱逐的叶子节点。
        返回被释放的物理 block id 列表。
        """    
        evicted_blocks = self.radix_tree.evict(num_blocks_needed)
        if evicted_blocks:
            # RadixTree 不再持有这些 block，给 BlockManager 的 ref_count -1
            # 如果 -1 后到 0，BlockManager 自动把 block 加回 free pool
            self.block_manager.dec_ref(evicted_blocks)
        return evicted_blocks
    
    def peek(self, tokens: list[int]) -> tuple[list[int], int]:
        """只查询，不改 ref。用于测试或只读检查。"""
        return self.radix_tree.match(tokens)

    def hit_rate(self):
        total = self.stats['hit_tokens'] + self.stats['miss_tokens']
        return self.stats['hit_tokens'] / total if total > 0 else 0