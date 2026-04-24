#radix_tree.py
import torch
import torch.distributed as dist
import os, sys
import time
from typing import List, Optional, Dict
from queue import PriorityQueue

class KVCacheRadixTreeNode:
    """
    RadixTree的节点
    
    children: 指向子节点的指针集合
    value:这个节点对应的逻辑块的id
    ref_count: 引用技术，只有=0的时候，才可以释放物理块
    
    """
    def __init__(self, key_tokens: List[int], parent: 'KVCacheRadixTreeNode|None' = None):
        # 1. 路由信息
        # Key: Token ID 序列
        # Value: 指向子节点的指针
        self.key_tokens: List[int] = key_tokens
        
        # 子节点字典。
        # Key: 子节点 key_tokens 的第一个 Token ID（用于 O(1) 快速路由）。
        # Value: 子节点对象。
        self.children: Dict[int, 'KVCacheRadixTreeNode'] = {}
        # 指向父节点的指针（用于回溯、合并和引用计数向上传递）
        self.parent: 'KVCacheRadixTreeNode|None' = parent

        # 2. 缓存负载 (Payload)
        # 该片段对应的逻辑块（或物理块）的 ID 列表。
        # 引擎拿到这个列表，就知道去显存的哪里取 KV 张量。
        self.cached_blocks: List[int] = []
        
        # 3. 缓存管理元数据 (核心变化)
        # 引用计数：记录当前有多少个正在执行的请求（Sequence）正在使用这个节点（这段前缀）。
        # 当 lock_ref > 0 时，该节点及其关联的逻辑块绝对不能被释放。
        # self.ref_count: int = 0 
        self.lock_ref: int = 0 

        # 最后访问时间：用于 LRU (最近最少使用) 淘汰策略。
        # 只有当 ref_count == 0 时，这个时间戳才有意义。
        self.last_access_time: float = time.time()

    def is_leaf(self) -> bool:
        """判断是否为叶子节点（用于判断是否可以被完全剪枝）"""
        return len(self.children) == 0

    def is_evictable(self) -> bool:
        """判断当前节点是否可以被释放（引用计数为0）"""
        return self.lock_ref == 0

    def update_access_time(self) -> None:
        """更新最后访问时间（每次有请求 hit 到这里，或者 lock_ref 归零时调用）"""
        self.last_access_time = time.time()

class KVCacheRadixTree:
    """
    管理 token prefix → block_ids 的映射。
    
    约定：
    - tokens 用 List[int] 表示（token id 序列）
    - block_ids 用 List[int] 表示（物理 block id 列表）
    - tokens 和 block_ids 的对应关系：
      每 block_size 个 token 对应一个 block_id
      tokens[:block_size] → block_ids[0]
      tokens[block_size:2*block_size] → block_ids[1]
      以此类推
    """

    def __init__(self, block_size: int):
        """
        block_size: 每个 block 容纳多少 token，用于 token↔block 的对齐
        """
        self.block_size = block_size

        # 根节点，key_tokens 为空列表
        self.root: KVCacheRadixTreeNode = KVCacheRadixTreeNode(key_tokens=[])

        # 记录当前树中正在管理的所有节点，按 last_access_time 排序的优先队列/有序字典。
        # 用于在 O(1) 或 O(log N) 时间内找到最久未使用的节点进行 Evict。
        self.evictable_queue = PriorityQueue()

    def insert(self, tokens: list[int], block_ids: list[int]) -> list[int]:
        """
        插入 token 序列和对应 block。返回 RadixTree 新持有的 block id 列表
        （调用方据此调用 BlockManager.inc_ref 保持引用计数一致）。

        三种情况对应的返回值：
        完全匹配已有路径：返回 []（RadixTree 没新增持有）
        无匹配创建新节点：返回完整 block_ids
        部分匹配分裂：返回只属于新后缀分支的 block_ids
        """
        if not tokens:
            return []
            
        newly_held = []
        curr_node = self.root
        i = 0  # 当前在 tokens 中的深度索引
        
        while i < len(tokens):
            if tokens[i] not in curr_node.children:
                # 找不到匹配的前缀，将剩余的 tokens 全部作为一个新叶子节点插入
                new_node = KVCacheRadixTreeNode(tokens[i:], curr_node)
                # 计算属于这个新节点的 blocks (根据 block_size 边界)
                first_block_idx = i // self.block_size
                new_node.cached_blocks = block_ids[first_block_idx:]
                new_node.lock_ref = 0  # 新插入的分支，目前没有prompt在tree里登记，需要加入可驱逐队列
                new_node.update_access_time()
                if new_node.is_leaf():
                    self.evictable_queue.put((new_node.last_access_time, id(new_node), new_node))
                curr_node.children[tokens[i]] = new_node

                newly_held.extend(block_ids[first_block_idx:]) # ← 只记录新后缀
                break
                
            child = curr_node.children[tokens[i]]
            edge_tokens = child.key_tokens
            
            # 寻找当前边上的最长公共前缀
            match_len = 0
            while (match_len < len(edge_tokens) and 
                   i + match_len < len(tokens) and 
                   edge_tokens[match_len] == tokens[i + match_len]):
                match_len += 1
                
            if match_len == len(edge_tokens):
                # 完全匹配了这条边，继续向下遍历
                i += match_len
                curr_node = child
                # child.ref_count += 1  # 共享路径增加引用
                child.update_access_time()
                # 幂等操作，如果完全匹配已有路径，循环最终会平稳结束，不做任何事
            else:
                # 部分匹配，触发节点分裂 (Split)
                split_node = KVCacheRadixTreeNode(edge_tokens[:match_len], curr_node)
                
                # 计算分配给父节点（split_node）的 physical blocks 数量
                # 公式：匹配结束处的 block 索引 - 匹配开始处的 block 索引
                num_split_blocks = (i + match_len) // self.block_size - i // self.block_size
                split_node.cached_blocks = child.cached_blocks[:num_split_blocks]
                split_node.lock_ref = child.lock_ref  # 继承原节点的引用计数，因为前缀是共享的
                
                # 更新原来的 child，使其成为 split_node 的子节点
                child.key_tokens = edge_tokens[match_len:]
                child.cached_blocks = child.cached_blocks[num_split_blocks:]
                child.parent = split_node
                split_node.children[child.key_tokens[0]] = child
                
                # 将 split_node 挂载到当前的父节点上
                curr_node.children[tokens[i]] = split_node
                
                # 将新 sequence 中未匹配完的后缀挂载到 split_node 上
                rem_tokens = tokens[i + match_len:]
                if rem_tokens:
                    new_node = KVCacheRadixTreeNode(rem_tokens, split_node)
                    first_block_idx = (i + match_len) // self.block_size
                    new_node.cached_blocks = block_ids[first_block_idx:]
                    new_node.lock_ref = 0
                    new_node.update_access_time()
                    if new_node.is_leaf():
                        self.evictable_queue.put((new_node.last_access_time, id(new_node), new_node))
                    split_node.children[new_node.key_tokens[0]] = new_node
                    
                    newly_held.extend(block_ids[first_block_idx:]) # 记录新后缀
                break

        return newly_held
        

    def match(self, tokens: list[int]) -> tuple[list[int], int]:
        """
        给定 token 序列，返回最长前缀匹配的 block_ids 和命中的 token 数。
        
        返回：
          block_ids: 命中的物理 block id 列表（可能为空）
          matched_len: 命中了多少个 token（block_size 的整数倍）
          
        例：tokens 有 100 个，匹配到前 64 个（4 个 block）
            返回 ([b0, b1, b2, b3], 64)
            调用方只需要 prefill tokens[64:]
        """
        matched_blocks = []
        matched_len = 0
        curr_node = self.root
        i = 0
        
        while i < len(tokens):
            if i < len(tokens) and tokens[i] not in curr_node.children:
                break
                
            child = curr_node.children[tokens[i]]
            edge_tokens = child.key_tokens
            
            match_len = 0
            while (match_len < len(edge_tokens) and 
                   i + match_len < len(tokens) and 
                   edge_tokens[match_len] == tokens[i + match_len]):
                match_len += 1
                
            if match_len == len(edge_tokens):
                # 完整匹配该边缘
                i += match_len
                matched_len += match_len
                matched_blocks.extend(child.cached_blocks)
                curr_node = child
            else:
                # 只有部分匹配该边缘，向下取整到 block_size 边界
                # 哪怕匹配了 3 个 token，如果 block_size=2，也只能安全使用前面 2 个 token 对应的 1 个 Block
                valid_blocks_count = match_len // self.block_size
                matched_len += valid_blocks_count * self.block_size
                matched_blocks.extend(child.cached_blocks[:valid_blocks_count])
                break
                
        # 返回前统一对齐到 block_size
        aligned_len = (matched_len // self.block_size) * self.block_size
        aligned_blocks_count = aligned_len // self.block_size
        return matched_blocks[:aligned_blocks_count], aligned_len

    def match_prefix(self, tokens: list[int]) -> tuple[list[int], int, 'KVCacheRadixTreeNode']:
        """
        找最长前缀匹配。

        返回:
        matched_blocks: 命中的逻辑 block id 列表
        matched_len:    命中的 token 数（= len(matched_blocks) * block_size，天然对齐）
        last_node:      最后一个完整匹配的节点指针（用于 inc/dec_lock_ref）
                        如果没有完整匹配任何边，返回 root

        设计要点:
        - last_node 只在 edge 被完整匹配时更新（保证指针稳定性）
        - 部分匹配的 edge 也能贡献 block（按 block_size 向下取整）
        - 但部分匹配不更新 last_node（避免锁定可能被分裂的节点）
        """
        matched_blocks = []
        last_node = self.root
        curr_node = self.root
        i = 0

        while i < len(tokens):
            if tokens[i] not in curr_node.children:
                break

            child = curr_node.children[tokens[i]]
            edge = child.key_tokens

            # 计算这条边上的匹配长度
            match_len = 0
            while (match_len < len(edge) and
                i + match_len < len(tokens) and
                edge[match_len] == tokens[i + match_len]):
                match_len += 1

            if match_len == len(edge):
                # ── 完整匹配 ──
                matched_blocks.extend(child.cached_blocks)
                last_node = child          # ← 更新指针
                i += match_len
                curr_node = child
            else:
                # ── 部分匹配：取 block 对齐的部分 ──
                valid_blocks = match_len // self.block_size
                matched_blocks.extend(child.cached_blocks[:valid_blocks])
                # last_node 不更新（留在 parent）
                break

        # matched_len 直接从 block 数量算，天然对齐
        matched_len = len(matched_blocks) * self.block_size
        return matched_blocks, matched_len, last_node


    def inc_lock_ref(self, node: 'KVCacheRadixTreeNode'):
        """
        从 node 向上走到 root（不含 root），每个节点 lock_ref += 1。
        
        语义：一个新的活跃请求正在使用从 root 到 node 的路径上的所有节点。
        """
        curr = node
        while curr is not None and curr is not self.root:
            curr.lock_ref += 1
            curr = curr.parent


    def dec_lock_ref(self, node: 'KVCacheRadixTreeNode'):
        """
        从 node 向上走到 root（不含 root），每个节点 lock_ref -= 1。
        lock_ref 降到 0 的叶子节点加入可驱逐队列。
        
        语义：一个活跃请求不再使用从 root 到 node 的路径。
        """
        curr = node
        while curr is not None and curr is not self.root:
            curr.lock_ref -= 1
            assert curr.lock_ref >= 0, \
                f"lock_ref underflow on node id={id(curr)}, key_len={len(curr.key_tokens)}"

            if curr.lock_ref == 0:
                curr.update_access_time()
                if curr.is_leaf():
                    self.evictable_queue.put(
                        (curr.last_access_time, id(curr), curr)
                    )
            curr = curr.parent


    def evict(self, num_blocks_needed: int) -> List[int]:
        """
        Block Manager 发现没有空闲物理块时调用。
        
        输入: 需要腾出多少个物理块的空间。
        输出: 可以被回收的逻辑块 ID 列表（Block Manager 以此为依据进行覆写）。
        
        内部行为:
            1. 从 `self.evictable_queue` 中弹出 `last_access_time` 最早的节点 (LRU)。
            2. 收集该节点身上的 `cached_blocks` 准备返回。
            3. 将该节点从树中删除。
            4. 检查该节点的父节点：如果父节点只剩下一个子节点，触发【节点合并 (Merge)】，压缩树的高度。
            5. 循环执行，直到收集到的 blocks 数量达到 `num_blocks_needed`。
        """
        evicted_blocks = []
        while len(evicted_blocks) < num_blocks_needed and not self.evictable_queue.empty():
            q_time, _, node = self.evictable_queue.get()
            
            # 延迟删除的有效性检查 (Lazy Deletion Check)
            if node.lock_ref > 0:
                continue # 可能在队列期间又被使用了
            if q_time != node.last_access_time:
                continue # 是个旧状态记录，最新的记录还在后面
            if not node.is_leaf():
                continue # 必须是从叶子节点往上修剪
            if node.parent is None or node.parent.children.get(node.key_tokens[0]) != node:
                continue # 节点已经被修剪过了
                
            # 执行驱逐并收集空闲块
            evicted_blocks.extend(node.cached_blocks)
            self._remove_node(node)
            
        return evicted_blocks
    
    def delete(self, block_id: int) -> None:
        """
        删除包含指定 block_id 的节点（驱逐时调用）。
        删除后如果父节点只剩一个孩子，合并父子节点（保持压缩性质）。
        """
        # DFS 查找包含该 block_id 的节点并删除
        node = self._find_node_with_block(self.root, block_id)
        if node and node is not self.root:
            self._remove_node(node)


    # --- 内部辅助方法 ---
    
    def _find_node_with_block(self, node: KVCacheRadixTreeNode, block_id: int) -> KVCacheRadixTreeNode|None:
        if block_id in node.cached_blocks:
            return node
        for child in node.children.values():
            res = self._find_node_with_block(child, block_id)
            if res: 
                return res
        return None

    def _remove_node(self, node: KVCacheRadixTreeNode) -> None:
        """
        核心合并逻辑：从树中断开该节点，并处理父节点的级联反应（合并或加入淘汰队列）。
        """
        if node.parent is None:
            return
            
        parent = node.parent
        first_token = node.key_tokens[0]
        
        if first_token in parent.children and parent.children[first_token] == node:
            # 1. 物理移除该节点
            del parent.children[first_token]
            node.parent = None
            
            # 2. 检查父节点的状态改变
            if parent is not self.root:
                if len(parent.children) == 0:
                    # 父节点变成了新的叶子节点，如果它没人用，级联放入淘汰队列
                    if parent.lock_ref == 0:
                        self.evictable_queue.put((parent.last_access_time, id(parent), parent))
                        
                elif len(parent.children) == 1:
                    # 父节点只剩唯一一个孩子了，触发【节点合并 (Merge)】以维持基数树的压缩特性
                    only_child = list(parent.children.values())[0]
                    
                    # 将唯一子节点的内容吸收到父节点中
                    parent.key_tokens.extend(only_child.key_tokens)
                    parent.cached_blocks.extend(only_child.cached_blocks)
                    parent.children = only_child.children
                    
                    # 更新孙子节点的父指针
                    for child in parent.children.values():
                        child.parent = parent

                    # merge 后 parent 可能变成 ref=0 的叶子，需要入队
                    if parent.lock_ref == 0 and parent.is_leaf():
                        parent.update_access_time()
                        self.evictable_queue.put(
                            (parent.last_access_time, id(parent), parent)
                        )