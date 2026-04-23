"""
Prefix Cache Benchmark — Long Prompt Edition
=============================================
4096+ token 的共享前缀 + ~200 token 的可变 user query。
在这个 workload 下，GPU prefill 是绝对主导，
prefix cache 的 TTFT 加速应该非常显著。

用法：
  python benchmark_prefix_cache_long.py
  torchrun --nproc_per_node=2 benchmark_prefix_cache_long.py
"""

import torch
import time
import os
from transformers import AutoTokenizer


# ────────────────────────────────────────────
# Part 1: 构造超长 system prompt（~4096 tokens）
# ────────────────────────────────────────────

LONG_SYSTEM_PROMPT = """You are an expert AI assistant specialized in software engineering, system design, and distributed systems. You have deep knowledge of the following areas and should provide comprehensive, well-structured answers.

## Programming Languages and Paradigms
You are proficient in Python, C++, Rust, Go, Java, JavaScript, TypeScript, and many other programming languages. You understand object-oriented programming, functional programming, procedural programming, and actor-based concurrency models. You can write clean, efficient, and well-documented code in all of these languages. You follow best practices including proper error handling, type safety, comprehensive testing, and clear documentation.

## System Design and Architecture
You understand microservices architecture, monolithic architecture, event-driven architecture, and serverless computing. You can design systems that handle millions of requests per second, with proper load balancing, caching, and database sharding strategies. You know how to design for high availability, fault tolerance, and disaster recovery. You understand CAP theorem, PACELC theorem, and their practical implications.

## Distributed Systems
You have deep knowledge of distributed consensus algorithms including Paxos, Raft, and Byzantine fault tolerance. You understand distributed transactions, two-phase commit, saga pattern, and eventual consistency models. You can explain vector clocks, Lamport timestamps, and causal ordering. You know how to design distributed hash tables, consistent hashing, and gossip protocols.

## Database Systems
You are expert in both relational databases (PostgreSQL, MySQL, SQLite) and NoSQL databases (MongoDB, Cassandra, Redis, DynamoDB). You understand B-tree and LSM-tree storage engines, write-ahead logging, MVCC concurrency control, and query optimization. You can design efficient database schemas, write complex SQL queries, and optimize database performance through proper indexing, partitioning, and replication strategies.

## Cloud Computing and DevOps
You are proficient with AWS, GCP, and Azure cloud platforms. You understand containerization with Docker, orchestration with Kubernetes, and infrastructure as code with Terraform and Pulumi. You know CI/CD best practices, monitoring and observability with Prometheus and Grafana, and incident response procedures.

## Machine Learning Systems
You understand the full ML lifecycle from data collection to model deployment. You know how to design feature stores, training pipelines, model serving infrastructure, and A/B testing frameworks. You are familiar with distributed training techniques including data parallelism, model parallelism, pipeline parallelism, and expert parallelism. You understand quantization, pruning, knowledge distillation, and other model compression techniques.

## Operating Systems and Hardware
You have deep knowledge of Linux internals, memory management, process scheduling, file systems, and network stacks. You understand CPU architecture, cache hierarchies, NUMA topology, and GPU computing. You know how to profile and optimize system performance using tools like perf, strace, flamegraphs, and hardware performance counters.

## Networking
You understand TCP/IP, UDP, HTTP/2, HTTP/3, gRPC, WebSocket, and other networking protocols. You know about DNS, CDN, load balancers, reverse proxies, and API gateways. You can design network architectures that are secure, scalable, and performant. You understand network security including TLS, certificate management, OAuth, and JWT.

## Security
You follow security best practices including the principle of least privilege, defense in depth, and zero trust architecture. You understand common vulnerabilities (OWASP Top 10), encryption algorithms, key management, and secure coding practices. You can perform threat modeling and security code reviews.

## Data Engineering
You are proficient with Apache Spark, Apache Kafka, Apache Flink, and Apache Airflow. You understand batch processing, stream processing, lambda architecture, and kappa architecture. You can design data pipelines that are reliable, scalable, and maintainable. You know about data quality, data governance, and data lineage tracking.

## Software Engineering Best Practices
You advocate for clean code, SOLID principles, design patterns, and refactoring techniques. You understand agile methodologies, code review best practices, and technical debt management. You can write comprehensive unit tests, integration tests, and end-to-end tests. You know how to use version control systems effectively and manage complex branching strategies.

## Communication and Problem Solving
When answering questions, you should:
1. First understand the context and constraints of the problem
2. Break down complex problems into smaller, manageable components
3. Consider multiple approaches and their tradeoffs
4. Provide concrete examples and code snippets when helpful
5. Highlight potential pitfalls and edge cases
6. Suggest follow-up resources for deeper learning

You should format your responses clearly using markdown, with appropriate headers, bullet points, and code blocks. You should be concise but thorough, and always prioritize accuracy over speed.

## Additional Context for This Session

This is a production environment where reliability and correctness are paramount. The user is a senior engineer who values precise technical details and practical advice. They may ask about system design, code review, debugging, performance optimization, or architecture decisions. Always consider production-readiness, monitoring, and operational concerns in your answers.

The following reference material is provided for context in this conversation session:

### Reference: Common Design Patterns

The Singleton pattern ensures a class has only one instance and provides a global point of access to it. The Factory Method pattern defines an interface for creating objects but lets subclasses decide which class to instantiate. The Observer pattern defines a one-to-many dependency between objects so that when one object changes state, all dependents are notified. The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. The Decorator pattern attaches additional responsibilities to an object dynamically. The Adapter pattern converts the interface of a class into another interface clients expect. The Template Method pattern defines the skeleton of an algorithm, deferring some steps to subclasses. The Command pattern encapsulates a request as an object, parameterizing clients with different requests. The Iterator pattern provides a way to access elements of an aggregate sequentially. The Proxy pattern provides a surrogate or placeholder for another object to control access to it.

### Reference: Distributed Systems Concepts

In distributed computing, the FLP impossibility result states that in an asynchronous system where at least one process can crash, there is no deterministic algorithm that solves the consensus problem. This fundamental limitation drives the design of practical consensus algorithms that make additional assumptions about timing. The Paxos algorithm, proposed by Leslie Lamport, achieves consensus in an asynchronous system by using a leader-based approach with three phases: prepare, accept, and learn. The Raft algorithm, designed as a more understandable alternative to Paxos, decomposes consensus into leader election, log replication, and safety. Both algorithms guarantee safety but may not guarantee liveness under certain network conditions.

Consistency models in distributed systems range from strong consistency (linearizability) to eventual consistency. Linearizability requires that all operations appear to execute atomically and in real-time order. Sequential consistency relaxes this by requiring only that operations appear to execute in some sequential order consistent with program order. Causal consistency requires that causally related operations are seen in the same order by all processes, while allowing concurrent operations to be seen in different orders. Eventual consistency is the weakest model, guaranteeing only that if no new updates are made, all replicas will eventually converge to the same state.

### Reference: Performance Optimization Techniques

Performance optimization should always be guided by measurement. Use profiling tools to identify bottlenecks before optimizing. Common optimization techniques include: algorithmic improvements (reducing time complexity), data structure optimization (choosing the right data structure for the access pattern), memory optimization (reducing allocations, improving cache locality), I/O optimization (batching, prefetching, asynchronous operations), and concurrency optimization (parallelism, lock-free data structures, work stealing). Always benchmark before and after optimization to verify the improvement and check for regressions.

Cache optimization is particularly important in modern systems. L1 cache access takes about 1 nanosecond, L2 about 4 nanoseconds, L3 about 12 nanoseconds, and main memory about 100 nanoseconds. This 100x difference between L1 and main memory makes cache-friendly data structures and access patterns critical for performance. Techniques include structure-of-arrays vs array-of-structures, cache-oblivious algorithms, prefetching, and avoiding false sharing in multi-threaded code.

### Reference: Kubernetes Architecture

Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications. The control plane consists of the API server (which exposes the Kubernetes API), etcd (a distributed key-value store for cluster state), the scheduler (which assigns pods to nodes), and the controller manager (which runs control loops). Worker nodes run the kubelet (which manages containers on the node), kube-proxy (which handles network routing), and the container runtime. Key abstractions include Pods (the smallest deployable units), Services (stable network endpoints), Deployments (declarative updates for Pods), StatefulSets (for stateful applications), and DaemonSets (for running a pod on every node).

### Reference: Database Internals

B-tree indexes organize data in a balanced tree structure where each node contains multiple keys and child pointers. The branching factor is typically chosen to match the disk page size, minimizing I/O operations. B+ trees, a variant used in most databases, store all data in leaf nodes and maintain a linked list between leaves for efficient range scans. LSM-trees (Log-Structured Merge-trees) optimize write performance by buffering writes in memory and periodically flushing sorted runs to disk. Compaction merges multiple sorted runs to maintain read performance. Write-ahead logging (WAL) ensures durability by recording all modifications to a log before applying them to the main data structure. This allows recovery after crashes by replaying the log.

MVCC (Multi-Version Concurrency Control) allows multiple transactions to access the same data concurrently without blocking. Each transaction sees a consistent snapshot of the database. PostgreSQL implements MVCC by storing multiple versions of each row, with visibility determined by transaction IDs. This approach avoids read locks entirely, allowing readers and writers to proceed without blocking each other.

Now, given all the above context, please answer the following question thoughtfully and comprehensively:

"""

# 不同的 user query，每个约 30-50 token
USER_QUERIES_LONG = [
    "Explain how you would design a distributed rate limiter that works across multiple data centers with different latency characteristics. Consider both token bucket and sliding window approaches.",
    "I have a PostgreSQL database that is experiencing slow queries on a table with 500 million rows. The table has several indexes but queries still take over 10 seconds. Walk me through your debugging and optimization process step by step.",
    "Compare and contrast the Raft and Paxos consensus algorithms in terms of their practical implementation complexity, failure handling, and performance characteristics in production systems with varying network conditions.",
    "Design a real-time feature store that can serve features with sub-millisecond latency for an online ML inference system handling 100k requests per second. Address consistency, freshness, and fault tolerance.",
    "Explain the tradeoffs between synchronous and asynchronous replication in distributed databases, and describe scenarios where each approach is more appropriate. Include discussion of chain replication.",
    "How would you implement a distributed tracing system from scratch? Cover the data model, sampling strategies, storage backend selection, and query interface design.",
    "Walk me through the process of diagnosing and fixing a memory leak in a long-running Python service. Include discussion of tools, techniques, and common pitfalls specific to CPython's memory management.",
    "Design a multi-tenant SaaS platform that provides strong isolation guarantees while maximizing resource utilization. Discuss the database-per-tenant vs shared-database approaches and their implications.",
]


def verify_prompt_length(tokenizer):
    """验证 prompt 长度符合预期"""
    system_tokens = len(tokenizer.encode(LONG_SYSTEM_PROMPT))
    print(f"System prompt length: {system_tokens} tokens")

    for i, query in enumerate(USER_QUERIES_LONG):
        full = LONG_SYSTEM_PROMPT + query
        total = len(tokenizer.encode(full))
        query_tokens = len(tokenizer.encode(query))
        print(f"  Query {i}: {query_tokens} tokens, total: {total} tokens")

    return system_tokens


# ────────────────────────────────────────────
# Part 2: TTFT 测量（复用之前的逻辑）
# ────────────────────────────────────────────

def measure_ttft(engine, prompt, max_new_tokens=10):
    """提交单个请求，测量 TTFT"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    req_id = engine.submit(prompt, max_new_tokens=max_new_tokens, temperature=0, top_p=1.0)

    start.record()
    completed = engine.step()
    end.record()
    torch.cuda.synchronize()
    ttft_ms = start.elapsed_time(end)

    output = ""
    if completed:
        for rid, text in completed:
            if rid == req_id:
                output = text

    while not output:
        completed = engine.step()
        for rid, text in completed:
            if rid == req_id:
                output = text

    return ttft_ms


def create_engine(enable_prefix_cache, tp_size, rank, device):
    from model import Qwen25_15B, ModelConfig
    from weights import load_weights
    from cache import PagedKVCache
    from blockmanager import BlockManager
    from engine import Engine

    cfg = ModelConfig(tp_size=tp_size)
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights", tp_size=tp_size, tp_rank=rank)
    model = model.to(torch.bfloat16).to(device)

    tokenizer = AutoTokenizer.from_pretrained("./weights")
    BLOCK_SIZE = 16

    bm = BlockManager(
        num_gpu_blocks=1000,      # 长 prompt 需要更多 block
        num_cpu_blocks=0,
        block_size=BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim,
        dtype=torch.bfloat16,
    )

    engine = Engine(
        model=model, tokenizer=tokenizer, block_manager=bm,
        cache_cls=PagedKVCache, device=device,
        use_cuda_graph=True,
        local_tp_size=tp_size, rank=rank,
        enable_prefix_cache=enable_prefix_cache,
    )

    return engine, tokenizer


def run_one_config(enable_prefix_cache, tp_size, rank, device, label):
    engine, tokenizer = create_engine(enable_prefix_cache, tp_size, rank, device)

    system_tokens = len(tokenizer.encode(LONG_SYSTEM_PROMPT))
    num_queries = len(USER_QUERIES_LONG)

    # warmup
    _ = measure_ttft(engine, "Hello world", max_new_tokens=3)

    results = []

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  {label}  (system prompt ≈ {system_tokens} tokens)")
        print(f"{'='*70}")
        print(f"  {'#':>3}  {'total_tok':>9}  {'shared':>6}  {'new':>5}  {'TTFT(ms)':>10}")
        print(f"  {'-'*3}  {'-'*9}  {'-'*6}  {'-'*5}  {'-'*10}")

    for i, query in enumerate(USER_QUERIES_LONG):
        full_prompt = LONG_SYSTEM_PROMPT + query
        total_tokens = len(tokenizer.encode(full_prompt))
        query_tokens = len(tokenizer.encode(query))

        ttft = measure_ttft(engine, full_prompt, max_new_tokens=10)
        results.append({
            'idx': i,
            'total': total_tokens,
            'shared': system_tokens,
            'new': query_tokens,
            'ttft': ttft,
            'cold': (i == 0),
        })

        if rank == 0:
            marker = "COLD" if i == 0 else "warm"
            print(f"  {i:>3}  {total_tokens:>9}  {system_tokens:>6}  {query_tokens:>5}  "
                  f"{ttft:>10.2f}  {marker}")

    # stats
    if enable_prefix_cache and engine.prefix_cache is not None:
        hr = engine.prefix_cache.hit_rate()
        if rank == 0:
            print(f"\n  Hit rate: {hr*100:.1f}%")

    del engine
    torch.cuda.empty_cache()
    return results


def main():
    import torch.distributed as dist

    tp_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if tp_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained("./weights")
    if rank == 0:
        print("=== Prompt Length Verification ===")
        verify_prompt_length(tokenizer)
    del tokenizer

    # 先跑 OFF（承担 warmup 开销）
    results_off = run_one_config(False, tp_size, rank, device, "Prefix Cache: OFF")
    results_on = run_one_config(True, tp_size, rank, device, "Prefix Cache: ON")

    # Summary
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  Summary: Long Prompt Prefix Cache Benchmark")
        print(f"{'='*70}")
        print(f"  {'#':>3}  {'total':>6}  {'OFF(ms)':>10}  {'ON(ms)':>10}  {'speedup':>8}  {'note'}")
        print(f"  {'-'*3}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")

        cold_off, cold_on = 0, 0
        warm_off, warm_on = 0, 0
        n_warm = 0

        for roff, ron in zip(results_off, results_on):
            speedup = roff['ttft'] / ron['ttft'] if ron['ttft'] > 0 else float('inf')
            note = "COLD" if roff['cold'] else "warm"
            print(f"  {roff['idx']:>3}  {roff['total']:>6}  {roff['ttft']:>10.2f}  "
                  f"{ron['ttft']:>10.2f}  {speedup:>7.2f}x  {note}")

            if roff['cold']:
                cold_off += roff['ttft']
                cold_on += ron['ttft']
            else:
                warm_off += roff['ttft']
                warm_on += ron['ttft']
                n_warm += 1

        print(f"\n  Cold (1st request): OFF={cold_off:.2f}ms  ON={cold_on:.2f}ms  "
              f"speedup={cold_off/cold_on:.2f}x")
        if n_warm > 0:
            avg_off = warm_off / n_warm
            avg_on = warm_on / n_warm
            print(f"  Warm (avg of {n_warm}):  OFF={avg_off:.2f}ms  ON={avg_on:.2f}ms  "
                  f"speedup={avg_off/avg_on:.2f}x")

    if tp_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()