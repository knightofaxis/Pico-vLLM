# test_cuda_graph_basics.py
import torch
import time

device = 'cuda'

# ============================================================
# 实验一：普通执行 vs CUDA Graph 的速度差异
# ============================================================
print("=" * 50)
print("实验一：矩阵乘法，普通 vs Graph")
print("=" * 50)

N = 1536
A = torch.randn(N, N, device=device, dtype=torch.bfloat16)
B = torch.randn(N, N, device=device, dtype=torch.bfloat16)
out = torch.empty(N, N, device=device, dtype=torch.bfloat16)

# 预热
for _ in range(5):
    torch.mm(A, B, out=out)
torch.cuda.synchronize()

# 普通执行
RUNS = 1000
t0 = time.perf_counter()
for _ in range(RUNS):
    torch.mm(A, B, out=out)
torch.cuda.synchronize()
normal_ms = (time.perf_counter() - t0) / RUNS * 1000
print(f"普通执行：{normal_ms:.4f} ms/次")

# CUDA Graph capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    torch.mm(A, B, out=out)

# CUDA Graph replay
t0 = time.perf_counter()
for _ in range(RUNS):
    g.replay()
torch.cuda.synchronize()
graph_ms = (time.perf_counter() - t0) / RUNS * 1000
print(f"CUDA Graph：{graph_ms:.4f} ms/次")
print(f"加速比：{normal_ms / graph_ms:.2f}x")

# ============================================================
# 实验二：CUDA Graph 的关键约束——输入必须是同一块内存
# ============================================================
print("\n" + "=" * 50)
print("实验二：静态 buffer 的使用方式")
print("=" * 50)

# capture 时用的 tensor，之后必须原地修改，不能换新 tensor
static_input  = torch.zeros(4, device=device)
static_output = torch.empty(4, device=device)

g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    static_output.copy_(static_input * 2 + 1)

# 用法：修改 static_input 的值，然后 replay
static_input.copy_(torch.tensor([1., 2., 3., 4.]))
g2.replay()
print(f"input=[1,2,3,4] → output={static_output.tolist()}")
# 期望：[3, 5, 7, 9]

static_input.copy_(torch.tensor([10., 20., 30., 40.]))
g2.replay()
print(f"input=[10,20,30,40] → output={static_output.tolist()}")
# 期望：[21, 41, 61, 81]

# ============================================================
# 实验三：如果换了新 tensor 会怎样（错误示范）
# ============================================================
print("\n" + "=" * 50)
print("实验三：错误示范——不能换新 tensor")
print("=" * 50)

static_x = torch.tensor([1., 2., 3., 4.], device=device)
static_y = torch.empty(4, device=device)

g3 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g3):
    static_y.copy_(static_x + 10)

g3.replay()
print(f"正常 replay: {static_y.tolist()}")  # [11, 12, 13, 14]

# 错误：创建新 tensor 赋值给变量名，graph 不感知
new_x = torch.tensor([100., 200., 300., 400.], device=device)
# static_x = new_x  ← 不能这样！graph 里记录的是原来的 data_ptr
# 正确做法：
static_x.copy_(new_x)  # ← 原地修改值，保持 data_ptr 不变
g3.replay()
print(f"copy_ 后 replay: {static_y.tolist()}")  # [110, 210, 310, 410]

print("\n核心规则：")
print("  capture 后只能用 .copy_() 修改 tensor 的值")
print("  不能用 = 替换 tensor 本身（data_ptr 会变）")

# ============================================================
# 实验四：多 kernel 串联，模拟 model forward
# ============================================================
print("=" * 50)
print("实验四：多 kernel 串联，模拟 model forward")
print("=" * 50)

# 模拟 851 个小 kernel（28层 × ~30个算子）
N = 256  # 故意用小矩阵，让 kernel 执行时间短，launch overhead 占比高
tensors = [torch.randn(N, N, device=device, dtype=torch.bfloat16) for _ in range(10)]
out = torch.empty(N, N, device=device, dtype=torch.bfloat16)

def run_many_kernels():
    x = tensors[0]
    for i in range(1, len(tensors)):
        x = torch.mm(x, tensors[i])
        x = x + tensors[0]          # elementwise add
        x = torch.nn.functional.silu(x)  # activation
    return x

# 预热
for _ in range(5):
    run_many_kernels()
torch.cuda.synchronize()

# 普通执行
RUNS = 1000
t0 = time.perf_counter()
for _ in range(RUNS):
    run_many_kernels()
torch.cuda.synchronize()
normal_ms = (time.perf_counter() - t0) / RUNS * 1000
print(f"普通执行（{len(tensors)*3} kernels）：{normal_ms:.4f} ms/次")

# CUDA Graph
static_tensors = [t.clone() for t in tensors]
g4 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g4):
    x = static_tensors[0]
    for i in range(1, len(static_tensors)):
        x = torch.mm(x, static_tensors[i])
        x = x + static_tensors[0]
        x = torch.nn.functional.silu(x)

t0 = time.perf_counter()
for _ in range(RUNS):
    g4.replay()
torch.cuda.synchronize()
graph_ms = (time.perf_counter() - t0) / RUNS * 1000
print(f"CUDA Graph（{len(tensors)*3} kernels）：{graph_ms:.4f} ms/次")
print(f"加速比：{normal_ms / graph_ms:.2f}x")

print(f"\nWSL 下每次 launch ~15μs，{len(tensors)*3} 个 kernel 的 overhead：{len(tensors)*3*0.015:.1f}ms")
print(f"Graph 把这些压缩到 ~0.015ms（1次 launch）")