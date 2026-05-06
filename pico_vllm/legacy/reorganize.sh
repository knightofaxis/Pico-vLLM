#!/bin/bash
# reorganize.sh — 一键重组 pico_vllm 目录结构
# 用法：cd PicovLLM && bash reorganize.sh
#
# 执行前请确保：
#   1. 所有改动已 git commit
#   2. git status 确认工作区干净
#
# 执行后手动做：
#   1. grep 检查旧 import 是否残留
#   2. 跑一次测试确认没破坏

set -e

echo "=== Step 1: 创建目录结构 ==="

cd pico_vllm

mkdir -p ops/triton
mkdir -p tests
mkdir -p benchmarks
mkdir -p profiling

echo "=== Step 2: 移动 Triton kernel ==="

mv Attention.py                    ops/triton/attention.py
mv fused_add_norm.py               ops/triton/fused_add_norm.py
mv Fused_RoPE_KVcache_store.py     ops/triton/fused_rope_kvcache_store.py
mv SwiGLU.py                       ops/triton/swiglu.py
mv store_kvcache.py                ops/triton/store_kvcache.py
touch ops/triton/__init__.py

echo "=== Step 3: 移动测试文件 ==="

for f in test_*.py; do
    [ -f "$f" ] && mv "$f" tests/
done
touch tests/__init__.py

echo "=== Step 4: 移动 benchmark 文件 ==="

for f in benchmark_*.py; do
    [ -f "$f" ] && mv "$f" benchmarks/
done

echo "=== Step 5: 移动 profiling 文件 ==="

for f in profile_*; do
    [ -f "$f" ] && mv "$f" profiling/
done

echo "=== Step 6: 创建 __init__.py ==="

cat > __init__.py << 'EOF'
"""
Pico-vLLM: 从零手写的 LLM 推理引擎
"""
EOF

echo "=== Step 7: 更新 kernel import 路径 ==="

find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/from Attention import/from ops.triton.attention import/g'
find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/import Attention/from ops.triton import attention as Attention/g'

find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/from fused_add_norm import/from ops.triton.fused_add_norm import/g'
find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/import fused_add_norm/from ops.triton import fused_add_norm/g'

find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/from Fused_RoPE_KVcache_store import/from ops.triton.fused_rope_kvcache_store import/g'
find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/import Fused_RoPE_KVcache_store/from ops.triton import fused_rope_kvcache_store as Fused_RoPE_KVcache_store/g'

find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/from SwiGLU import/from ops.triton.swiglu import/g'
find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/import SwiGLU/from ops.triton import swiglu as SwiGLU/g'

find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/from store_kvcache import/from ops.triton.store_kvcache import/g'
find . -name "*.py" -not -path "./ops/triton/*" | xargs sed -i \
    's/import store_kvcache/from ops.triton import store_kvcache/g'

echo "=== Step 8: 为 tests/benchmarks 添加 path ==="

cat > tests/conftest.py << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYEOF

cat > benchmarks/run_env.py << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PYEOF

for f in tests/test_*.py; do
    if [ -f "$f" ] && ! grep -q "sys.path" "$f"; then
        sed -i '1i\import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))' "$f"
    fi
done

for f in benchmarks/benchmark_*.py; do
    if [ -f "$f" ] && ! grep -q "sys.path" "$f"; then
        sed -i '1i\import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))' "$f"
    fi
done

echo ""
echo "=== 重组完成 ==="
echo ""
echo "目标结构："
find . -type f -name "*.py" | sort | head -30
echo ""
echo "检查步骤："
echo "  1. grep -rn 'from Attention import' .    # 应无结果"
echo "  2. grep -rn 'from SwiGLU import' .       # 应无结果"
echo "  3. grep -rn 'from Fused_RoPE' .          # 应无结果"
echo "  4. python model.py  或跑测试验证 import 正确"
