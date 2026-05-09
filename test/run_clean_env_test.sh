#!/bin/bash
set -e

echo "=== Step 1: Clean old installs ==="
rm -rf /tmp/cumm-rocm /tmp/spconv_rocm
pip uninstall -y cumm-rocm spconv-rocm 2>/dev/null || true

echo "=== Step 2: Clone fresh ==="
cd /tmp && git clone --branch rocm https://github.com/ZJLi2013/cumm-rocm.git 2>&1 | tail -2
cd /tmp && git clone --branch rocm https://github.com/ZJLi2013/spconv_rocm.git 2>&1 | tail -2

echo "=== Step 3: Install cumm-rocm ==="
cd /tmp/cumm-rocm && pip install -e . -q 2>&1 | tail -3

echo "=== Step 4: Install spconv-rocm ==="
cd /tmp/spconv_rocm && pip install -e . -q 2>&1 | tail -3

echo "=== Step 5: Verify installs ==="
pip show cumm-rocm 2>/dev/null | head -2
pip show spconv-rocm 2>/dev/null | head -2

echo "=== Step 6: Run cumm-rocm dispatch test ==="
cd /tmp/cumm-rocm && python3 -m pytest test/test_implicit_gemm.py::TestImplicitGemmDispatch -v --tb=short 2>&1 | tail -15

echo "=== Step 7: Run spconv-rocm full tests ==="
cd /tmp/spconv_rocm && python3 -m pytest test/test_sparse_conv.py -v --tb=short 2>&1 | tail -40

echo "=== Step 8: Verify implicit GEMM dispatch ==="
cd /tmp/spconv_rocm && python3 test/check_dispatch.py 2>&1 | tail -5

echo "=== Step 9: Run A/B benchmark ==="
cd /tmp/spconv_rocm && python3 test/bench_implicit_gemm.py 2>&1

echo "=== DONE ==="
