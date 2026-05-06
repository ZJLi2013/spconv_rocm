# SpConv-ROCm: Sparse Convolution for AMD GPUs

Sparse convolution library for AMD ROCm GPUs, forked from [spconv](https://github.com/traveller59/spconv).
Replaces 33k lines of pccm/cumm C++ codegen with ~1.5k lines of Python + self-contained HIP kernels.

## Architecture

```
Original:  Python → pccm (codegen) → C++ → nvcc → GPU     (152 files, 33k lines)
ROCm:      Python → HIP kernel + torch.mm/FlyDSL → GPU    (~1.5k lines)
```

| Component | Original | ROCm |
|-----------|----------|------|
| GEMM | pccm C++ codegen (~20k lines) | `torch.mm` (fp32) / FlyDSL `hgemm_splitk` (fp16/bf16) |
| indice pairs | CUDA hash table (pccm/tensorview) | Self-contained HIP kernel (Murmur3 + open-addressing) |
| indice_conv loop | C++ for-loop (pccm) | C++ for-loop (JIT compiled, ATen API) |
| Python binding | pccm → nvcc → .so | `torch.utils.cpp_extension.load()` JIT |

## Requirements

- Python >= 3.8
- PyTorch >= 2.0 (ROCm build)
- [cumm-rocm](https://github.com/ZJLi2013/cumm-rocm) (provides FlyDSLGemmTuner for fp16/bf16)
- ROCm 6.x+ with hipCUB

## Install

```bash
# Install cumm-rocm first
git clone https://github.com/ZJLi2013/cumm-rocm.git
cd cumm-rocm && git checkout rocm && pip install -e .

# Install spconv-rocm
git clone https://github.com/ZJLi2013/spconv_rocm.git
cd spconv_rocm && git checkout rocm && pip install -e .
```

HIP kernels are JIT-compiled on first use (~10s).

## Usage

```python
import spconv.pytorch as spconv
import torch

# Same API as original spconv
conv = spconv.SubMConv3d(16, 32, 3).cuda()
# SparseConv3d, SparseConvTranspose3d, SparseSequential, etc. all work
```

## Features

- [x] `SubMConv3d`, `SparseConv3d`, `SparseConvTranspose3d`
- [x] `SparseSequential`, `ToDense`
- [x] `SparseMaxPool3d`, `SparseAvgPool3d`
- [x] Forward + backward (training supported)
- [x] fp32, fp16, bf16, AMP
- [x] 1D, 2D, 3D sparse convolution
- [x] indice_key reuse, dilation, bias
- [x] Encoder-Decoder (UNet) architecture
- [x] 28 tests passing on MI308X (gfx942)

## Performance (MI308X gfx942)

### Single layer (SubM 3x3x3)

| Component | N=5k, C=32 | N=50k, C=16→32 |
|-----------|-----------|----------------|
| indice_pairs (HIP kernel) | 0.46ms | 0.52ms |
| indice_conv (C++ gather+GEMM+scatter) | 1.45ms | 1.43ms |
| **full forward** | **1.93ms** | **1.98ms** |

### 4-layer backbone (typical 3D detection)

```
SubM(16→32) → Stride2(32→64) → SubM(64→64) → Stride2(64→128)
B=2, N=20k/batch, spatial=[200,200,10]
```

| Phase | Time |
|-------|------|
| Forward | **11.2ms** |
| Forward + Backward | **26.9ms** |

### indice pairs: HIP kernel vs alternatives

| Implementation | SubM N=50k, 3x3x3 |
|---------------|-------------------|
| CPU Python dict | ~200-500ms |
| GPU PyTorch native (sort+searchsorted) | 15.2ms |
| **GPU HIP kernel (current)** | **0.6ms** |
| Original CUDA (reference) | ~2ms |

## Code Structure

```
spconv/
  pytorch/
    ops.py           — GEMM dispatch (FlyDSL/torch.mm), indice_conv, pooling
    conv.py          — SparseConv3d, SubMConv3d modules
    core.py          — SparseConvTensor
  csrc_hip/
    hash_table.h           — GPU hash table (Murmur3 + open-addressing, ~95 lines)
    indice_pairs_kernel.hip — SubM/Regular conv indice pairs HIP kernels (~470 lines)
    indice_pairs_api.cpp    — PyTorch C++ extension binding + indice_conv C++ loop (~230 lines)
test/
  test_sparse_conv.py  — 28 tests (correctness + numerical validation)
  bench_e2e.py         — End-to-end benchmark
  bench_indice_pairs.py — indice pairs kernel benchmark
```

## Next Step: Implicit GEMM (fused gather+GEMM+scatter)

Current bottleneck is `indice_conv`: 27 independent small GEMMs with separate gather/scatter per kernel position (~1.4ms, 72% of forward time).

**Implicit GEMM** fuses gather + GEMM + scatter into a single kernel launch:

```
Current (Native path):
  for k in 0..26:
    buf = features[input_indices[k]]     ← gather kernel
    out = buf × weight[k]               ← GEMM kernel
    output[output_indices[k]] += out     ← scatter kernel
  Total: 81 kernel launches

Implicit GEMM:
  1 fused kernel:
    GEMM tile directly reads features via indices (no temp buffer)
    All 27 positions computed in one launch
    Results written directly to output
  Total: 1 kernel launch
```

This is the approach used by the original spconv (`ConvAlgo.MaskImplicitGemm` via cumm). The ROCm version will implement this as a FlyDSL/HIP fused kernel.

## License

Apache 2.0
