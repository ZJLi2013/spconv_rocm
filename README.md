# SpConv-ROCm: Sparse Convolution for AMD GPUs

Sparse convolution library for AMD ROCm platform, forked from [spconv](https://github.com/traveller59/spconv).

## Overview

This is a **pure Python** reimplementation of spconv's Native (gather-GEMM-scatter) path, targeting AMD MI300X (gfx942) and MI350 (gfx950) GPUs. All CUDA/pccm/C++ dependencies have been removed.

## Architecture

```
Input SparseConvTensor
    │
    ├── get_indice_pairs()    ← Python/CPU (dict hash)
    │
    ├── gather (torch.index_select)
    │
    ├── GEMM (torch.mm → FlyDSL hgemm_splitk)
    │
    ├── scatter (torch.scatter_add / index_add)
    │
    └── Output SparseConvTensor
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0 (ROCm build)
- [cumm-rocm](https://github.com/ZJLi2013/cumm-rocm) (provides FlyDSLGemmTuner)

## Install

```bash
pip install -e .
```

## Usage

```python
import spconv.pytorch as spconv
import torch

# Same API as original spconv
conv = spconv.SparseConv3d(in_channels=16, out_channels=32, kernel_size=3)
```

## Current Status (v2.3.8.rocm1)

- [x] `ConvAlgo.Native` path (gather-GEMM-scatter)
- [x] `SparseConv3d`, `SubMConv3d`, `SparseConvTranspose3d`
- [x] `SparseSequential`, `ToDense`
- [x] `SparseMaxPool3d`, `SparseAvgPool3d`
- [x] Pure Python — no C++ compilation needed
- [ ] End-to-end remote GPU validation (pending)

## Performance Notes

Current implementation prioritizes **correctness over speed**:

| Component | Implementation | vs Original |
|-----------|---------------|-------------|
| GEMM | `torch.mm` (→ FlyDSL) | ~1x (same hardware MFMA) |
| indice pairs | Python/CPU `dict` | **10-100x slower** |
| conv path | Native only (no implicit GEMM) | 2-3x slower |

## Next Steps — Performance Optimization

| Priority | Task | Approach |
|----------|------|----------|
| P0 | GPU indice pairs | `torch.unique` + `torch.searchsorted` on GPU |
| P1 | implicit GEMM | FlyDSL fused gather+GEMM+scatter kernel |
| P2 | hipify indice kernel | `hipify-perl` on pccm-generated .cu → hipcc |

## License

Apache 2.0
