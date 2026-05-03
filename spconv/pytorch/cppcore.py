# spconv-rocm: pure PyTorch replacement for cppcore (no cumm.tensorview)

import torch
from typing import Dict, Optional
from spconv.constants import AllocKeys


# dtype mapping stubs (kept for interface compatibility)
_TORCH_DTYPE_TO_TV = {
    torch.float32: 0,
    torch.float64: 1,
    torch.float16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.int8: 5,
    torch.int16: 6,
    torch.uint8: 7,
    torch.bfloat16: 8,
}


def torch_tensor_to_tv(tensor: torch.Tensor):
    """Identity pass-through — we use torch.Tensor directly on ROCm."""
    return tensor


def get_current_stream() -> int:
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return 0


def get_arch():
    """Return (major, minor) for ROCm GPU. gfx942 → (9, 4)."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return (props.major, props.minor)
    return (0, 0)


class TorchAllocator:
    """Pure Python allocator using torch tensors (replaces C++ ExternalAllocator)."""

    def __init__(self, device):
        self.device = device
        self.allocated: Dict[str, torch.Tensor] = {}

    def zeros(self, name: str, shape, dtype, device=None):
        t = torch.zeros(shape, dtype=dtype, device=device or self.device)
        self.allocated[name] = t
        return t

    def empty(self, name: str, shape, dtype, device=None):
        t = torch.empty(shape, dtype=dtype, device=device or self.device)
        self.allocated[name] = t
        return t


class TorchSpconvMatmul:
    """ext_mm implementation using torch.mm — handles center/subm dense matmul."""

    def __init__(self, alloc: TorchAllocator):
        self.alloc = alloc

    def indice_conv_init_gemm(self, features: torch.Tensor, filters: torch.Tensor,
                              out_features: torch.Tensor, is_subm: bool, stream: int = 0):
        """Center position matmul for submanifold conv: out += features @ filters[center]."""
        if is_subm:
            torch.mm(features, filters, out=out_features)

    def indice_conv_cpu_gemm(self, features: torch.Tensor, filters: torch.Tensor,
                             out_features: torch.Tensor):
        torch.mm(features, filters, out=out_features)
