# spconv-rocm: minimal core definitions (stripped of CUDA codegen params)

from enum import Enum


class ConvAlgo(Enum):
    Native = 0
    MaskImplicitGemm = 1
    MaskSplitImplicitGemm = 2


class AlgoHint(Enum):
    NoHint = 0b000
    Fowrard = 0b001
    BackwardInput = 0b010
    BackwardWeight = 0b100
