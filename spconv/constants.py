# spconv-rocm constants (cleaned of pccm/cumm CUDA dependencies)

import os
from pathlib import Path
import enum

PACKAGE_NAME = "spconv-rocm"
PACKAGE_ROOT = Path(__file__).parent.resolve()

NDIM_DONT_CARE = 3
FILTER_HWIO = False
ALL_WEIGHT_IS_KRSC = True

SAVED_WEIGHT_LAYOUT = os.getenv("SPCONV_SAVED_WEIGHT_LAYOUT", "")
if SAVED_WEIGHT_LAYOUT != "":
    assert SAVED_WEIGHT_LAYOUT in ["KRSC", "RSKC", "RSCK"]

SPCONV_BWD_SPLITK = list(map(int, os.getenv("SPCONV_BWD_SPLITK", "1,2,4,8,16,32,64").split(",")))

SPCONV_DEBUG_SAVE_PATH = os.getenv("SPCONV_DEBUG_SAVE_PATH", "")
SPCONV_DEBUG_WEIGHT = False
SPCONV_DEBUG_CPP_ONLY = False
DISABLE_JIT = True

# ROCm: force Python-based GEMM path (no C++ extension)
SPCONV_CPP_GEMM = False
SPCONV_CPP_INDICE_PAIRS = False
SPCONV_CPP_INDICE_PAIRS_IGEMM = False
SPCONV_USE_DIRECT_TABLE = False
SPCONV_FX_TRACE_MODE = False
SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE = 1.1
SPCONV_ALLOW_TF32 = False
SPCONV_INT8_DEBUG = False
SPCONV_DO_SORT = os.getenv("SPCONV_DO_SORT", "1") == "1"

# NVRTCMode stub (not used on ROCm)
SPCONV_NVRTC_MODE = 0
SPCONV_DEBUG_NVRTC_KERNELS = False


class AllocKeys:
    PairBwd = "PairBwd"
    IndiceNumPerLoc = "IndiceNumPerLoc"
    PairMask = "PairMask"
    MaskArgSort = "MaskArgSort"
    OutIndices = "OutIndices"
    PairFwd = "PairFwd"
    PairMaskBwd = "PairMaskBwd"
    MaskArgSortBwd = "MaskArgSortBwd"
    MaskOutputFwd = "MaskOutputFwd"
    OutFeatures = "OutFeatures"
    Features = "Features"
    Filters = "Filters"
    OutBp = "OutBp"
    DIn = "DIn"
    DFilters = "DFilters"
    InpBuffer = "InpBuffer"
    OutBuffer = "OutBuffer"
    IndicePairsUniq = "IndicePairsUniq"
    IndicePairsUniqBackup = "IndicePairsUniqBackup"
    HashKOrKV = "HashKOrKV"
    HashV = "HashV"
    ThrustTemp = "ThrustTemp"
    TightUniqueCount = "TightUniqueCount"
