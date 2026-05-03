# spconv-rocm: no C++ extensions needed
# All GEMM computation goes through FlyDSL via cumm-rocm

CPU_ONLY_BUILD = False
BUILD_CUMM_VERSION = "0.9.0.rocm1"
BUILD_PCCM_VERSION = "N/A"
HAS_BOOST = False
COMPILED_CUDA_ARCHS = set()
COMPILED_CUDA_GEMM_ARCHS = set()
