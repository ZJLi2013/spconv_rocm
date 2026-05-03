# spconv-rocm: voxelization utilities
# TODO: implement PyTorch-based voxelization or hipify C++ version
# For now, import guard to avoid breaking the package

import contextlib


def nullcontext():
    return contextlib.nullcontext()


class Point2VoxelCPU:
    """Stub — voxelization not yet ported to ROCm."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Point2Voxel is not yet available in spconv-rocm. "
            "Use torch_scatter or pre-computed voxel indices."
        )


Point2VoxelCPU1d = Point2VoxelCPU
Point2VoxelCPU2d = Point2VoxelCPU
Point2VoxelCPU3d = Point2VoxelCPU
Point2VoxelCPU4d = Point2VoxelCPU
Point2VoxelGPU1d = Point2VoxelCPU
Point2VoxelGPU2d = Point2VoxelCPU
Point2VoxelGPU3d = Point2VoxelCPU
Point2VoxelGPU4d = Point2VoxelCPU
