# spconv-rocm: simplified tools (no C++ CUDAKernelTimer)

import contextlib
from typing import Dict


class CUDAKernelTimer:
    """Stub timer — ROCm version does not use C++ kernel timing."""

    def __init__(self, enable: bool = True) -> None:
        self.enable = False
        self._timer = None

    def namespace(self, name: str):
        return contextlib.nullcontext()

    def record(self, name: str, stream: int = 0):
        return contextlib.nullcontext()

    def get_all_pair_time(self) -> Dict[str, float]:
        return {}

    @staticmethod
    def collect_by_name(name: str, res: Dict[str, float]):
        filtered_res: Dict[str, float] = {}
        for k, v in res.items():
            if name in k.split("."):
                filtered_res[k] = v
        return filtered_res
