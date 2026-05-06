"""Benchmark: HIP kernel vs PyTorch native indice pairs."""
import time
import torch
from spconv.pytorch.ops import _get_indice_pairs_gpu, _get_indice_pairs_hip, _get_hip_module

def make_sparse_indices(N, spatial_shape, batch_size, device):
    ndim = len(spatial_shape)
    coords = torch.stack([
        torch.randint(0, s, (N,), device=device) for s in spatial_shape
    ], dim=1)
    batch_ids = torch.randint(0, batch_size, (N, 1), device=device, dtype=torch.int32)
    indices = torch.cat([batch_ids, coords], dim=1).int()
    indices = torch.unique(indices, dim=0)
    return indices

def bench_fn(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return min(times), sum(times) / len(times)

def main():
    device = "cuda"
    configs = [
        {"label": "SubM N=1k,  3x3x3", "N": 1000,  "spatial": [40,40,40], "ksize": [3,3,3], "subm": True},
        {"label": "SubM N=10k, 3x3x3", "N": 10000, "spatial": [80,80,80], "ksize": [3,3,3], "subm": True},
        {"label": "SubM N=50k, 3x3x3", "N": 50000, "spatial": [200,200,200], "ksize": [3,3,3], "subm": True},
        {"label": "Conv N=10k, 3x3x3 s=2", "N": 10000, "spatial": [80,80,80], "ksize": [3,3,3], "subm": False},
    ]

    hip = _get_hip_module()
    print(f"HIP module loaded: {hip is not None}")
    print()
    print(f"{'Config':<30} {'N_actual':>8} {'Native (ms)':>12} {'HIP (ms)':>10} {'Speedup':>8}")
    print("-" * 72)

    for cfg in configs:
        torch.manual_seed(42)
        indices = make_sparse_indices(cfg["N"], cfg["spatial"], 1, device)
        N_actual = indices.shape[0]
        ksize = cfg["ksize"]
        ndim = len(ksize)
        dilation = [1] * ndim
        stride = [1] * ndim if cfg["subm"] else [2] * ndim
        padding = [k // 2 for k in ksize]
        out_padding = [0] * ndim

        def run_native():
            return _get_indice_pairs_gpu(
                indices, 1, cfg["spatial"], ksize, stride, padding,
                dilation, out_padding, cfg["subm"], False)

        def run_hip():
            return _get_indice_pairs_hip(
                indices, 1, cfg["spatial"], ksize, stride, padding,
                dilation, out_padding, cfg["subm"], False)

        t_native_min, t_native_avg = bench_fn(run_native)
        t_hip_min, t_hip_avg = bench_fn(run_hip)
        speedup = t_native_avg / t_hip_avg

        print(f"{cfg['label']:<30} {N_actual:>8} {t_native_avg:>10.1f}ms {t_hip_avg:>8.1f}ms {speedup:>7.1f}x")

if __name__ == "__main__":
    main()
