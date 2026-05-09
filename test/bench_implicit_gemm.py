"""A/B benchmark: native gather+GEMM+scatter vs cumm-rocm implicit GEMM.

Runs each config with both paths, side-by-side, to quantify the performance
delta from the implicit GEMM integration.

Usage:
    PYTHONPATH=/tmp/cumm-rocm:/tmp/spconv_rocm python3 test/bench_implicit_gemm.py
"""
import time
import torch
import torch.nn as nn


def make_sparse_input(batch_size, num_points, in_channels, spatial_shape, device):
    import spconv.pytorch as spconv
    all_indices = []
    all_features = []
    for b in range(batch_size):
        coords = torch.stack([
            torch.randint(0, s, (num_points,), device=device) for s in spatial_shape
        ], dim=1)
        batch_col = torch.full((num_points, 1), b, dtype=torch.int32, device=device)
        idx = torch.cat([batch_col, coords], dim=1)
        all_indices.append(idx)
        all_features.append(torch.randn(num_points, in_channels, device=device))
    indices = torch.cat(all_indices, dim=0).int()
    features = torch.cat(all_features, dim=0)
    return spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)


def bench(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) * 1000
        times.append(t)
    avg = sum(times) / len(times)
    mi = min(times)
    return avg, mi


def bench_indice_conv_ab(features, weight, ip, ipn, n_out, subm, label):
    """Run indice_conv with implicit GEMM on/off."""
    from spconv.pytorch import ops

    results = {}

    # --- implicit GEMM path ---
    ops._IMPLICIT_GEMM_AVAILABLE = None  # reset
    with torch.no_grad():
        avg, mi = bench(lambda: ops.indice_conv(features, weight, ip, ipn, n_out, subm=subm))
    used_implicit = ops._IMPLICIT_GEMM_AVAILABLE
    tag = "implicit_gemm" if used_implicit else "native(fallback)"
    results["implicit"] = (avg, mi, tag)

    # --- force native path ---
    saved = ops._IMPLICIT_GEMM_AVAILABLE
    ops._IMPLICIT_GEMM_AVAILABLE = False
    with torch.no_grad():
        avg_n, mi_n = bench(lambda: ops.indice_conv(features, weight, ip, ipn, n_out, subm=subm))
    results["native"] = (avg_n, mi_n, "native(C++ at::mm)")
    ops._IMPLICIT_GEMM_AVAILABLE = saved

    print(f"\n  {label}")
    for key in ["native", "implicit"]:
        avg, mi, tag = results[key]
        print(f"    {tag:<30s}  avg={avg:7.2f}ms  min={mi:7.2f}ms")

    avg_i = results["implicit"][0]
    avg_n = results["native"][0]
    speedup = avg_n / avg_i if avg_i > 0 else 0
    delta = (avg_i / avg_n - 1) * 100
    print(f"    => implicit vs native: {delta:+.1f}%  (speedup={speedup:.2f}x)")
    return results


def bench_single_conv_ab(batch_size, num_points, in_ch, out_ch, spatial_shape,
                          ksize, stride, subm, device, label):
    import spconv.pytorch as spconv
    from spconv.pytorch.ops import get_indice_pairs, indice_conv
    from spconv.core import ConvAlgo
    from spconv.pytorch import ops

    x = make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
    if subm:
        conv = spconv.SubMConv3d(in_ch, out_ch, ksize, bias=False).to(device)
    else:
        conv = spconv.SparseConv3d(in_ch, out_ch, ksize, stride=stride,
                                    padding=ksize//2, bias=False).to(device)
    conv.eval()

    ndim = len(spatial_shape)
    ks = [ksize] * ndim
    st = [stride] * ndim
    pad = [k // 2 for k in ks]
    dil = [1] * ndim
    opad = [0] * ndim

    out_inds, ip, ipn = get_indice_pairs(
        x.indices, batch_size, spatial_shape, ConvAlgo.Native,
        ks, st, pad, dil, opad, subm=subm)
    n_out = out_inds.shape[0]
    w = conv.weight.data

    print(f"\n--- {label} (N={x.indices.shape[0]}, C_in={in_ch}, C_out={out_ch}) ---")

    bench_indice_conv_ab(x.features, w, ip, ipn, n_out, subm, "indice_conv")

    # Full forward A/B
    with torch.no_grad():
        # implicit
        ops._IMPLICIT_GEMM_AVAILABLE = None
        avg_i, mi_i = bench(lambda: conv(x))
        tag_i = "implicit_gemm" if ops._IMPLICIT_GEMM_AVAILABLE else "native(fallback)"

        # native
        ops._IMPLICIT_GEMM_AVAILABLE = False
        avg_n, mi_n = bench(lambda: conv(x))
        ops._IMPLICIT_GEMM_AVAILABLE = None

    print(f"\n  full forward:")
    print(f"    native(C++ at::mm)              avg={avg_n:7.2f}ms  min={mi_n:7.2f}ms")
    print(f"    {tag_i:<30s}  avg={avg_i:7.2f}ms  min={mi_i:7.2f}ms")
    delta = (avg_i / avg_n - 1) * 100
    print(f"    => implicit vs native: {delta:+.1f}%")

    return {
        "label": label,
        "conv_native": avg_n, "conv_implicit": avg_i,
    }


def bench_backbone_ab(device):
    import spconv.pytorch as spconv
    from spconv.pytorch import ops

    batch_size = 2
    num_points = 20000
    spatial_shape = [200, 200, 10]

    backbone = spconv.SparseSequential(
        spconv.SubMConv3d(16, 32, 3, bias=False),
        nn.BatchNorm1d(32), nn.ReLU(),
        spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(64), nn.ReLU(),
        spconv.SubMConv3d(64, 64, 3, bias=False),
        nn.BatchNorm1d(64), nn.ReLU(),
        spconv.SparseConv3d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(128), nn.ReLU(),
    ).to(device)
    backbone.eval()

    x = make_sparse_input(batch_size, num_points, 16, spatial_shape, device)

    print(f"\n--- Backbone (B={batch_size}, N={num_points}/batch, spatial={spatial_shape}) ---")
    print(f"  SubM(16→32) → ↓2(32→64) → SubM(64→64) → ↓2(64→128)")

    with torch.no_grad():
        # implicit GEMM
        ops._IMPLICIT_GEMM_AVAILABLE = None
        avg_i, mi_i = bench(lambda: backbone(x))

        # native
        ops._IMPLICIT_GEMM_AVAILABLE = False
        avg_n, mi_n = bench(lambda: backbone(x))
        ops._IMPLICIT_GEMM_AVAILABLE = None

    print(f"\n  backbone forward:")
    print(f"    native(C++ at::mm)              avg={avg_n:7.2f}ms  min={mi_n:7.2f}ms")
    print(f"    implicit_gemm                   avg={avg_i:7.2f}ms  min={mi_i:7.2f}ms")
    delta = (avg_i / avg_n - 1) * 100
    speedup = avg_n / avg_i if avg_i > 0 else 0
    print(f"    => implicit vs native: {delta:+.1f}%  (speedup={speedup:.2f}x)")

    return {"native": avg_n, "implicit": avg_i}


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"ROCm/CUDA: {torch.version.cuda}")
    print("=" * 70)

    configs = [
        (1, 5000, 32, 32, [100,100,100], 3, 1, True, "SubM 3x3x3, N=5k, C=32"),
        (1, 20000, 32, 64, [200,200,200], 3, 1, True, "SubM 3x3x3, N=20k, C=32→64"),
        (1, 50000, 16, 32, [200,200,200], 3, 1, True, "SubM 3x3x3, N=50k, C=16→32"),
        (1, 20000, 64, 128, [200,200,200], 3, 2, False, "Conv 3x3x3 s=2, N=20k, C=64→128"),
    ]

    results = []
    for bs, n, ci, co, sp, ks, st, subm, label in configs:
        r = bench_single_conv_ab(bs, n, ci, co, sp, ks, st, subm, device, label)
        results.append(r)

    bb = bench_backbone_ab(device)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<40s} {'Native(ms)':>10s} {'Implicit(ms)':>12s} {'Delta':>8s}")
    print("-" * 72)
    for r in results:
        delta = (r["conv_implicit"] / r["conv_native"] - 1) * 100
        print(f"{r['label']:<40s} {r['conv_native']:10.2f} {r['conv_implicit']:12.2f} {delta:+7.1f}%")
    delta_bb = (bb["implicit"] / bb["native"] - 1) * 100
    print(f"{'backbone forward':<40s} {bb['native']:10.2f} {bb['implicit']:12.2f} {delta_bb:+7.1f}%")


if __name__ == "__main__":
    main()
