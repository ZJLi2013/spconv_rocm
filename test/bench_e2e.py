"""End-to-end spconv-rocm benchmark on MI300X.

Measures:
  1. get_indice_pairs (HIP kernel)
  2. indice_conv forward (gather → GEMM → scatter)
  3. full SparseConv3d.forward (indice_pairs + indice_conv)
  4. typical 3D detection backbone (SubM → BN → ReLU → Strided downsample)
"""
import time
import torch
import torch.nn as nn

def make_sparse_input(batch_size, num_points, in_channels, spatial_shape, device):
    import spconv.pytorch as spconv
    ndim = len(spatial_shape)
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
    indices = torch.cat(all_indices, dim=0)
    features = torch.cat(all_features, dim=0)
    indices = indices.int()
    return spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)


def bench(fn, warmup=5, repeats=20, label=""):
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
    print(f"  {label:<45} avg={avg:7.2f}ms  min={mi:7.2f}ms")
    return avg


def bench_single_conv(batch_size, num_points, in_ch, out_ch, spatial_shape,
                       ksize, stride, subm, device, label):
    import spconv.pytorch as spconv
    x = make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
    if subm:
        conv = spconv.SubMConv3d(in_ch, out_ch, ksize, bias=False).to(device)
    else:
        conv = spconv.SparseConv3d(in_ch, out_ch, ksize, stride=stride,
                                    padding=ksize//2, bias=False).to(device)
    conv.eval()

    print(f"\n--- {label} (N={x.indices.shape[0]}, C_in={in_ch}, C_out={out_ch}) ---")

    # Benchmark full forward
    with torch.no_grad():
        bench(lambda: conv(x), label="full forward")

    # Benchmark indice_pairs only
    from spconv.pytorch.ops import get_indice_pairs
    from spconv.core import ConvAlgo
    ndim = len(spatial_shape)
    ks = [ksize] * ndim if isinstance(ksize, int) else ksize
    st = [stride] * ndim if isinstance(stride, int) else stride
    pad = [k // 2 for k in ks]
    dil = [1] * ndim
    opad = [0] * ndim

    bench(lambda: get_indice_pairs(
        x.indices, batch_size, spatial_shape, ConvAlgo.Native,
        ks, st, pad, dil, opad, subm=subm),
        label="indice_pairs only")

    # Benchmark indice_conv only (reuse precomputed pairs)
    from spconv.pytorch.ops import indice_conv
    out_inds, ip, ipn = get_indice_pairs(
        x.indices, batch_size, spatial_shape, ConvAlgo.Native,
        ks, st, pad, dil, opad, subm=subm)
    n_out = out_inds.shape[0]
    w = conv.weight.data

    with torch.no_grad():
        bench(lambda: indice_conv(x.features, w, ip, ipn, n_out, subm=subm),
              label="indice_conv only (gather+GEMM+scatter)")


def bench_backbone(device):
    """Typical 3D detection backbone: 4 stages of SubM + strided downsample."""
    import spconv.pytorch as spconv

    batch_size = 2
    num_points = 20000
    spatial_shape = [200, 200, 10]  # typical voxel grid (X, Y, Z)

    backbone = spconv.SparseSequential(
        # Stage 1: SubM 16→32
        spconv.SubMConv3d(16, 32, 3, bias=False),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        # Downsample 1: stride=2
        spconv.SparseConv3d(32, 64, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        # Stage 2: SubM 64→64
        spconv.SubMConv3d(64, 64, 3, bias=False),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        # Downsample 2: stride=2
        spconv.SparseConv3d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(128),
        nn.ReLU(),
    ).to(device)
    backbone.eval()

    x = make_sparse_input(batch_size, num_points, 16, spatial_shape, device)

    print(f"\n--- 3D Detection Backbone (B={batch_size}, N={num_points}/batch, "
          f"spatial={spatial_shape}) ---")
    print(f"  Architecture: SubM(16→32) → ↓2(32→64) → SubM(64→64) → ↓2(64→128)")

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            backbone(x)
        torch.cuda.synchronize()

        bench(lambda: backbone(x), label="backbone forward (4 layers)")

    # Also measure backward
    x_train = make_sparse_input(batch_size, num_points, 16, spatial_shape, device)
    backbone.train()
    def fwd_bwd():
        out = backbone(x_train)
        loss = out.features.sum()
        loss.backward()
    bench(fwd_bwd, label="backbone forward + backward")


def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"ROCm/CUDA: {torch.version.cuda}")

    # Single conv benchmarks
    bench_single_conv(1, 5000, 32, 32, [100,100,100], 3, 1, True, device,
                      "SubM 3x3x3, N=5k, C=32")
    bench_single_conv(1, 20000, 32, 64, [200,200,200], 3, 1, True, device,
                      "SubM 3x3x3, N=20k, C=32→64")
    bench_single_conv(1, 50000, 16, 32, [200,200,200], 3, 1, True, device,
                      "SubM 3x3x3, N=50k, C=16→32")
    bench_single_conv(1, 20000, 64, 128, [200,200,200], 3, 2, False, device,
                      "Conv 3x3x3 s=2, N=20k, C=64→128")

    # Backbone benchmark
    bench_backbone(device)


if __name__ == "__main__":
    main()
