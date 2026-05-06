"""Compare HIP kernel vs PyTorch native indice pairs output."""
import torch
from spconv.pytorch.ops import _get_indice_pairs_gpu, _get_indice_pairs_hip

torch.manual_seed(42)
N = 20
spatial_shape = [7, 7, 7]
batch_size = 1
ksize = [3, 3, 3]
dilation = [1, 1, 1]
stride = [1, 1, 1]
padding = [1, 1, 1]

coords = torch.randint(0, 7, (N, 3), device="cuda")
batch_ids = torch.zeros(N, 1, dtype=torch.int32, device="cuda")
indices = torch.cat([batch_ids, coords], dim=1).int()
indices = torch.unique(indices, dim=0)
N = indices.shape[0]

out1, ip1, ipn1 = _get_indice_pairs_gpu(
    indices, batch_size, spatial_shape, ksize, stride, padding, dilation,
    [0, 0, 0], True, False)
out2, ip2, ipn2 = _get_indice_pairs_hip(
    indices, batch_size, spatial_shape, ksize, stride, padding, dilation,
    [0, 0, 0], True, False)

print(f"N: {N}")
print(f"ipn native: {ipn1.cpu().tolist()}")
print(f"ipn hip:    {ipn2.cpu().tolist()}")

kv = 27
mismatches = 0
for kid in range(kv):
    n1 = ipn1[kid].item()
    n2 = ipn2[kid].item()
    if n1 != n2:
        print(f"kid={kid}: count mismatch native={n1} hip={n2}")
        mismatches += 1
    min_n = min(n1, n2)
    if min_n > 0:
        # Sort pairs by input index for stable comparison
        # (atomicAdd order is non-deterministic)
        p1_0 = ip1[kid, 0, :min_n].cpu()
        p1_1 = ip1[kid, 1, :min_n].cpu()
        p2_0 = ip2[kid, 0, :min_n].cpu()
        p2_1 = ip2[kid, 1, :min_n].cpu()

        # Create (in, out) pairs and sort
        pairs1 = set(zip(p1_0.tolist(), p1_1.tolist()))
        pairs2 = set(zip(p2_0.tolist(), p2_1.tolist()))

        if pairs1 != pairs2:
            only_in_native = pairs1 - pairs2
            only_in_hip = pairs2 - pairs1
            print(f"kid={kid}: pair MISMATCH (n1={n1}, n2={n2})")
            if only_in_native:
                print(f"  only in native: {sorted(only_in_native)[:5]}")
            if only_in_hip:
                print(f"  only in hip:    {sorted(only_in_hip)[:5]}")
            mismatches += 1

if mismatches == 0:
    print("ALL MATCH!")
else:
    print(f"{mismatches} mismatches found")
