"""Quick check: verify implicit GEMM dispatch is active."""
import torch
import spconv.pytorch as spconv
from spconv.pytorch import ops

conv = spconv.SubMConv3d(32, 32, 3).cuda()

N = 5000
spatial = [100, 100, 100]
coords = torch.randint(0, min(spatial), (N, 3), device="cuda")
batch_idx = torch.zeros(N, 1, dtype=torch.int32, device="cuda")
indices = torch.cat([batch_idx, coords.int()], dim=1)
features = torch.randn(N, 32, device="cuda")

from spconv.pytorch.core import SparseConvTensor
x = SparseConvTensor(features, indices, spatial, 1)

with torch.no_grad():
    out = conv(x)

print(f"Output shape: {out.features.shape}")
print(f"Implicit GEMM available: {ops._IMPLICIT_GEMM_AVAILABLE}")
if ops._IMPLICIT_GEMM_AVAILABLE:
    print("SUCCESS: implicit GEMM path is active")
else:
    print("INFO: implicit GEMM NOT used (fell through to native)")
