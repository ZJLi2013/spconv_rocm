"""Test spconv-rocm sparse convolution on ROCm GPU.

Run: python -m pytest test/test_sparse_conv.py -v
Requires: PyTorch with ROCm support, AMD GPU available.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_sparse_input(batch_size, num_points, in_channels, spatial_shape, device):
    """Create a random SparseConvTensor."""
    import spconv.pytorch as spconv

    ndim = len(spatial_shape)
    # random coordinates within spatial shape
    coords = []
    for _ in range(batch_size):
        for _ in range(num_points):
            coord = [np.random.randint(0, s) for s in spatial_shape]
            coords.append(coord)

    indices = []
    for b in range(batch_size):
        for i in range(num_points):
            indices.append([b] + coords[b * num_points + i])

    indices = torch.tensor(indices, dtype=torch.int32, device=device)
    features = torch.randn(batch_size * num_points, in_channels,
                           dtype=torch.float32, device=device)

    return spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)


class TestSparseConv3d:
    def test_subm_conv_forward(self, device):
        """SubManifold conv: output indices == input indices."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [10, 10, 10]
        batch_size = 2
        num_points = 50

        conv = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=True).to(device)
        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        out = conv(x)

        assert out.features.shape == (batch_size * num_points, out_ch)
        assert out.spatial_shape == spatial_shape
        assert torch.equal(out.indices, x.indices)

    def test_sparse_conv_forward(self, device):
        """Standard sparse conv: output shape changes with stride."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [10, 10, 10]
        batch_size = 1
        num_points = 30

        conv = spconv.SparseConv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=True).to(device)
        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        out = conv(x)

        assert out.features.shape[1] == out_ch
        assert out.spatial_shape == [5, 5, 5]

    def test_conv1x1(self, device):
        """1x1 conv is just a dense matmul."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 32, 64
        spatial_shape = [8, 8, 8]
        batch_size = 1
        num_points = 20

        conv = spconv.SparseConv3d(in_ch, out_ch, 1, bias=False).to(device)
        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        out = conv(x)

        assert out.features.shape == (num_points, out_ch)
        ref = torch.mm(x.features, conv.weight)
        assert torch.allclose(out.features, ref, atol=1e-5)

    def test_sequential(self, device):
        """SparseSequential with conv + BN + ReLU."""
        import spconv.pytorch as spconv

        in_ch, mid_ch, out_ch = 16, 32, 64
        spatial_shape = [8, 8, 8]
        batch_size = 1
        num_points = 25

        model = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, mid_ch, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(mid_ch),
            spconv.SparseReLU(),
            spconv.SubMConv3d(mid_ch, out_ch, 3, padding=1, bias=False),
        ).to(device)

        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        out = model(x)

        assert out.features.shape == (num_points, out_ch)
        assert (out.features >= 0).any()  # ReLU applied

    def test_to_dense(self, device):
        """Convert SparseConvTensor to dense."""
        import spconv.pytorch as spconv

        in_ch = 4
        spatial_shape = [5, 5, 5]
        batch_size = 1
        num_points = 10

        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        dense = x.dense()

        assert dense.shape == (batch_size, in_ch, 5, 5, 5)
