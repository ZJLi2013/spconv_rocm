# spconv-rocm: Pure Python ops (gather-GEMM-scatter via FlyDSLGemmTuner)
# Replaces C++ ConvGemmOps + cumm CUDA kernels

import torch
import numpy as np
from typing import List, Optional, Tuple
from spconv.core import ConvAlgo
from spconv.constants import ALL_WEIGHT_IS_KRSC, AllocKeys

# FlyDSL GEMM backend (via cumm-rocm)
_FLYDSL_TUNER = None

def _get_flydsl_tuner():
    """Lazy-init FlyDSLGemmTuner (only when FlyDSL is available)."""
    global _FLYDSL_TUNER
    if _FLYDSL_TUNER is None:
        try:
            from cumm.gemm_tuner import FlyDSLGemmTuner
            _FLYDSL_TUNER = FlyDSLGemmTuner()
        except ImportError:
            _FLYDSL_TUNER = False  # FlyDSL not available, fallback to torch.mm
    return _FLYDSL_TUNER


def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM dispatch: FlyDSL for fp16/bf16, torch.mm for fp32.
    
    a: [M, K], b: [K, N] → output: [M, N]
    """
    tuner = _get_flydsl_tuner()
    if tuner and a.dtype in (torch.float16, torch.bfloat16):
        m, k = a.shape
        n = b.shape[1]
        # FlyDSL interface: C = A @ B^T, so B must be [N, K]
        b_t = b.t().contiguous()  # [K, N] → [N, K]
        c = torch.zeros(m, n, dtype=a.dtype, device=a.device)
        tuner.matmul(c, a.contiguous(), b_t)
        return c
    else:
        return torch.mm(a, b)


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] *
                (kernel_size[i] - 1) - 1) // stride[i] + 1
        if size < 0:
            size = 0
        output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation,
                           output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + dilation[i] * (
            kernel_size[i] - 1) + output_padding[i] + 1
        output_size.append(size)
    return output_size


def get_indice_pairs(indices: torch.Tensor,
                     batch_size: int,
                     spatial_shape: List[int],
                     algo: ConvAlgo,
                     ksize: List[int],
                     stride: List[int],
                     padding: List[int],
                     dilation: List[int],
                     out_padding: List[int],
                     subm: bool = False,
                     transposed: bool = False,
                     is_train: bool = True,
                     alloc=None,
                     timer=None):
    """Compute indice pairs for sparse convolution (CPU implementation).
    
    Returns:
        out_inds: output indices [N_out, ndim+1]
        indice_pairs: [kv, 2, N_max] — gather/scatter index pairs
        indice_pair_num: [kv] — number of active pairs per kernel position
    """
    ndim = len(spatial_shape)
    kv = int(np.prod(ksize))
    device = indices.device
    num_points = indices.shape[0]

    if subm:
        out_inds = indices
        num_out = num_points
    else:
        out_inds, num_out = _compute_output_indices(
            indices, batch_size, spatial_shape, ksize, stride,
            padding, dilation, out_padding, transposed)

    indice_pairs = torch.full((kv, 2, max(num_points, num_out)), -1,
                              dtype=torch.int32, device=device)
    indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)

    _build_indice_pairs(indices, out_inds, indice_pairs, indice_pair_num,
                        batch_size, spatial_shape, ksize, stride, padding,
                        dilation, subm, transposed)

    return out_inds, indice_pairs, indice_pair_num


def _compute_output_indices(indices, batch_size, spatial_shape, ksize,
                            stride, padding, dilation, out_padding, transposed):
    """Compute output spatial indices for non-submanifold conv."""
    ndim = len(spatial_shape)
    if transposed:
        out_spatial = get_deconv_output_size(spatial_shape, ksize, stride,
                                             padding, dilation, out_padding)
    else:
        out_spatial = get_conv_output_size(spatial_shape, ksize, stride,
                                           padding, dilation)

    # For each input point, compute all possible output positions
    coords = indices[:, 1:].cpu().numpy()  # [N, ndim]
    batch_ids = indices[:, 0].cpu().numpy()  # [N]

    out_coords_set = set()
    for i in range(len(coords)):
        for offset in np.ndindex(*ksize):
            if transposed:
                out_coord = tuple(
                    (coords[i][d] + padding[d] - offset[d] * dilation[d]) * stride[d] +
                    offset[d] * dilation[d]
                    for d in range(ndim)
                )
            else:
                out_coord = tuple(
                    (coords[i][d] + padding[d] - offset[d] * dilation[d]) // stride[d]
                    for d in range(ndim)
                )
            valid = all(0 <= out_coord[d] < out_spatial[d] for d in range(ndim))
            if valid:
                out_coords_set.add((int(batch_ids[i]),) + out_coord)

    if len(out_coords_set) == 0:
        out_inds = torch.zeros((0, ndim + 1), dtype=indices.dtype, device=indices.device)
        return out_inds, 0

    out_list = sorted(out_coords_set)
    out_inds = torch.tensor(out_list, dtype=indices.dtype, device=indices.device)
    return out_inds, len(out_list)


def _build_indice_pairs(indices, out_inds, indice_pairs, indice_pair_num,
                        batch_size, spatial_shape, ksize, stride, padding,
                        dilation, subm, transposed):
    """Build gather/scatter index pairs between input and output."""
    ndim = len(spatial_shape)
    kv = int(np.prod(ksize))
    device = indices.device

    # Build hash map: (batch, z, y, x, ...) → index
    in_coords = indices.cpu().numpy()
    out_coords = out_inds.cpu().numpy()

    in_hash = {}
    for i, c in enumerate(in_coords):
        in_hash[tuple(c)] = i

    out_hash = {}
    for i, c in enumerate(out_coords):
        out_hash[tuple(c)] = i

    # For submanifold conv, input == output
    if subm:
        ref_coords = in_coords
        ref_hash = in_hash
    else:
        ref_coords = out_coords
        ref_hash = out_hash

    ksize_arr = np.array(ksize)
    offsets = list(np.ndindex(*ksize))

    for kid, offset in enumerate(offsets):
        count = 0
        for out_idx, out_c in enumerate(out_coords):
            batch_id = out_c[0]
            out_spatial = out_c[1:]

            if subm:
                in_spatial = tuple(
                    int(out_spatial[d] + offset[d] - ksize[d] // 2)
                    for d in range(ndim)
                )
            else:
                in_spatial = tuple(
                    int(out_spatial[d] * stride[d] - padding[d] + offset[d] * dilation[d])
                    for d in range(ndim)
                )

            in_key = (int(batch_id),) + in_spatial
            if in_key in in_hash:
                in_idx = in_hash[in_key]
                indice_pairs[kid, 0, count] = in_idx
                indice_pairs[kid, 1, count] = out_idx
                count += 1

        indice_pair_num[kid] = count


def indice_conv(features: torch.Tensor,
                filters: torch.Tensor,
                indice_pairs: torch.Tensor,
                indice_pair_num: torch.Tensor,
                num_activate_out: int,
                inverse: bool = False,
                subm: bool = False,
                algo: ConvAlgo = ConvAlgo.Native,
                timer=None) -> torch.Tensor:
    """Native path sparse convolution: gather → GEMM → scatter.
    
    Args:
        features: [N_in, C_in] input features
        filters: [kv, C_in, C_out] (KRSC layout with ALL_WEIGHT_IS_KRSC=True)
        indice_pairs: [kv, 2, N_max] gather/scatter indices
        indice_pair_num: [kv] active pair count per kernel position
        num_activate_out: number of output points
    
    Returns:
        out_features: [N_out, C_out]
    """
    kv = filters.shape[0]
    c_in = features.shape[1]
    c_out = filters.shape[2]
    device = features.device
    dtype = features.dtype

    out_features = torch.zeros(num_activate_out, c_out, dtype=dtype, device=device)

    for i in range(kv):
        n_act = int(indice_pair_num[i].item())
        if n_act == 0:
            continue

        inp_inds = indice_pairs[i, 0, :n_act].long()
        out_inds = indice_pairs[i, 1, :n_act].long()

        # gather input features
        inp_gathered = features[inp_inds]  # [n_act, C_in]

        # GEMM: [n_act, C_in] @ [C_in, C_out] → [n_act, C_out]
        w = filters[i]  # [C_in, C_out]
        gemm_out = _gemm(inp_gathered, w)

        # scatter-add to output
        out_features.index_add_(0, out_inds, gemm_out)

    return out_features


def indice_conv_backward(features: torch.Tensor,
                         filters: torch.Tensor,
                         out_bp: torch.Tensor,
                         indice_pairs: torch.Tensor,
                         indice_pair_num: torch.Tensor,
                         inverse: bool = False,
                         subm: bool = False,
                         algo: ConvAlgo = ConvAlgo.Native,
                         timer=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for native sparse conv.
    
    Returns:
        din: gradient w.r.t. input features [N_in, C_in]
        dfilters: gradient w.r.t. filters [kv, C_in, C_out]
    """
    kv = filters.shape[0]
    c_in = features.shape[1]
    c_out = filters.shape[2]
    n_in = features.shape[0]
    device = features.device
    dtype = features.dtype

    din = torch.zeros(n_in, c_in, dtype=dtype, device=device)
    dfilters = torch.zeros_like(filters)

    for i in range(kv):
        n_act = int(indice_pair_num[i].item())
        if n_act == 0:
            continue

        inp_inds = indice_pairs[i, 0, :n_act].long()
        out_inds = indice_pairs[i, 1, :n_act].long()

        # gather
        inp_gathered = features[inp_inds]  # [n_act, C_in]
        out_bp_gathered = out_bp[out_inds]  # [n_act, C_out]

        w = filters[i]  # [C_in, C_out]

        # dfilters[i] = inp_gathered^T @ out_bp_gathered
        dfilters[i] = _gemm(inp_gathered.t(), out_bp_gathered)

        # din: scatter grad back to input
        # din_gathered = out_bp_gathered @ w^T
        din_gathered = _gemm(out_bp_gathered, w.t())
        din.index_add_(0, inp_inds, din_gathered)

    return din, dfilters


def indice_maxpool(features: torch.Tensor,
                   indice_pairs: torch.Tensor,
                   indice_pair_num: torch.Tensor,
                   num_activate_out: int) -> torch.Tensor:
    """Max pooling over sparse indices."""
    c = features.shape[1]
    kv = indice_pairs.shape[0]
    device = features.device
    dtype = features.dtype

    out_features = torch.full((num_activate_out, c), float('-inf'),
                              dtype=dtype, device=device)

    for i in range(kv):
        n_act = int(indice_pair_num[i].item())
        if n_act == 0:
            continue
        inp_inds = indice_pairs[i, 0, :n_act].long()
        out_inds = indice_pairs[i, 1, :n_act].long()
        inp_gathered = features[inp_inds]
        out_features[out_inds] = torch.maximum(out_features[out_inds], inp_gathered)

    # Replace -inf with 0 for positions with no input
    out_features = torch.where(out_features == float('-inf'),
                               torch.zeros_like(out_features), out_features)
    return out_features


def indice_maxpool_backward(features: torch.Tensor,
                            out_features: torch.Tensor,
                            out_bp: torch.Tensor,
                            indice_pairs: torch.Tensor,
                            indice_pair_num: torch.Tensor) -> torch.Tensor:
    """Backward for sparse max pooling."""
    n_in = features.shape[0]
    c = features.shape[1]
    kv = indice_pairs.shape[0]
    device = features.device
    dtype = features.dtype

    din = torch.zeros(n_in, c, dtype=dtype, device=device)

    for i in range(kv):
        n_act = int(indice_pair_num[i].item())
        if n_act == 0:
            continue
        inp_inds = indice_pairs[i, 0, :n_act].long()
        out_inds = indice_pairs[i, 1, :n_act].long()

        inp_gathered = features[inp_inds]
        out_gathered = out_features[out_inds]
        mask = (inp_gathered == out_gathered).float()
        din.index_add_(0, inp_inds, mask * out_bp[out_inds])

    return din


def global_pool_rearrange(coords: torch.Tensor, batch_size: int):
    """Rearrange coords by batch for global pooling."""
    batch_indices = coords[:, 0]
    counts = torch.zeros(batch_size, dtype=torch.int64, device=coords.device)
    for b in range(batch_size):
        counts[b] = (batch_indices == b).sum()
    return counts
