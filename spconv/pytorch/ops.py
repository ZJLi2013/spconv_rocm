# spconv-rocm: Pure Python ops (gather-GEMM-scatter via FlyDSLGemmTuner)
# Replaces C++ ConvGemmOps + cumm CUDA kernels

import torch
import numpy as np
from typing import List, Optional, Tuple
from spconv.core import ConvAlgo
from spconv.constants import ALL_WEIGHT_IS_KRSC, AllocKeys
import functools

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


###############################################################################
# GPU indice pairs via PyTorch native ops (sort + searchsorted)
# Mirrors original CUDA algorithm without custom kernels.
###############################################################################

def _coord_to_key(coords: torch.Tensor, spatial_shape: List[int],
                  batch_size: int) -> torch.Tensor:
    """Encode (batch, z, y, x, ...) into a single int64 scalar key.

    Mimics the original LinearLayout / LayoutNPQ:
        key = batch * prod(spatial_shape) + z * (Y*X) + y * X + x

    Args:
        coords: [N, ndim+1] int tensor (batch, spatial_dims...)
        spatial_shape: [ndim] spatial extents
    Returns:
        keys: [N] int64 tensor
    """
    ndim = len(spatial_shape)
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * spatial_shape[d + 1]
    vol = strides[0] * spatial_shape[0]

    keys = coords[:, 0].to(torch.int64) * vol
    for d in range(ndim):
        keys = keys + coords[:, d + 1].to(torch.int64) * strides[d]
    return keys


def _build_hash_table(keys: torch.Tensor):
    """Build a sort-based lookup table from keys.

    Returns:
        sorted_keys: sorted unique keys
        sorted_vals: original indices corresponding to sorted_keys
    """
    sorted_keys, sorted_indices = torch.sort(keys)
    return sorted_keys, sorted_indices


def _hash_lookup(sorted_keys: torch.Tensor, sorted_vals: torch.Tensor,
                 query_keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lookup query_keys in a sorted hash table.

    Returns:
        found_vals: values for found keys (index into original array), -1 if not found
        found_mask: bool tensor, True where key was found
    """
    pos = torch.searchsorted(sorted_keys, query_keys)
    pos = pos.clamp(max=sorted_keys.shape[0] - 1)
    found_mask = sorted_keys[pos] == query_keys
    found_vals = torch.where(found_mask, sorted_vals[pos],
                             torch.tensor(-1, dtype=sorted_vals.dtype,
                                          device=sorted_vals.device))
    return found_vals, found_mask


def _get_kernel_offsets(ksize: List[int]) -> torch.Tensor:
    """Generate kernel offset tuples as a tensor.

    Returns: [kv, ndim] int tensor with all kernel offsets.
    """
    ranges = [torch.arange(k) for k in ksize]
    grids = torch.meshgrid(*ranges, indexing='ij')
    return torch.stack([g.reshape(-1) for g in grids], dim=1)  # [kv, ndim]


def _get_indice_pairs_gpu(indices: torch.Tensor,
                          batch_size: int,
                          spatial_shape: List[int],
                          ksize: List[int],
                          stride: List[int],
                          padding: List[int],
                          dilation: List[int],
                          out_padding: List[int],
                          subm: bool,
                          transposed: bool):
    """GPU implementation of indice pairs using sort + searchsorted.

    Mirrors the original two-stage CUDA algorithm but uses only PyTorch ops.
    """
    ndim = len(spatial_shape)
    kv = int(np.prod(ksize))
    device = indices.device
    N = indices.shape[0]

    offsets = _get_kernel_offsets(ksize).to(device)  # [kv, ndim]

    if subm:
        return _get_indice_pairs_subm_gpu(
            indices, batch_size, spatial_shape, ksize, dilation,
            offsets, device)
    else:
        return _get_indice_pairs_conv_gpu(
            indices, batch_size, spatial_shape, ksize, stride,
            padding, dilation, out_padding, offsets, transposed, device)


def _get_indice_pairs_subm_gpu(indices, batch_size, spatial_shape, ksize,
                                dilation, offsets, device):
    """SubM conv: input == output, build hash on input, query neighbors."""
    ndim = len(spatial_shape)
    kv = offsets.shape[0]
    N = indices.shape[0]

    in_keys = _coord_to_key(indices, spatial_shape, batch_size)
    sorted_keys, sorted_vals = _build_hash_table(in_keys)

    center = kv // 2
    half_ksize = torch.tensor([k // 2 for k in ksize], device=device)

    indice_pairs = torch.full((kv, 2, N), -1, dtype=torch.int32, device=device)
    indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)

    # Center kernel position: identity mapping
    arange_n = torch.arange(N, dtype=torch.int32, device=device)
    indice_pairs[center, 0, :N] = arange_n
    indice_pairs[center, 1, :N] = arange_n
    indice_pair_num[center] = N

    in_spatial = indices[:, 1:]  # [N, ndim]

    for kid in range(kv):
        if kid == center:
            continue
        offset = offsets[kid]  # [ndim]
        neighbor_spatial = in_spatial + (offset - half_ksize) * torch.tensor(
            dilation, device=device)

        valid_bounds = (neighbor_spatial >= 0).all(dim=1)
        for d in range(ndim):
            valid_bounds = valid_bounds & (neighbor_spatial[:, d] < spatial_shape[d])

        if not valid_bounds.any():
            continue

        valid_idx = valid_bounds.nonzero(as_tuple=True)[0]
        neighbor_coords = torch.cat([
            indices[valid_idx, 0:1],
            neighbor_spatial[valid_idx]
        ], dim=1)

        query_keys = _coord_to_key(neighbor_coords, spatial_shape, batch_size)
        found_vals, found_mask = _hash_lookup(sorted_keys, sorted_vals, query_keys)

        actual_found = found_mask.nonzero(as_tuple=True)[0]
        n_found = actual_found.shape[0]
        if n_found == 0:
            continue

        out_indices = valid_idx[actual_found]  # output point indices
        in_indices = found_vals[actual_found]  # input point indices

        indice_pairs[kid, 0, :n_found] = in_indices.int()
        indice_pairs[kid, 1, :n_found] = out_indices.int()
        indice_pair_num[kid] = n_found

        # Exploit symmetry: mirror kernel position
        mirror_kid = kv - 1 - kid
        if mirror_kid != kid and mirror_kid != center:
            indice_pairs[mirror_kid, 0, :n_found] = out_indices.int()
            indice_pairs[mirror_kid, 1, :n_found] = in_indices.int()
            indice_pair_num[mirror_kid] = n_found

    return indices, indice_pairs, indice_pair_num


def _get_indice_pairs_conv_gpu(indices, batch_size, spatial_shape, ksize,
                                stride, padding, dilation, out_padding,
                                offsets, transposed, device):
    """Regular/transposed conv: 2-stage approach.

    Stage 1: for each input × offset → compute output coord, collect unique outputs.
    Stage 2: build hash on unique outputs, lookup to fill indice_pairs[1].
    """
    ndim = len(spatial_shape)
    kv = offsets.shape[0]
    N = indices.shape[0]

    if transposed:
        out_spatial_shape = get_deconv_output_size(spatial_shape, ksize, stride,
                                                    padding, dilation, out_padding)
    else:
        out_spatial_shape = get_conv_output_size(spatial_shape, ksize, stride,
                                                  padding, dilation)

    padding_t = torch.tensor(padding, device=device)
    stride_t = torch.tensor(stride, device=device)
    dilation_t = torch.tensor(dilation, device=device)
    out_spatial_t = torch.tensor(out_spatial_shape, device=device)

    in_spatial = indices[:, 1:]  # [N, ndim]
    batch_ids = indices[:, 0]  # [N]

    all_in_idx = []
    all_out_keys = []
    all_kid = []

    for kid in range(kv):
        offset = offsets[kid]  # [ndim]
        if transposed:
            out_sp = (in_spatial + padding_t - offset * dilation_t) * stride_t + offset * dilation_t
        else:
            out_sp = (in_spatial + padding_t - offset * dilation_t).div(
                stride_t, rounding_mode='floor')
            remainder = (in_spatial + padding_t - offset * dilation_t) % stride_t
            stride_valid = (remainder == 0).all(dim=1)

        valid = torch.ones(N, dtype=torch.bool, device=device)
        if not transposed:
            valid = valid & stride_valid
        for d in range(ndim):
            valid = valid & (out_sp[:, d] >= 0) & (out_sp[:, d] < out_spatial_t[d])

        if not valid.any():
            continue

        valid_idx = valid.nonzero(as_tuple=True)[0]
        out_coords = torch.cat([
            batch_ids[valid_idx].unsqueeze(1),
            out_sp[valid_idx]
        ], dim=1).int()

        out_keys = _coord_to_key(out_coords, out_spatial_shape, batch_size)

        all_in_idx.append(valid_idx)
        all_out_keys.append(out_keys)
        all_kid.append(torch.full((valid_idx.shape[0],), kid,
                                  dtype=torch.int32, device=device))

    if len(all_in_idx) == 0:
        out_inds = torch.zeros((0, ndim + 1), dtype=indices.dtype, device=device)
        indice_pairs = torch.full((kv, 2, 1), -1, dtype=torch.int32, device=device)
        indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)
        return out_inds, indice_pairs, indice_pair_num

    cat_in_idx = torch.cat(all_in_idx)
    cat_out_keys = torch.cat(all_out_keys)
    cat_kid = torch.cat(all_kid)

    # Stage 1.5: unique output coords
    unique_out_keys, inverse = torch.unique(cat_out_keys, return_inverse=True)
    num_out = unique_out_keys.shape[0]

    # Decode unique keys back to coordinates
    out_inds = torch.zeros((num_out, ndim + 1), dtype=indices.dtype, device=device)
    vol_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        vol_strides[d] = vol_strides[d + 1] * out_spatial_shape[d + 1]
    vol = vol_strides[0] * out_spatial_shape[0]

    remainder = unique_out_keys.clone()
    out_inds[:, 0] = (remainder // vol).to(indices.dtype)
    remainder = remainder % vol
    for d in range(ndim):
        out_inds[:, d + 1] = (remainder // vol_strides[d]).to(indices.dtype)
        remainder = remainder % vol_strides[d]

    # Stage 2: fill indice_pairs using inverse mapping for output indices
    out_idx_mapped = inverse.int()  # maps each pair to unique output index
    N_max = max(N, num_out)
    indice_pairs = torch.full((kv, 2, N_max), -1, dtype=torch.int32, device=device)
    indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)

    for kid in range(kv):
        mask = cat_kid == kid
        if not mask.any():
            continue
        kid_in_idx = cat_in_idx[mask].int()
        kid_out_idx = out_idx_mapped[mask]
        n_pairs = kid_in_idx.shape[0]
        indice_pairs[kid, 0, :n_pairs] = kid_in_idx
        indice_pairs[kid, 1, :n_pairs] = kid_out_idx
        indice_pair_num[kid] = n_pairs

    return out_inds, indice_pairs, indice_pair_num


###############################################################################
# Public API
###############################################################################

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
    """Compute indice pairs for sparse convolution.

    Uses GPU-accelerated sort+searchsorted when on CUDA/ROCm device,
    falls back to CPU dict implementation otherwise.

    Returns:
        out_inds: output indices [N_out, ndim+1]
        indice_pairs: [kv, 2, N_max] — gather/scatter index pairs
        indice_pair_num: [kv] — number of active pairs per kernel position
    """
    if indices.is_cuda:
        return _get_indice_pairs_gpu(
            indices, batch_size, spatial_shape, ksize, stride,
            padding, dilation, out_padding, subm, transposed)

    # CPU fallback (original Python dict implementation)
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

        # scatter-add to output (cast if AMP autocast changed dtype)
        if gemm_out.dtype != out_features.dtype:
            gemm_out = gemm_out.to(out_features.dtype)
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
