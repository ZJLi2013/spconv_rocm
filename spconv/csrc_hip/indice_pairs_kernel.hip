#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "hash_table.h"

namespace spconv_hip {

// Max supported spatial dimensions (1D, 2D, 3D, 4D).
constexpr int kMaxNDim = 4;

// Convolution problem descriptor — replaces tensorview's ConvProblem.
struct ConvProblem {
    int batch_size;
    int ndim;
    int input_dims[kMaxNDim];
    int output_dims[kMaxNDim];
    int ksize[kMaxNDim];
    int padding[kMaxNDim];
    int stride[kMaxNDim];
    int dilation[kMaxNDim];
};

// Compute linear layout offset from (batch, spatial...) coordinates.
// Equivalent to TensorGeneric layout_npq: key = batch * vol + z*Y*X + y*X + x
template <typename T>
__host__ __device__ __forceinline__
T coord_to_linear(const int* coord, const int* dims, int ndim) {
    T key = (T)coord[0];  // batch
    for (int d = 0; d < ndim; d++) {
        key = key * (T)dims[d] + (T)coord[d + 1];
    }
    return key;
}

// Inverse: linear offset → (batch, spatial...) coordinates.
template <typename T>
__host__ __device__ __forceinline__
void linear_to_coord(T key, int* coord, const int* dims, int ndim) {
    for (int d = ndim - 1; d >= 0; d--) {
        coord[d + 1] = (int)(key % (T)dims[d]);
        key /= (T)dims[d];
    }
    coord[0] = (int)key;  // batch
}

// Decompose linear filter offset into per-dimension filter indices.
__host__ __device__ __forceinline__
void filter_offset_to_ndim(int filter_offset, int* r, const int* ksize, int ndim) {
    for (int d = ndim - 1; d >= 0; d--) {
        r[d] = filter_offset % ksize[d];
        filter_offset /= ksize[d];
    }
}

// ============================================================================
// SubM convolution kernels
// ============================================================================

// Build hash table: insert all input coordinates.
template <typename K>
__global__ void build_subm_hash_table_kernel(
    LinearHashTable<K, int> table,
    const int* __restrict__ indices_in,  // [N, ndim+1]
    const int* __restrict__ output_dims,
    int ndim,
    int num_indices)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_indices;
         i += gridDim.x * blockDim.x) {
        K key = coord_to_linear<K>(indices_in + i * (ndim + 1), output_dims, ndim);
        table.insert(key, i);
    }
}

// Compute SubM indice pairs. Uses 2D grid: blockIdx.y = filter_offset.
// Only processes half the kernel positions (+ center); symmetry fills the other half.
template <typename K>
__global__ void calc_subm_conv_indices_kernel(
    LinearHashTable<K, int> table,
    const int* __restrict__ indices_in,  // [N, ndim+1]
    int* __restrict__ indice_pairs,      // [2, kv, N]
    int* __restrict__ indice_num_per_loc, // [kv]
    const int* __restrict__ input_dims,
    const int* __restrict__ ksize_d,
    const int* __restrict__ dilation_d,
    int ndim,
    int num_indices,
    int indices_pair_size,
    int kv)
{
    int filter_offset = blockIdx.y;
    int filter_offset_mul_ips = filter_offset * indices_pair_size;
    int mirror_offset = kv - 1 - filter_offset;
    int mirror_offset_mul_ips = mirror_offset * indices_pair_size;
    int ips_mul_kv = indices_pair_size * kv;

    if (filter_offset == kv / 2) {
        // Center kernel position: identity mapping.
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < num_indices;
             i += gridDim.x * blockDim.x) {
            indice_pairs[filter_offset_mul_ips + i] = i;
            indice_pairs[ips_mul_kv + filter_offset_mul_ips + i] = i;
        }
    } else {
        // Compute filter indices for this offset.
        int r[kMaxNDim];
        filter_offset_to_ndim(filter_offset, r, ksize_d, ndim);

        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < num_indices;
             i += gridDim.x * blockDim.x) {
            const int* cur_idx = indices_in + i * (ndim + 1);

            // Compute neighbor coordinate: input_coord + (r - ksize/2) * dilation
            int neighbor[kMaxNDim + 1];
            neighbor[0] = cur_idx[0];  // batch
            bool valid = true;
            for (int d = 0; d < ndim; d++) {
                int offset_d = r[d] - ksize_d[d] / 2;
                neighbor[d + 1] = cur_idx[d + 1] + offset_d * dilation_d[d];
                if (neighbor[d + 1] < 0 || neighbor[d + 1] >= input_dims[d]) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                K key = coord_to_linear<K>(neighbor, input_dims, ndim);
                int slot = table.lookup_offset(key);
                if (slot != -1) {
                    int v = table.values[slot];
                    int old_num = atomicAdd(indice_num_per_loc + filter_offset, 1);
                    // Forward: indice_pairs[0, filter_offset, old_num] = i (output)
                    //          indice_pairs[1, filter_offset, old_num] = v (input)
                    indice_pairs[filter_offset_mul_ips + old_num] = i;
                    indice_pairs[ips_mul_kv + filter_offset_mul_ips + old_num] = v;
                    // Mirror: indice_pairs[0, mirror, old_num] = v
                    //         indice_pairs[1, mirror, old_num] = i
                    indice_pairs[mirror_offset_mul_ips + old_num] = v;
                    indice_pairs[ips_mul_kv + mirror_offset_mul_ips + old_num] = i;
                }
            }
        }
    }
}

// ============================================================================
// Regular (strided / transposed) convolution kernels
// ============================================================================

// Stage 1: For each input point × filter offset, compute output coord offset.
// Store input index in indice_pairs[0], output coord offset in indice_pairs_uniq.
template <typename K>
__global__ void calc_conv_indices_stage1_kernel(
    const int* __restrict__ indices_in,  // [N, ndim+1]
    int* __restrict__ indice_pairs_in,   // [kv, N] — stores input indices
    K* __restrict__ indice_pairs_uniq,   // [kv * N + 1] — stores output coord offsets
    int* __restrict__ indice_num_per_loc, // [kv]
    const int* __restrict__ output_dims,
    const int* __restrict__ ksize_d,
    const int* __restrict__ stride_d,
    const int* __restrict__ padding_d,
    const int* __restrict__ dilation_d,
    int ndim,
    int num_indices_in,
    int indices_pair_size,
    int kv,
    bool transposed)
{
    int filter_offset = blockIdx.y;
    int fo_mul_ips = filter_offset * indices_pair_size;

    int r[kMaxNDim];
    filter_offset_to_ndim(filter_offset, r, ksize_d, ndim);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_indices_in;
         i += gridDim.x * blockDim.x) {
        const int* cur = indices_in + i * (ndim + 1);

        int out_coord[kMaxNDim + 1];
        out_coord[0] = cur[0];  // batch
        bool valid = (out_coord[0] >= 0);

        if (transposed) {
            for (int d = 0; d < ndim; d++) {
                out_coord[d + 1] = cur[d + 1] * stride_d[d] - padding_d[d]
                                   + r[d] * dilation_d[d];
                if (out_coord[d + 1] < 0 || out_coord[d + 1] >= output_dims[d]) {
                    valid = false;
                    break;
                }
            }
        } else {
            for (int d = 0; d < ndim; d++) {
                int tmp = cur[d + 1] + padding_d[d] - r[d] * dilation_d[d];
                if (tmp % stride_d[d] != 0) { valid = false; break; }
                out_coord[d + 1] = tmp / stride_d[d];
                if (out_coord[d + 1] < 0 || out_coord[d + 1] >= output_dims[d]) {
                    valid = false;
                    break;
                }
            }
        }

        if (valid) {
            int old_num = atomicAdd(indice_num_per_loc + filter_offset, 1);
            K offset = coord_to_linear<K>(out_coord, output_dims, ndim);
            if (old_num < indices_pair_size) {
                indice_pairs_in[fo_mul_ips + old_num] = i;
                indice_pairs_uniq[fo_mul_ips + old_num] = offset;
            }
        }
    }
}

// Initialize uniq buffer with sentinel values.
template <typename K>
__global__ void clean_indices_uniq_kernel(K* data, int64_t size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += (int64_t)gridDim.x * blockDim.x) {
        data[i] = std::numeric_limits<K>::max();
    }
}

// Build hash table from unique output coord offsets, assign sequential indices.
// Also writes output coordinates by inverting the linear offset.
template <typename K>
__global__ void build_conv_hash_and_assign_kernel(
    LinearHashTable<K, int> table,
    int* __restrict__ out_indices,  // [num_out, ndim+1]
    const K* __restrict__ unique_offsets,
    const int* __restrict__ output_dims,
    int ndim,
    int num_out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_out;
         i += gridDim.x * blockDim.x) {
        K offset = unique_offsets[i];
        table.insert(offset, i);
        linear_to_coord<K>(offset, out_indices + i * (ndim + 1), output_dims, ndim);
    }
}

// Stage 2: Use hash table to map output coord offsets → sequential output indices.
template <typename K>
__global__ void calc_conv_indices_stage2_kernel(
    LinearHashTable<K, int> table,
    const K* __restrict__ indice_pairs_uniq_before_sort,  // [kv, N]
    int* __restrict__ indice_pairs_out,  // [kv, N]
    int num_indices_in,
    int indices_pair_size)
{
    int filter_offset = blockIdx.y;
    int fo_mul_ips = filter_offset * indices_pair_size;
    const K* uniq_filter = indice_pairs_uniq_before_sort + fo_mul_ips;
    int* out_filter = indice_pairs_out + fo_mul_ips;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_indices_in;
         i += gridDim.x * blockDim.x) {
        K coord_offset = uniq_filter[i];
        if (coord_offset != std::numeric_limits<K>::max()) {
            int slot = table.lookup_offset(coord_offset);
            if (slot != -1) {
                out_filter[i] = table.values[slot];
            }
        }
    }
}

// ============================================================================
// Host-side orchestration (called from PyTorch C++ extension)
// ============================================================================

// SubM convolution: build hash table + compute indice pairs in 2 kernel launches.
template <typename K>
void generate_subm_conv_inds(
    const int* indices_in,   // [N, ndim+1] on device
    int* indice_pairs,       // [2, kv, N] on device
    int* indice_num_per_loc, // [kv] on device
    int num_act_in,
    int batch_size,
    const int* input_dims_h,  // host
    const int* ksize_h,       // host
    const int* dilation_h,    // host
    int ndim,
    hipStream_t stream)
{
    int kv = 1;
    for (int d = 0; d < ndim; d++) kv *= ksize_h[d];

    int hash_size = num_act_in * 2;
    if (hash_size < 128) hash_size = 128;

    // Allocate hash table on device via hipMalloc (temporary).
    K* hash_keys;
    int* hash_vals;
    hipMalloc(&hash_keys, hash_size * sizeof(K));
    hipMalloc(&hash_vals, hash_size * sizeof(int));

    // Allocate device copies of dimension arrays.
    int* d_input_dims;
    int* d_ksize;
    int* d_dilation;
    hipMalloc(&d_input_dims, ndim * sizeof(int));
    hipMalloc(&d_ksize, ndim * sizeof(int));
    hipMalloc(&d_dilation, ndim * sizeof(int));
    hipMemcpyAsync(d_input_dims, input_dims_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_ksize, ksize_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_dilation, dilation_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);

    LinearHashTable<K, int> table(hash_keys, hash_vals, hash_size);

    // Clear hash table.
    int threads = 256;
    int blocks_clear = (hash_size + threads - 1) / threads;
    hipLaunchKernelGGL(clear_hash_table_kernel<K>, dim3(blocks_clear), dim3(threads), 0, stream,
                       hash_keys, hash_size);

    // Build hash table from input coordinates.
    int blocks_build = (num_act_in + threads - 1) / threads;
    hipLaunchKernelGGL(build_subm_hash_table_kernel<K>, dim3(blocks_build), dim3(threads), 0, stream,
                       table, indices_in, d_input_dims, ndim, num_act_in);

    // Compute indice pairs: 2D grid with blockIdx.y = filter_offset.
    // Only process half the kernel + center (kv/2 + 1 slices), mirror fills the rest.
    int blocks_x = (num_act_in + threads - 1) / threads;
    dim3 grid(blocks_x, kv / 2 + 1);
    hipLaunchKernelGGL(calc_subm_conv_indices_kernel<K>, grid, dim3(threads), 0, stream,
                       table, indices_in, indice_pairs, indice_num_per_loc,
                       d_input_dims, d_ksize, d_dilation,
                       ndim, num_act_in, num_act_in, kv);

    hipStreamSynchronize(stream);
    hipFree(hash_keys);
    hipFree(hash_vals);
    hipFree(d_input_dims);
    hipFree(d_ksize);
    hipFree(d_dilation);
}

// Regular convolution: 3-stage pipeline.
// Stage 1: compute output coords per input × filter_offset.
// Stage 1.5: sort + unique to get unique output points.
// Stage 2: hash table lookup to assign sequential output indices.
// Returns: number of unique output points.
template <typename K>
int generate_conv_inds(
    const int* indices_in,   // [N, ndim+1] on device
    int* indice_pairs,       // [2, kv, N] on device — first half is input indices
    int* indice_num_per_loc, // [kv] on device
    int* out_indices,        // [max_out, ndim+1] on device
    int num_act_in,
    int batch_size,
    const int* input_dims_h,
    const int* output_dims_h,
    const int* ksize_h,
    const int* stride_h,
    const int* padding_h,
    const int* dilation_h,
    int ndim,
    bool transposed,
    hipStream_t stream)
{
    int kv = 1;
    for (int d = 0; d < ndim; d++) kv *= ksize_h[d];

    int64_t uniq_size = (int64_t)num_act_in * kv + 1;

    // Allocate temporaries.
    K* indice_pairs_uniq;        // for stage 1 output coord offsets
    K* indice_pairs_uniq_copy;   // copy before sort (stage 2 needs original order)
    hipMalloc(&indice_pairs_uniq, uniq_size * sizeof(K));
    hipMalloc(&indice_pairs_uniq_copy, uniq_size * sizeof(K));

    int* d_output_dims;
    int* d_input_dims;
    int* d_ksize;
    int* d_stride;
    int* d_padding;
    int* d_dilation;
    hipMalloc(&d_output_dims, ndim * sizeof(int));
    hipMalloc(&d_input_dims, ndim * sizeof(int));
    hipMalloc(&d_ksize, ndim * sizeof(int));
    hipMalloc(&d_stride, ndim * sizeof(int));
    hipMalloc(&d_padding, ndim * sizeof(int));
    hipMalloc(&d_dilation, ndim * sizeof(int));
    hipMemcpyAsync(d_output_dims, output_dims_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_input_dims, input_dims_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_ksize, ksize_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_stride, stride_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_padding, padding_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_dilation, dilation_h, ndim * sizeof(int), hipMemcpyHostToDevice, stream);

    int threads = 256;

    // Clean uniq buffer.
    int blocks_clean = ((int)uniq_size + threads - 1) / threads;
    hipLaunchKernelGGL(clean_indices_uniq_kernel<K>, dim3(blocks_clean), dim3(threads), 0, stream,
                       indice_pairs_uniq, uniq_size);

    // Stage 1: compute output coord offsets.
    int blocks_x = (num_act_in + threads - 1) / threads;
    dim3 grid_s1(blocks_x, kv);
    hipLaunchKernelGGL(calc_conv_indices_stage1_kernel<K>, grid_s1, dim3(threads), 0, stream,
                       indices_in, indice_pairs,  // indice_pairs[0] part
                       indice_pairs_uniq, indice_num_per_loc,
                       d_output_dims, d_ksize, d_stride, d_padding, d_dilation,
                       ndim, num_act_in, num_act_in, kv, transposed);

    // Copy uniq before sort (stage 2 needs original order for each filter position).
    hipMemcpyAsync(indice_pairs_uniq_copy, indice_pairs_uniq,
                   uniq_size * sizeof(K), hipMemcpyDeviceToDevice, stream);

    // Stage 1.5: sort + unique via hipCUB.
    size_t sort_temp_bytes = 0;
    hipcub::DeviceRadixSort::SortKeys(nullptr, sort_temp_bytes,
                                       indice_pairs_uniq, indice_pairs_uniq,
                                       (int)uniq_size, 0, sizeof(K) * 8, stream);
    void* sort_temp;
    hipMalloc(&sort_temp, sort_temp_bytes);

    // Sort in-place (double-buffer: input=output uses temp internally).
    K* sorted_keys;
    hipMalloc(&sorted_keys, uniq_size * sizeof(K));
    hipcub::DeviceRadixSort::SortKeys(sort_temp, sort_temp_bytes,
                                       indice_pairs_uniq, sorted_keys,
                                       (int)uniq_size, 0, sizeof(K) * 8, stream);

    // Unique.
    int* d_num_selected;
    hipMalloc(&d_num_selected, sizeof(int));
    size_t unique_temp_bytes = 0;
    K* unique_out;
    hipMalloc(&unique_out, uniq_size * sizeof(K));
    hipcub::DeviceSelect::Unique(nullptr, unique_temp_bytes,
                                  sorted_keys, unique_out,
                                  d_num_selected, (int)uniq_size, stream);
    void* unique_temp;
    hipMalloc(&unique_temp, unique_temp_bytes);
    hipcub::DeviceSelect::Unique(unique_temp, unique_temp_bytes,
                                  sorted_keys, unique_out,
                                  d_num_selected, (int)uniq_size, stream);

    int num_out_act;
    hipMemcpyAsync(&num_out_act, d_num_selected, sizeof(int), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    // The last unique element is the sentinel (max value), subtract 1.
    num_out_act -= 1;

    // Stage 2: build hash table from unique output offsets, then lookup.
    int hash_size = num_out_act * 2;
    if (hash_size < 128) hash_size = 128;
    K* hash_keys;
    int* hash_vals;
    hipMalloc(&hash_keys, hash_size * sizeof(K));
    hipMalloc(&hash_vals, hash_size * sizeof(int));

    LinearHashTable<K, int> table(hash_keys, hash_vals, hash_size);

    int blocks_ht = (hash_size + threads - 1) / threads;
    hipLaunchKernelGGL(clear_hash_table_kernel<K>, dim3(blocks_ht), dim3(threads), 0, stream,
                       hash_keys, hash_size);

    int blocks_out = (num_out_act + threads - 1) / threads;
    hipLaunchKernelGGL(build_conv_hash_and_assign_kernel<K>, dim3(blocks_out), dim3(threads), 0, stream,
                       table, out_indices, unique_out, d_output_dims, ndim, num_out_act);

    // Stage 2 lookup: map each (filter, input) pair's output coord to sequential index.
    dim3 grid_s2(blocks_x, kv);
    hipLaunchKernelGGL(calc_conv_indices_stage2_kernel<K>, grid_s2, dim3(threads), 0, stream,
                       table, indice_pairs_uniq_copy,
                       indice_pairs + kv * num_act_in,  // indice_pairs[1] part
                       num_act_in, num_act_in);

    // Cleanup.
    hipFree(indice_pairs_uniq);
    hipFree(indice_pairs_uniq_copy);
    hipFree(sorted_keys);
    hipFree(sort_temp);
    hipFree(unique_out);
    hipFree(unique_temp);
    hipFree(d_num_selected);
    hipFree(hash_keys);
    hipFree(hash_vals);
    hipFree(d_output_dims);
    hipFree(d_input_dims);
    hipFree(d_ksize);
    hipFree(d_stride);
    hipFree(d_padding);
    hipFree(d_dilation);

    return num_out_act;
}

// Explicit instantiations for int32 and int64 key types.
template void generate_subm_conv_inds<int32_t>(
    const int*, int*, int*, int, int, const int*, const int*, const int*, int, hipStream_t);
template void generate_subm_conv_inds<int64_t>(
    const int*, int*, int*, int, int, const int*, const int*, const int*, int, hipStream_t);

template int generate_conv_inds<int32_t>(
    const int*, int*, int*, int*, int, int,
    const int*, const int*, const int*, const int*, const int*, const int*,
    int, bool, hipStream_t);
template int generate_conv_inds<int64_t>(
    const int*, int*, int*, int*, int, int,
    const int*, const int*, const int*, const int*, const int*, const int*,
    int, bool, hipStream_t);

}  // namespace spconv_hip
