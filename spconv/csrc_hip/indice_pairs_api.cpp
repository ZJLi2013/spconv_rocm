#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

namespace spconv_hip {

template <typename K>
void generate_subm_conv_inds(
    const int* indices_in, int* indice_pairs, int* indice_num_per_loc,
    int num_act_in, int batch_size,
    const int* input_dims_h, const int* ksize_h, const int* dilation_h,
    int ndim, hipStream_t stream);

template <typename K>
int generate_conv_inds(
    const int* indices_in, int* indice_pairs, int* indice_num_per_loc,
    int* out_indices, int num_act_in, int batch_size,
    const int* input_dims_h, const int* output_dims_h,
    const int* ksize_h, const int* stride_h, const int* padding_h,
    const int* dilation_h, int ndim, bool transposed, hipStream_t stream);

}  // namespace spconv_hip

// Check whether the spatial volume fits in int32.
static bool check_use_int32(const std::vector<int>& spatial_shape, int batch_size) {
    int64_t vol = (int64_t)batch_size;
    for (int d : spatial_shape) vol *= d;
    return vol < (int64_t)std::numeric_limits<int32_t>::max();
}

// SubM indice pairs: input == output points.
// Returns: (indice_pairs [2, kv, N], indice_pair_num [kv])
std::vector<torch::Tensor> get_indice_pairs_subm(
    torch::Tensor indices,       // [N, ndim+1] int32 on GPU
    int batch_size,
    std::vector<int> spatial_shape,
    std::vector<int> ksize,
    std::vector<int> dilation)
{
    TORCH_CHECK(indices.is_cuda(), "indices must be on GPU");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");

    int ndim = (int)spatial_shape.size();
    int N = indices.size(0);
    int kv = 1;
    for (int k : ksize) kv *= k;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(indices.device());
    auto indice_pairs = torch::full({2, kv, N}, -1, options);
    auto indice_pair_num = torch::zeros({kv}, options);

    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();

    bool use_i32 = check_use_int32(spatial_shape, batch_size);
    if (use_i32) {
        spconv_hip::generate_subm_conv_inds<int32_t>(
            indices.data_ptr<int>(),
            indice_pairs.data_ptr<int>(),
            indice_pair_num.data_ptr<int>(),
            N, batch_size,
            spatial_shape.data(), ksize.data(), dilation.data(),
            ndim, stream);
    } else {
        spconv_hip::generate_subm_conv_inds<int64_t>(
            indices.data_ptr<int>(),
            indice_pairs.data_ptr<int>(),
            indice_pair_num.data_ptr<int>(),
            N, batch_size,
            spatial_shape.data(), ksize.data(), dilation.data(),
            ndim, stream);
    }

    return {indice_pairs, indice_pair_num};
}

// Regular/transposed conv indice pairs.
// Returns: (out_indices [N_out, ndim+1], indice_pairs [2, kv, N], indice_pair_num [kv])
std::vector<torch::Tensor> get_indice_pairs_conv(
    torch::Tensor indices,       // [N, ndim+1] int32 on GPU
    int batch_size,
    std::vector<int> spatial_shape,   // input spatial shape
    std::vector<int> output_shape,    // output spatial shape
    std::vector<int> ksize,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    bool transposed)
{
    TORCH_CHECK(indices.is_cuda(), "indices must be on GPU");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");

    int ndim = (int)spatial_shape.size();
    int N = indices.size(0);
    int kv = 1;
    for (int k : ksize) kv *= k;

    // Upper bound on output points: N * kv (before dedup).
    int max_out = N * kv;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(indices.device());
    auto indice_pairs = torch::full({2, kv, N}, -1, options);
    auto indice_pair_num = torch::zeros({kv}, options);
    auto out_indices = torch::zeros({max_out, ndim + 1}, options);

    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();

    int num_out_act;
    bool use_i32 = check_use_int32(output_shape, batch_size);
    if (use_i32) {
        num_out_act = spconv_hip::generate_conv_inds<int32_t>(
            indices.data_ptr<int>(),
            indice_pairs.data_ptr<int>(),
            indice_pair_num.data_ptr<int>(),
            out_indices.data_ptr<int>(),
            N, batch_size,
            spatial_shape.data(), output_shape.data(),
            ksize.data(), stride.data(), padding.data(), dilation.data(),
            ndim, transposed, stream);
    } else {
        num_out_act = spconv_hip::generate_conv_inds<int64_t>(
            indices.data_ptr<int>(),
            indice_pairs.data_ptr<int>(),
            indice_pair_num.data_ptr<int>(),
            out_indices.data_ptr<int>(),
            N, batch_size,
            spatial_shape.data(), output_shape.data(),
            ksize.data(), stride.data(), padding.data(), dilation.data(),
            ndim, transposed, stream);
    }

    // Trim output indices to actual count.
    out_indices = out_indices.slice(0, 0, num_out_act).contiguous();

    return {out_indices, indice_pairs, indice_pair_num};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_indice_pairs_subm", &get_indice_pairs_subm,
          "SubM conv indice pairs (HIP kernel)");
    m.def("get_indice_pairs_conv", &get_indice_pairs_conv,
          "Regular/transposed conv indice pairs (HIP kernel)");
}
