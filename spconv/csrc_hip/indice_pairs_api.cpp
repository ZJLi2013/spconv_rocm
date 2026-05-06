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

// ---------------------------------------------------------------------------
// indice_conv forward: C++ for-loop to eliminate Python→C++ overhead per iter
// ---------------------------------------------------------------------------
torch::Tensor indice_conv_forward(
    torch::Tensor features,         // [N_in, C_in]
    torch::Tensor filters,          // [kv, C_in, C_out]
    torch::Tensor indice_pairs,     // [kv, 2, N_max]
    torch::Tensor indice_pair_num,  // [kv]
    int64_t num_activate_out,
    bool subm)
{
    int kv = filters.size(0);
    int c_out = filters.size(2);

    auto out_features = torch::zeros({num_activate_out, c_out},
        features.options());

    // 1 sync instead of kv syncs: copy pair counts to CPU
    auto pn_cpu = indice_pair_num.to(torch::kCPU, torch::kInt32);
    auto* pn = pn_cpu.data_ptr<int>();

    for (int i = 0; i < kv; ++i) {
        int nhot = pn[i];
        if (nhot == 0) continue;

        auto inp_inds = indice_pairs.select(0, i).select(0, 0).slice(0, 0, nhot).to(torch::kLong);
        auto out_inds = indice_pairs.select(0, i).select(0, 1).slice(0, 0, nhot).to(torch::kLong);

        auto inp_gathered = features.index_select(0, inp_inds);  // [nhot, C_in]
        auto w = filters.select(0, i);                           // [C_in, C_out]
        auto result = at::mm(inp_gathered, w);                   // [nhot, C_out]

        if (result.dtype() != out_features.dtype()) {
            result = result.to(out_features.dtype());
        }
        out_features.index_add_(0, out_inds, result);
    }

    return out_features;
}

// ---------------------------------------------------------------------------
// indice_conv backward: C++ for-loop
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> indice_conv_backward(
    torch::Tensor features,         // [N_in, C_in]
    torch::Tensor filters,          // [kv, C_in, C_out]
    torch::Tensor out_bp,           // [N_out, C_out]
    torch::Tensor indice_pairs,     // [kv, 2, N_max]
    torch::Tensor indice_pair_num,  // [kv]
    bool subm)
{
    int kv = filters.size(0);
    int c_in = features.size(1);
    int n_in = features.size(0);

    auto din = torch::zeros({n_in, c_in}, features.options());
    auto dfilters = torch::zeros_like(filters);

    auto pn_cpu = indice_pair_num.to(torch::kCPU, torch::kInt32);
    auto* pn = pn_cpu.data_ptr<int>();

    for (int i = 0; i < kv; ++i) {
        int nhot = pn[i];
        if (nhot == 0) continue;

        auto inp_inds = indice_pairs.select(0, i).select(0, 0).slice(0, 0, nhot).to(torch::kLong);
        auto out_inds = indice_pairs.select(0, i).select(0, 1).slice(0, 0, nhot).to(torch::kLong);

        auto inp_gathered = features.index_select(0, inp_inds);    // [nhot, C_in]
        auto out_bp_gathered = out_bp.index_select(0, out_inds);    // [nhot, C_out]
        auto w = filters.select(0, i);                              // [C_in, C_out]

        // dfilters[i] = inp_gathered^T @ out_bp_gathered
        dfilters.select(0, i) = at::mm(inp_gathered.t(), out_bp_gathered);

        // din: scatter grad back
        auto din_gathered = at::mm(out_bp_gathered, w.t());
        din.index_add_(0, inp_inds, din_gathered);
    }

    return {din, dfilters};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_indice_pairs_subm", &get_indice_pairs_subm,
          "SubM conv indice pairs (HIP kernel)");
    m.def("get_indice_pairs_conv", &get_indice_pairs_conv,
          "Regular/transposed conv indice pairs (HIP kernel)");
    m.def("indice_conv_forward", &indice_conv_forward,
          "Sparse conv forward with C++ for-loop (eliminates Python overhead)");
    m.def("indice_conv_backward", &indice_conv_backward,
          "Sparse conv backward with C++ for-loop");
}
