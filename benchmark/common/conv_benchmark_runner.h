//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <benchmark/benchmark.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark/common/benchmark_common.h"
#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Global conv benchmark parameters (set by main.cpp CLI parsing)
// ============================================================================

struct ConvShape
{
    size_t ic = 64;
    size_t oc = 128;
    size_t ih = 28;
    size_t iw = 28;
    size_t kh = 3;
    size_t kw = 3;
    size_t stride_h = 1;
    size_t stride_w = 1;
    size_t pad_h = 1;
    size_t pad_w = 1;
    size_t groups = 1;

    size_t oh() const { return (ih + 2 * pad_h - kh) / stride_h + 1; }
    size_t ow() const { return (iw + 2 * pad_w - kw) / stride_w + 1; }
    size_t ic_per_group() const { return ic / groups; }
    size_t oc_per_group() const { return oc / groups; }

    // CF GEMM dimensions: M=OC_per_group, N=OH*OW, K=IC_per_group*KH*KW
    size_t cf_m() const { return oc_per_group(); }
    size_t cf_n() const { return oh() * ow(); }
    size_t cf_k() const { return ic_per_group() * kh * kw; }

    // CL GEMM dimensions: M=OH*OW, N=OC_per_group, K=KH*KW*IC_per_group
    size_t cl_m() const { return oh() * ow(); }
    size_t cl_n() const { return oc_per_group(); }
    size_t cl_k() const { return kh * kw * ic_per_group(); }

    tqt_conv_params to_params() const
    {
        tqt_conv_params p;
        p.input_channel = ic;
        p.output_channel = oc;
        p.input_shape[0] = 1;
        p.input_shape[1] = ih;
        p.input_shape[2] = iw;
        p.output_shape[0] = 1;
        p.output_shape[1] = oh();
        p.output_shape[2] = ow();
        p.kernel_shape[0] = 1;
        p.kernel_shape[1] = kh;
        p.kernel_shape[2] = kw;
        p.stride[0] = 1;
        p.stride[1] = stride_h;
        p.stride[2] = stride_w;
        p.dilation[0] = 1;
        p.dilation[1] = 1;
        p.dilation[2] = 1;
        p.pad[0] = 0;
        p.pad[1] = pad_h;
        p.pad[2] = pad_w;
        p.groups = groups;
        return p;
    }
};

inline ConvShape &global_conv_shape()
{
    static ConvShape shape;
    return shape;
}

/// Conv FLOPS: 2 * OC * OH * OW * IC_per_group * KH * KW
inline double conv_flops(const ConvShape &s)
{
    return 2.0 * static_cast<double>(s.oc) * static_cast<double>(s.oh()) *
           static_cast<double>(s.ow()) * static_cast<double>(s.ic_per_group()) *
           static_cast<double>(s.kh) * static_cast<double>(s.kw);
}

// ============================================================================
// Function pointer types for non-packed conv-CF (channel first)
// ============================================================================

template <typename ClampT>
struct NonPackedConvCfBenchFuncs
{
    size_t (*get_b_im2col_size)(size_t k, size_t n);
    void (*run_im2col)(size_t group_idx, const void *activation, void *im2col_buf,
                       const tqt_conv_params *params);
    void (*run_conv)(size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a,
                     const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
                     size_t ldd, const void *bias, ClampT clamp_min, ClampT clamp_max);
};

// ============================================================================
// Function pointer types for non-packed conv-CL (channel last)
// ============================================================================

template <typename ClampT>
struct NonPackedConvClBenchFuncs
{
    size_t (*get_a_im2col_size)(size_t m, size_t k);
    void (*run_im2col)(size_t group_idx, const void *activation, void *im2col_buf,
                       const tqt_conv_params *params);
    void (*run_conv)(size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a,
                     const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
                     size_t ldd, const void *bias, ClampT clamp_min, ClampT clamp_max);
};

// ============================================================================
// Function pointer types for packed conv-CF (channel first)
// ============================================================================

template <typename ClampT>
struct PackedConvCfBenchFuncs
{
    size_t (*get_a_packed_size)(size_t m, size_t k);
    size_t (*get_b_im2colpack_size)(size_t n, size_t k);
    void (*run_a_pack)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                       const void *A, void *A_packed);
    void (*run_b_im2colpack)(size_t group_idx, const void *activation, size_t ldb_packed,
                             size_t k_idx, void *B_packed, const tqt_conv_params *params);
    void (*run_conv)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
                     const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc,
                     void *D, size_t ldd, const void *bias, ClampT clamp_min, ClampT clamp_max);
};

// ============================================================================
// Function pointer types for packed conv-CL (channel last)
// ============================================================================

template <typename ClampT>
struct PackedConvClBenchFuncs
{
    size_t (*get_a_im2colpack_size)(size_t m, size_t k);
    size_t (*get_b_packed_size)(size_t n, size_t k);
    void (*run_bt_pack)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                        const void *B, void *B_packed);
    void (*run_a_im2colpack)(size_t group_idx, const void *activation, size_t lda_packed,
                             size_t k_idx, void *A_packed, const tqt_conv_params *params);
    void (*run_conv)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
                     const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc,
                     void *D, size_t ldd, const void *bias, ClampT clamp_min, ClampT clamp_max);
};

// ============================================================================
// Non-packed Conv-CF benchmark (im2col + gemm)
// ============================================================================

template <typename T, typename ClampT>
inline void run_nonpacked_convcf_benchmark(::benchmark::State &state,
                                           const NonPackedConvCfBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cf_m();
    const size_t n = cs.cf_n();
    const size_t k = cs.cf_k();

    // Activation: CF layout [IC, IH, IW]
    std::vector<T> activation(cs.ic * cs.ih * cs.iw);
    fill_random<T>(activation.data(), activation.size());

    // Weight: [OC_per_group, K]
    std::vector<T> weight(m * k);
    fill_random<T>(weight.data(), weight.size());

    // im2col buffer: [K, N]
    const size_t im2col_size = funcs.get_b_im2col_size(k, n);
    std::vector<uint8_t> im2col_buf(im2col_size);

    // Output: [OC_per_group, OH*OW]
    std::vector<T> output(m * n, static_cast<T>(0));

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_im2col(0, activation.data(), im2col_buf.data(), &params);
        funcs.run_conv(m, n, k, weight.data(), k, 0, im2col_buf.data(), n, 0, nullptr, n,
                       output.data(), n, nullptr, clamp_min, clamp_max);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Non-packed Conv-CL benchmark (im2col + gemm)
// ============================================================================

template <typename T, typename ClampT>
inline void run_nonpacked_convcl_benchmark(::benchmark::State &state,
                                           const NonPackedConvClBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cl_m();
    const size_t n = cs.cl_n();
    const size_t k = cs.cl_k();

    // Activation: CL layout [IH, IW, IC]
    std::vector<T> activation(cs.ih * cs.iw * cs.ic);
    fill_random<T>(activation.data(), activation.size());

    // Weight: [OC_per_group, K] (transposed layout)
    std::vector<T> weight(n * k);
    fill_random<T>(weight.data(), weight.size());

    // im2col buffer: [M, K]
    const size_t im2col_size = funcs.get_a_im2col_size(m, k);
    std::vector<uint8_t> im2col_buf(im2col_size);

    // Output: [OH*OW, OC_per_group]
    std::vector<T> output(m * n, static_cast<T>(0));

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_im2col(0, activation.data(), im2col_buf.data(), &params);
        funcs.run_conv(m, n, k, im2col_buf.data(), k, 0, weight.data(), k, 0, nullptr, n,
                       output.data(), n, nullptr, clamp_min, clamp_max);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed Conv-CF benchmark (weight pre-packed, b_im2colpack + gemm)
// ============================================================================

template <typename T, typename ClampT>
inline void run_packed_convcf_benchmark(::benchmark::State &state,
                                        const PackedConvCfBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cf_m();
    const size_t n = cs.cf_n();
    const size_t k = cs.cf_k();

    // Activation: CF layout [IC, IH, IW]
    std::vector<T> activation(cs.ic * cs.ih * cs.iw);
    fill_random<T>(activation.data(), activation.size());

    // Weight: [OC_per_group, K]
    std::vector<T> weight(m * k);
    fill_random<T>(weight.data(), weight.size());

    // Pack weight (A) outside the timing loop
    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    std::vector<uint8_t> a_packed(a_packed_size);
    funcs.run_a_pack(m, k, k, k, 0, weight.data(), a_packed.data());

    // B packed buffer (im2colpack writes here)
    const size_t b_packed_size = funcs.get_b_im2colpack_size(n, k);
    std::vector<uint8_t> b_packed(b_packed_size);

    // Output
    std::vector<T> output(m * n, static_cast<T>(0));

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_b_im2colpack(0, activation.data(), k, 0, b_packed.data(), &params);
        funcs.run_conv(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, nullptr, n,
                       output.data(), n, nullptr, clamp_min, clamp_max);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed Conv-CL benchmark (weight pre-packed, a_im2colpack + gemm)
// ============================================================================

template <typename T, typename ClampT>
inline void run_packed_convcl_benchmark(::benchmark::State &state,
                                        const PackedConvClBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cl_m();
    const size_t n = cs.cl_n();
    const size_t k = cs.cl_k();

    // Activation: CL layout [IH, IW, IC]
    std::vector<T> activation(cs.ih * cs.iw * cs.ic);
    fill_random<T>(activation.data(), activation.size());

    // Weight: [OC_per_group, K] (transposed)
    std::vector<T> weight(n * k);
    fill_random<T>(weight.data(), weight.size());

    // Pack weight (B) outside the timing loop
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> b_packed(b_packed_size);
    funcs.run_bt_pack(n, k, k, k, 0, weight.data(), b_packed.data());

    // A packed buffer (a_im2colpack writes here)
    const size_t a_packed_size = funcs.get_a_im2colpack_size(m, k);
    std::vector<uint8_t> a_packed(a_packed_size);

    // Output
    std::vector<T> output(m * n, static_cast<T>(0));

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_a_im2colpack(0, activation.data(), k, 0, a_packed.data(), &params);
        funcs.run_conv(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, nullptr, n,
                       output.data(), n, nullptr, clamp_min, clamp_max);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Registration helpers
// ============================================================================

template <typename T, typename ClampT>
inline void register_nonpacked_convcf_benchmark(const char *name,
                                                const NonPackedConvCfBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_convcf_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

template <typename T, typename ClampT>
inline void register_nonpacked_convcl_benchmark(const char *name,
                                                const NonPackedConvClBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_convcl_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

template <typename T, typename ClampT>
inline void register_packed_convcf_benchmark(const char *name,
                                             const PackedConvCfBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_packed_convcf_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

template <typename T, typename ClampT>
inline void register_packed_convcl_benchmark(const char *name,
                                             const PackedConvClBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_packed_convcl_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

}  // namespace benchmark
}  // namespace torq_tile
