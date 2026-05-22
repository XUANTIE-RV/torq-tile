//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark/common/benchmark_common.h"
#include "benchmark/common/conv_benchmark_runner.h"
#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Function pointer types for non-packed requantized conv-CF (channel first)
// ============================================================================

/// Non-packed channel-first conv with requantization (int32 -> int8).
/// CF layout: A = weight [OC_per_group, K], B = im2col(activation) [K, N]
/// Uses mx1bias with tqt_qai8_qsi8dx_qai8_params (per-channel scale on A).
struct NonPackedRequantizedConvCfBenchFuncs
{
    size_t (*get_b_im2col_size)(size_t k, size_t n);
    void (*run_im2col)(size_t group_idx, const void *activation, void *im2col_buf,
                       const tqt_conv_params *params);
    void (*run_conv)(size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a,
                     const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
                     size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
                     const struct tqt_qai8_qsi8dx_qai8_params *params);
};

// ============================================================================
// Function pointer types for non-packed requantized conv-CL (channel last)
// ============================================================================

/// Non-packed channel-last conv with requantization (int32 -> int8).
/// CL layout: A = im2col(activation) [M, K], B = weight [N, K] (transposed)
/// Uses 1xnbias with tqt_qai8_qai8_qsi8cx_params (per-channel scale on B).
struct NonPackedRequantizedConvClBenchFuncs
{
    size_t (*get_a_im2col_size)(size_t m, size_t k);
    void (*run_im2col)(size_t group_idx, const void *activation, void *im2col_buf,
                       const tqt_conv_params *params);
    void (*run_conv)(size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a,
                     const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
                     size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
                     const struct tqt_qai8_qai8_qsi8cx_params *params);
};

// ============================================================================
// Function pointer types for packed requantized conv-CF (channel first)
// ============================================================================

/// Packed channel-first conv with requantization (int32 -> int8).
/// CF layout: A = weight [M, K] packed, B = im2col(activation) [K, N] packed
/// Uses mx1bias with tqt_qai8_qsi8dx_qai8_params (per-channel scale on A).
struct PackedRequantizedConvCfBenchFuncs
{
    size_t (*get_a_packed_size)(size_t m, size_t k);
    size_t (*get_b_im2colpack_size)(size_t n, size_t k);
    void (*run_a_pack)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                       const void *A, void *A_packed);
    void (*run_b_im2colpack)(size_t group_idx, const void *activation, size_t ldb_packed,
                             size_t k_idx, void *B_packed, const tqt_conv_params *params);
    void (*run_conv)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
                     const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc,
                     void *D, size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
                     const struct tqt_qai8_qsi8dx_qai8_params *params);
};

// ============================================================================
// Function pointer types for packed requantized conv-CL (channel last)
// ============================================================================

/// Packed channel-last conv with requantization (int32 -> int8).
/// CL layout: A = im2col(activation) [M, K] packed, B = weight [N, K] packed
/// Uses 1xnbias with tqt_qai8_qai8_qsi8cx_params (per-channel scale on B).
struct PackedRequantizedConvClBenchFuncs
{
    size_t (*get_a_im2colpack_size)(size_t m, size_t k);
    size_t (*get_b_packed_size)(size_t n, size_t k);
    void (*run_bt_pack)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                        const void *B, void *B_packed);
    void (*run_a_im2colpack)(size_t group_idx, const void *activation, size_t lda_packed,
                             size_t k_idx, void *A_packed, const tqt_conv_params *params);
    void (*run_conv)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
                     const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc,
                     void *D, size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
                     const struct tqt_qai8_qai8_qsi8cx_params *params);
};

// ============================================================================
// Quantization parameter helpers
// ============================================================================

/// Creates dummy per-channel scale_a parameters for benchmarking CF conv.
/// CF uses mx1bias: A has per-channel quantization (scale_a is per-channel).
inline struct tqt_qai8_qsi8dx_qai8_params make_cf_qparams(size_t m, std::vector<float> &scale_a_buf)
{
    if (scale_a_buf.size() < m) {
        scale_a_buf.resize(std::max(m, static_cast<size_t>(4096)));
        std::fill(scale_a_buf.begin(), scale_a_buf.end(), 0.1f);
    }
    struct tqt_qai8_qsi8dx_qai8_params params;
    params.scale_a = scale_a_buf.data();
    params.quant_channel_a = static_cast<int32_t>(m);
    params.scale_b = 0.1f;
    params.zero_point_b = 0;
    params.scale_d = 0.1f;
    params.zero_point_d = 0;
    return params;
}

/// Creates dummy per-channel scale_b parameters for benchmarking CL conv.
/// CL uses 1xnbias: B has per-channel quantization (scale_b is per-channel).
inline struct tqt_qai8_qai8_qsi8cx_params make_cl_qparams(size_t n, std::vector<float> &scale_b_buf)
{
    if (scale_b_buf.size() < n) {
        scale_b_buf.resize(std::max(n, static_cast<size_t>(4096)));
        std::fill(scale_b_buf.begin(), scale_b_buf.end(), 0.1f);
    }
    struct tqt_qai8_qai8_qsi8cx_params params;
    params.scale_a = 0.1f;
    params.zero_point_a = 0;
    params.scale_b = scale_b_buf.data();
    params.quant_channel_b = static_cast<int32_t>(n);
    params.scale_d = 0.1f;
    params.zero_point_d = 0;
    return params;
}

// ============================================================================
// Non-packed requantized Conv-CF benchmark (im2col + gemm)
// ============================================================================

inline void run_nonpacked_requantized_convcf_benchmark(
    ::benchmark::State &state, const NonPackedRequantizedConvCfBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cf_m();
    const size_t n = cs.cf_n();
    const size_t k = cs.cf_k();

    // Activation: CF layout [IC, IH, IW]
    std::vector<int8_t> activation(cs.ic * cs.ih * cs.iw);
    fill_random<int8_t>(activation.data(), activation.size(), -128.0f, 127.0f);

    // Weight: [OC_per_group, K]
    std::vector<int8_t> weight(m * k);
    fill_random<int8_t>(weight.data(), weight.size(), -128.0f, 127.0f);

    // im2col buffer: [K, N]
    const size_t im2col_size = funcs.get_b_im2col_size(k, n);
    std::vector<uint8_t> im2col_buf(im2col_size);

    // Bias: [m, 1] (int32)
    std::vector<int32_t> bias(m, 0);

    // Output: [OC_per_group, OH*OW] (int8)
    std::vector<int8_t> output(m * n, 0);

    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;

    // Quantization parameters
    static std::vector<float> scale_a_buf;
    auto qparams = make_cf_qparams(m, scale_a_buf);

    for (auto _ : state) {
        funcs.run_im2col(0, activation.data(), im2col_buf.data(), &params);
        funcs.run_conv(m, n, k, weight.data(), k, 0, im2col_buf.data(), n, 0, nullptr, n,
                       output.data(), n, bias.data(), clamp_min, clamp_max, &qparams);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Non-packed requantized Conv-CL benchmark (im2col + gemm)
// ============================================================================

inline void run_nonpacked_requantized_convcl_benchmark(
    ::benchmark::State &state, const NonPackedRequantizedConvClBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cl_m();
    const size_t n = cs.cl_n();
    const size_t k = cs.cl_k();

    // Activation: CL layout [IH, IW, IC]
    std::vector<int8_t> activation(cs.ih * cs.iw * cs.ic);
    fill_random<int8_t>(activation.data(), activation.size(), -128.0f, 127.0f);

    // Weight: [OC_per_group, K] (transposed layout)
    std::vector<int8_t> weight(n * k);
    fill_random<int8_t>(weight.data(), weight.size(), -128.0f, 127.0f);

    // im2col buffer: [M, K]
    const size_t im2col_size = funcs.get_a_im2col_size(m, k);
    std::vector<uint8_t> im2col_buf(im2col_size);

    // Bias: [1, n] (int32)
    std::vector<int32_t> bias(n, 0);

    // Output: [OH*OW, OC_per_group] (int8)
    std::vector<int8_t> output(m * n, 0);

    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;

    // Quantization parameters
    static std::vector<float> scale_b_buf;
    auto qparams = make_cl_qparams(n, scale_b_buf);

    for (auto _ : state) {
        funcs.run_im2col(0, activation.data(), im2col_buf.data(), &params);
        funcs.run_conv(m, n, k, im2col_buf.data(), k, 0, weight.data(), k, 0, nullptr, n,
                       output.data(), n, bias.data(), clamp_min, clamp_max, &qparams);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed requantized Conv-CF benchmark (weight pre-packed, b_im2colpack + gemm)
// ============================================================================

inline void run_packed_requantized_convcf_benchmark(::benchmark::State &state,
                                                    const PackedRequantizedConvCfBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cf_m();
    const size_t n = cs.cf_n();
    const size_t k = cs.cf_k();

    // Activation: CF layout [IC, IH, IW]
    std::vector<int8_t> activation(cs.ic * cs.ih * cs.iw);
    fill_random<int8_t>(activation.data(), activation.size(), -128.0f, 127.0f);

    // Weight: [OC_per_group, K]
    std::vector<int8_t> weight(m * k);
    fill_random<int8_t>(weight.data(), weight.size(), -128.0f, 127.0f);

    // Pack weight (A) outside the timing loop
    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    std::vector<uint8_t> a_packed(a_packed_size);
    funcs.run_a_pack(m, k, k, k, 0, weight.data(), a_packed.data());

    // B packed buffer (im2colpack writes here)
    const size_t b_packed_size = funcs.get_b_im2colpack_size(n, k);
    std::vector<uint8_t> b_packed(b_packed_size);

    // Bias: [m, 1] (int32)
    std::vector<int32_t> bias(m, 0);

    // Output (int8)
    std::vector<int8_t> output(m * n, 0);

    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;

    // Quantization parameters
    static std::vector<float> scale_a_buf;
    auto qparams = make_cf_qparams(m, scale_a_buf);

    for (auto _ : state) {
        funcs.run_b_im2colpack(0, activation.data(), k, 0, b_packed.data(), &params);
        funcs.run_conv(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, nullptr, n,
                       output.data(), n, bias.data(), clamp_min, clamp_max, &qparams);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed requantized Conv-CL benchmark (weight pre-packed, a_im2colpack + gemm)
// ============================================================================

inline void run_packed_requantized_convcl_benchmark(::benchmark::State &state,
                                                    const PackedRequantizedConvClBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cl_m();
    const size_t n = cs.cl_n();
    const size_t k = cs.cl_k();

    // Activation: CL layout [IH, IW, IC]
    std::vector<int8_t> activation(cs.ih * cs.iw * cs.ic);
    fill_random<int8_t>(activation.data(), activation.size(), -128.0f, 127.0f);

    // Weight: [OC_per_group, K] (transposed)
    std::vector<int8_t> weight(n * k);
    fill_random<int8_t>(weight.data(), weight.size(), -128.0f, 127.0f);

    // Pack weight (B) outside the timing loop
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> b_packed(b_packed_size);
    funcs.run_bt_pack(n, k, k, k, 0, weight.data(), b_packed.data());

    // A packed buffer (a_im2colpack writes here)
    const size_t a_packed_size = funcs.get_a_im2colpack_size(m, k);
    std::vector<uint8_t> a_packed(a_packed_size);

    // Bias: [1, n] (int32)
    std::vector<int32_t> bias(n, 0);

    // Output (int8)
    std::vector<int8_t> output(m * n, 0);

    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;

    // Quantization parameters
    static std::vector<float> scale_b_buf;
    auto qparams = make_cl_qparams(n, scale_b_buf);

    for (auto _ : state) {
        funcs.run_a_im2colpack(0, activation.data(), k, 0, a_packed.data(), &params);
        funcs.run_conv(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, nullptr, n,
                       output.data(), n, bias.data(), clamp_min, clamp_max, &qparams);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Registration helpers
// ============================================================================

inline void register_nonpacked_requantized_convcf_benchmark(
    const char *name, const NonPackedRequantizedConvCfBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_requantized_convcf_benchmark(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

inline void register_nonpacked_requantized_convcl_benchmark(
    const char *name, const NonPackedRequantizedConvClBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_requantized_convcl_benchmark(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

inline void register_packed_requantized_convcf_benchmark(
    const char *name, const PackedRequantizedConvCfBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_packed_requantized_convcf_benchmark(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

inline void register_packed_requantized_convcl_benchmark(
    const char *name, const PackedRequantizedConvClBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_packed_requantized_convcl_benchmark(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

}  // namespace benchmark
}  // namespace torq_tile
