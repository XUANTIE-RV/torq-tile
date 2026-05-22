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
#include "benchmark/common/quantized_conv_benchmark_runner.h"
#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Function pointer types for non-packed requantized igemmcf (channel-first)
// ============================================================================

/// Non-packed igemmcf with requantization (int32 -> int8).
/// CF layout: A = weight [OC_per_group, K], input fetched on-the-fly
/// Uses mx1bias with tqt_qai8_qsi8dx_qai8_params (per-channel scale on A).
struct NonPackedRequantizedIgemmcfBenchFuncs
{
    void (*run_igemm)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                      size_t n_idx, size_t k_idx_b, const void *A, size_t lda, size_t k_idx_a,
                      const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                      int8_t clamp_min, int8_t clamp_max, const tqt_conv_params *conv_params,
                      const struct tqt_qai8_qsi8dx_qai8_params *quant_params);
};

// ============================================================================
// Function pointer types for packed requantized igemmcf (channel-first)
// ============================================================================

/// Packed igemmcf with requantization (int32 -> int8).
/// CF layout: A = weight [OC_per_group, K] packed, input fetched on-the-fly
struct PackedRequantizedIgemmcfBenchFuncs
{
    size_t (*get_a_packed_size)(size_t m, size_t k);
    void (*run_a_pack)(size_t m, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx,
                       const void *weight, void *weight_packed);
    void (*run_igemm)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                      size_t n_idx, size_t k_idx_b, const void *A_packed, size_t lda,
                      size_t k_idx_a, const void *C, size_t ldc, void *D, size_t ldd,
                      const void *bias, int8_t clamp_min, int8_t clamp_max,
                      const tqt_conv_params *conv_params,
                      const struct tqt_qai8_qsi8dx_qai8_params *quant_params);
};

// ============================================================================
// Function pointer types for non-packed requantized igemmcl (channel-last)
// ============================================================================

/// Non-packed igemmcl with requantization (int32 -> int8).
/// CL layout: input fetched on-the-fly, B = weight [N, K] (transposed)
/// Uses 1xnbias with tqt_qai8_qai8_qsi8cx_params (per-channel scale on B).
struct NonPackedRequantizedIgemmclBenchFuncs
{
    void (*run_igemm)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                      size_t m_idx, size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                      const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                      int8_t clamp_min, int8_t clamp_max, const tqt_conv_params *conv_params,
                      const struct tqt_qai8_qai8_qsi8cx_params *quant_params);
};

// ============================================================================
// Function pointer types for packed requantized igemmcl (channel-last)
// ============================================================================

/// Packed igemmcl with requantization (int32 -> int8).
/// CL layout: input fetched on-the-fly, B = weight [N, K] packed
struct PackedRequantizedIgemmclBenchFuncs
{
    size_t (*get_b_packed_size)(size_t n, size_t k);
    void (*run_bt_pack)(size_t n, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx,
                        const void *weight, void *weight_packed);
    void (*run_igemm)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                      size_t m_idx, size_t k_idx_a, const void *B_packed, size_t ldb,
                      size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                      const void *bias, int8_t clamp_min, int8_t clamp_max,
                      const tqt_conv_params *conv_params,
                      const struct tqt_qai8_qai8_qsi8cx_params *quant_params);
};

// ============================================================================
// Non-packed requantized igemmcf benchmark
// ============================================================================

inline void run_nonpacked_requantized_igemmcf_benchmark(
    ::benchmark::State &state, const NonPackedRequantizedIgemmcfBenchFuncs &funcs)
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
        funcs.run_igemm(m, n, k, activation.data(), 0, 0, 0, weight.data(), k, 0, nullptr, n,
                        output.data(), n, bias.data(), clamp_min, clamp_max, &params, &qparams);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed requantized igemmcf benchmark
// ============================================================================

inline void run_packed_requantized_igemmcf_benchmark(
    ::benchmark::State &state, const PackedRequantizedIgemmcfBenchFuncs &funcs)
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
        funcs.run_igemm(m, n, k, activation.data(), 0, 0, 0, a_packed.data(), k, 0, nullptr, n,
                        output.data(), n, bias.data(), clamp_min, clamp_max, &params, &qparams);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Non-packed requantized igemmcl benchmark
// ============================================================================

inline void run_nonpacked_requantized_igemmcl_benchmark(
    ::benchmark::State &state, const NonPackedRequantizedIgemmclBenchFuncs &funcs)
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
        funcs.run_igemm(m, n, k, activation.data(), 0, 0, 0, weight.data(), k, 0, nullptr, n,
                        output.data(), n, bias.data(), clamp_min, clamp_max, &params, &qparams);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed requantized igemmcl benchmark
// ============================================================================

inline void run_packed_requantized_igemmcl_benchmark(
    ::benchmark::State &state, const PackedRequantizedIgemmclBenchFuncs &funcs)
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
        funcs.run_igemm(m, n, k, activation.data(), 0, 0, 0, b_packed.data(), k, 0, nullptr, n,
                        output.data(), n, bias.data(), clamp_min, clamp_max, &params, &qparams);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Registration helpers
// ============================================================================

inline void register_nonpacked_requantized_igemmcf_benchmark(
    const char *name, const NonPackedRequantizedIgemmcfBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_requantized_igemmcf_benchmark(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

inline void register_packed_requantized_igemmcf_benchmark(
    const char *name, const PackedRequantizedIgemmcfBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_packed_requantized_igemmcf_benchmark(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

inline void register_nonpacked_requantized_igemmcl_benchmark(
    const char *name, const NonPackedRequantizedIgemmclBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_requantized_igemmcl_benchmark(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

inline void register_packed_requantized_igemmcl_benchmark(
    const char *name, const PackedRequantizedIgemmclBenchFuncs &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_packed_requantized_igemmcl_benchmark(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

}  // namespace benchmark
}  // namespace torq_tile
