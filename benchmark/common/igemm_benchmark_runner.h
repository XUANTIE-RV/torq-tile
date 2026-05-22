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
#include "benchmark/common/conv_benchmark_runner.h"
#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Function pointer types for non-packed igemmcf (channel-first)
// ============================================================================

template <typename ClampT>
struct NonPackedIgemmcfBenchFuncs
{
    void (*run_igemm)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                      size_t n_idx, size_t k_idx_b, const void *A, size_t lda, size_t k_idx_a,
                      const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                      ClampT clamp_min, ClampT clamp_max, const tqt_conv_params *params);
};

// ============================================================================
// Function pointer types for packed igemmcf (channel-first)
// ============================================================================

template <typename ClampT>
struct PackedIgemmcfBenchFuncs
{
    size_t (*get_a_packed_size)(size_t m, size_t k);
    void (*run_a_pack)(size_t m, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx,
                       const void *weight, void *weight_packed);
    void (*run_igemm)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                      size_t n_idx, size_t k_idx_b, const void *A_packed, size_t lda,
                      size_t k_idx_a, const void *C, size_t ldc, void *D, size_t ldd,
                      const void *bias, ClampT clamp_min, ClampT clamp_max,
                      const tqt_conv_params *params);
};

// ============================================================================
// Function pointer types for non-packed igemmcl (channel-last)
// ============================================================================

template <typename ClampT>
struct NonPackedIgemmclBenchFuncs
{
    void (*run_igemm)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                      size_t m_idx, size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                      const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                      ClampT clamp_min, ClampT clamp_max, const tqt_conv_params *params);
};

// ============================================================================
// Function pointer types for packed igemmcl (channel-last)
// ============================================================================

template <typename ClampT>
struct PackedIgemmclBenchFuncs
{
    size_t (*get_b_packed_size)(size_t n, size_t k);
    void (*run_bt_pack)(size_t n, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx,
                        const void *weight, void *weight_packed);
    void (*run_igemm)(size_t m, size_t n, size_t k, const void *input, size_t group_idx,
                      size_t m_idx, size_t k_idx_a, const void *B_packed, size_t ldb,
                      size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                      const void *bias, ClampT clamp_min, ClampT clamp_max,
                      const tqt_conv_params *params);
};

// ============================================================================
// Non-packed igemmcf benchmark runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_nonpacked_igemmcf_benchmark(::benchmark::State &state,
                                            const NonPackedIgemmcfBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cf_m();
    const size_t n = cs.cf_n();
    const size_t k = cs.cf_k();

    std::vector<T> activation(cs.ic * cs.ih * cs.iw);
    fill_random<T>(activation.data(), activation.size());

    std::vector<T> weight(m * k);
    fill_random<T>(weight.data(), weight.size());

    std::vector<T> output(m * n, static_cast<T>(0));

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_igemm(m, n, k, activation.data(), 0, 0, 0, weight.data(), k, 0, nullptr, n,
                        output.data(), n, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed igemmcf benchmark runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_packed_igemmcf_benchmark(::benchmark::State &state,
                                         const PackedIgemmcfBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cf_m();
    const size_t n = cs.cf_n();
    const size_t k = cs.cf_k();

    std::vector<T> activation(cs.ic * cs.ih * cs.iw);
    fill_random<T>(activation.data(), activation.size());

    std::vector<T> weight(m * k);
    fill_random<T>(weight.data(), weight.size());

    const size_t w_packed_size = funcs.get_a_packed_size(m, k);
    std::vector<uint8_t> w_packed(w_packed_size);
    funcs.run_a_pack(m, k, k, k, 0, weight.data(), w_packed.data());

    std::vector<T> output(m * n, static_cast<T>(0));

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_igemm(m, n, k, activation.data(), 0, 0, 0, w_packed.data(), k, 0, nullptr, n,
                        output.data(), n, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Non-packed igemmcl benchmark runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_nonpacked_igemmcl_benchmark(::benchmark::State &state,
                                            const NonPackedIgemmclBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cl_m();
    const size_t n = cs.cl_n();
    const size_t k = cs.cl_k();

    std::vector<T> activation(cs.ih * cs.iw * cs.ic);
    fill_random<T>(activation.data(), activation.size());

    std::vector<T> weight(n * k);
    fill_random<T>(weight.data(), weight.size());

    std::vector<T> output(m * n, static_cast<T>(0));

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_igemm(m, n, k, activation.data(), 0, 0, 0, weight.data(), k, 0, nullptr, n,
                        output.data(), n, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(conv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed igemmcl benchmark runner
// ============================================================================

template <typename T, typename ClampT>
inline void run_packed_igemmcl_benchmark(::benchmark::State &state,
                                         const PackedIgemmclBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params params = cs.to_params();

    const size_t m = cs.cl_m();
    const size_t n = cs.cl_n();
    const size_t k = cs.cl_k();

    std::vector<T> activation(cs.ih * cs.iw * cs.ic);
    fill_random<T>(activation.data(), activation.size());

    std::vector<T> weight(n * k);
    fill_random<T>(weight.data(), weight.size());

    const size_t w_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> w_packed(w_packed_size);
    funcs.run_bt_pack(n, k, k, k, 0, weight.data(), w_packed.data());

    std::vector<T> output(m * n, static_cast<T>(0));

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_igemm(m, n, k, activation.data(), 0, 0, 0, w_packed.data(), k, 0, nullptr, n,
                        output.data(), n, nullptr, clamp_min, clamp_max, &params);
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
inline void register_nonpacked_igemmcf_benchmark(const char *name,
                                                 const NonPackedIgemmcfBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_igemmcf_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

template <typename T, typename ClampT>
inline void register_packed_igemmcf_benchmark(const char *name,
                                              const PackedIgemmcfBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_packed_igemmcf_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

template <typename T, typename ClampT>
inline void register_nonpacked_igemmcl_benchmark(const char *name,
                                                 const NonPackedIgemmclBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_igemmcl_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

template <typename T, typename ClampT>
inline void register_packed_igemmcl_benchmark(const char *name,
                                              const PackedIgemmclBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_packed_igemmcl_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.oc),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

}  // namespace benchmark
}  // namespace torq_tile
