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
#include <vector>

#include "benchmark/common/benchmark_common.h"
#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// A layout enumeration (for BLAS-style SGEMM)
// ============================================================================

enum class ALayout
{
    kRowMajor,    // A stored as [M, K]
    kTransposed,  // A stored as [K, M]
};

// ============================================================================
// Function pointer types for non-packed SGEMM variants
// (BLAS-style: D = alpha * op(A) * op(B) + beta * C, no bias)
// ============================================================================

template <typename DataT>
struct NonPackedSgemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using RunSgemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                  size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                  const void *C, size_t ldc, void *D, size_t ldd, DataT alpha,
                                  DataT beta);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBOffsetFunc get_b_offset;
    GetDOffsetFunc get_d_offset;
    RunSgemmFunc run_sgemm;
};

// ============================================================================
// Non-packed SGEMM benchmark function
// D = alpha * op(A) * op(B) + beta * C
// ============================================================================

template <typename T, ALayout kALayout, BLayout kBLayout>
void run_nonpacked_sgemm_benchmark(::benchmark::State &state, const NonPackedSgemmFuncs<T> &funcs,
                                   float alpha = 1.0f, float beta = 0.0f)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));

    // Allocate and fill A
    size_t lda;
    std::vector<T> A;
    if (kALayout == ALayout::kTransposed) {
        A.resize(k * m);
        fill_random(A.data(), k * m);
        lda = m;
    } else {
        A.resize(m * k);
        fill_random(A.data(), m * k);
        lda = k;
    }

    // Allocate and fill B
    size_t ldb;
    std::vector<T> B;
    if (kBLayout == BLayout::kTransposed) {
        B.resize(n * k);
        fill_random(B.data(), n * k);
        ldb = k;
    } else {
        B.resize(k * n);
        fill_random(B.data(), k * n);
        ldb = n;
    }

    // Output buffer D
    const size_t ldd = n;
    std::vector<T> D(m * n, static_cast<T>(0));

    const T alpha_val = static_cast<T>(alpha);
    const T beta_val = static_cast<T>(beta);

    for (auto _ : state) {
        funcs.run_sgemm(m, n, k, A.data(), lda, 0, B.data(), ldb, 0, nullptr, ldd, D.data(), ldd,
                        alpha_val, beta_val);
        ::benchmark::DoNotOptimize(D.data());
    }

    // Report GFLOPS
    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Helper: register a non-packed SGEMM benchmark
// ============================================================================

template <typename T, ALayout kALayout, BLayout kBLayout>
inline void register_nonpacked_sgemm_benchmark(const char *name,
                                               const NonPackedSgemmFuncs<T> &funcs,
                                               float alpha = 1.0f, float beta = 0.0f)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs, alpha, beta](::benchmark::State &state) {
            run_nonpacked_sgemm_benchmark<T, kALayout, kBLayout>(state, funcs, alpha, beta);
        });

    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

}  // namespace benchmark
}  // namespace torq_tile
