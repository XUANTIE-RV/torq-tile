//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <benchmark/benchmark.h>

#include <cstddef>
#include <string>
#include <vector>

#include "benchmark/common/benchmark_common.h"

namespace torq_tile
{
namespace benchmark
{

// Activation benchmarks come in two shapes:
//   single-operand (relu, gelu, silu): one read + one write -> 2 * m * n elements moved
//   gated          (reglu, geglu, swiglu): two reads + one write -> 3 * m * n elements moved
//
// The traffic counter is the only thing that really separates them, and it is the
// point of the gated form: fusing the gate into the activation moves 3 * m * n
// elements where a separate silu followed by a mulvv would move 5 * m * n.

using ActivationBenchmarkRunFunc = void (*)(size_t m, size_t n, const void *a, size_t lda, void *c,
                                            size_t ldc);

using GatedActivationBenchmarkRunFunc = void (*)(size_t m, size_t n, const void *a, size_t lda,
                                                 const void *b, size_t ldb, void *c, size_t ldc);

namespace detail
{

struct ActivationBenchmarkShape
{
    size_t m;
    size_t n;
};

/// The benchmark shape, taken from the global CLI override with a 128x128 default.
inline ActivationBenchmarkShape activation_benchmark_shape()
{
    ActivationBenchmarkShape shape{global_gemm_shape().m, global_gemm_shape().n};
    if (shape.m == 0) {
        shape.m = 128;
    }
    if (shape.n == 0) {
        shape.n = 128;
    }
    return shape;
}

inline std::string activation_benchmark_name(const char *name, ActivationBenchmarkShape shape)
{
    return std::string(name) + "/M=" + std::to_string(shape.m) + "/N=" + std::to_string(shape.n);
}

/// Bytes moved per iteration for `operands` reads plus one write.
inline double activation_benchmark_bytes(ActivationBenchmarkShape shape, size_t operands,
                                         size_t element_size)
{
    return static_cast<double>(operands + 1) * static_cast<double>(shape.m) *
           static_cast<double>(shape.n) * static_cast<double>(element_size);
}

inline void set_activation_benchmark_counter(::benchmark::State &state, double bytes_per_iter)
{
    state.counters["GB/s"] =
        ::benchmark::Counter(bytes_per_iter, ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::kIs1024);
}

}  // namespace detail

// ============================================================================
// Single-operand contract: run(m, n, a, lda, c, ldc)
// ============================================================================

template <typename T>
inline void run_activation_benchmark(::benchmark::State &state, size_t m, size_t n,
                                     ActivationBenchmarkRunFunc run)
{
    const size_t lda = n;
    const size_t ldc = n;

    std::vector<T> a(m * lda);
    std::vector<T> c(m * ldc);
    fill_random(a.data(), a.size());

    for (auto _ : state) {
        run(m, n, a.data(), lda, c.data(), ldc);
        ::benchmark::DoNotOptimize(c.data());
    }

    detail::set_activation_benchmark_counter(
        state, detail::activation_benchmark_bytes({m, n}, 1, sizeof(T)));
}

template <typename T>
inline void register_activation_benchmark(const char *name, ActivationBenchmarkRunFunc run)
{
    const detail::ActivationBenchmarkShape shape = detail::activation_benchmark_shape();
    const std::string full_name = detail::activation_benchmark_name(name, shape);
    const size_t m = shape.m;
    const size_t n = shape.n;

    ::benchmark::RegisterBenchmark(full_name.c_str(), [run, m, n](::benchmark::State &state) {
        run_activation_benchmark<T>(state, m, n, run);
    });
}

// ============================================================================
// Gated contract: run(m, n, a, lda, b, ldb, c, ldc)
// ============================================================================

template <typename T>
inline void run_gated_activation_benchmark(::benchmark::State &state, size_t m, size_t n,
                                           GatedActivationBenchmarkRunFunc run)
{
    const size_t lda = n;
    const size_t ldb = n;
    const size_t ldc = n;

    std::vector<T> a(m * lda);
    std::vector<T> b(m * ldb);
    std::vector<T> c(m * ldc);
    fill_random(a.data(), a.size());
    fill_random(b.data(), b.size());

    for (auto _ : state) {
        run(m, n, a.data(), lda, b.data(), ldb, c.data(), ldc);
        ::benchmark::DoNotOptimize(c.data());
    }

    detail::set_activation_benchmark_counter(
        state, detail::activation_benchmark_bytes({m, n}, 2, sizeof(T)));
}

template <typename T>
inline void register_gated_activation_benchmark(const char *name,
                                                GatedActivationBenchmarkRunFunc run)
{
    const detail::ActivationBenchmarkShape shape = detail::activation_benchmark_shape();
    const std::string full_name = detail::activation_benchmark_name(name, shape);
    const size_t m = shape.m;
    const size_t n = shape.n;

    ::benchmark::RegisterBenchmark(full_name.c_str(), [run, m, n](::benchmark::State &state) {
        run_gated_activation_benchmark<T>(state, m, n, run);
    });
}

}  // namespace benchmark
}  // namespace torq_tile
