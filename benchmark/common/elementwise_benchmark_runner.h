//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/common/benchmark_common.h"

namespace torq_tile
{
namespace benchmark
{

struct ElementwiseBenchmarkRandomConfig
{
    float low = -1.0f;
    float high = 1.0f;
    uint32_t seed = 42;
};

template <typename T, typename KernelFn>
inline void run_elementwise_vv_benchmark(::benchmark::State &state, size_t n, KernelFn kernel,
                                         const ElementwiseBenchmarkRandomConfig &a_config,
                                         const ElementwiseBenchmarkRandomConfig &b_config)
{
    std::vector<T> a(n);
    std::vector<T> b(n);
    std::vector<T> c(n);

    fill_random(a.data(), n, a_config.low, a_config.high, a_config.seed);
    fill_random(b.data(), n, b_config.low, b_config.high, b_config.seed);

    for (auto _ : state) {
        kernel(n, a.data(), b.data(), c.data());
    }

    const double bytes_per_iter = static_cast<double>(n) * sizeof(T) * 3.0;
    state.counters["GB/s"] =
        ::benchmark::Counter(bytes_per_iter, ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::kIs1024);
}

template <typename T, typename KernelFn>
inline void run_elementwise_vs_benchmark(::benchmark::State &state, size_t n, KernelFn kernel,
                                         T scalar_b,
                                         const ElementwiseBenchmarkRandomConfig &a_config)
{
    std::vector<T> a(n);
    std::vector<T> c(n);

    fill_random(a.data(), n, a_config.low, a_config.high, a_config.seed);

    for (auto _ : state) {
        kernel(n, a.data(), scalar_b, c.data());
    }

    const double bytes_per_iter = static_cast<double>(n) * sizeof(T) * 2.0;
    state.counters["GB/s"] =
        ::benchmark::Counter(bytes_per_iter, ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::kIs1024);
}

template <typename T, typename KernelFn>
inline void register_elementwise_vv_benchmark(const char *name, KernelFn kernel,
                                              const ElementwiseBenchmarkRandomConfig &a_config = {},
                                              const ElementwiseBenchmarkRandomConfig &b_config = {})
{
    size_t n = global_gemm_shape().n;
    if (n == 0) {
        n = 4096;
    }
    const std::string benchmark_name = std::string(name) + "/N=" + std::to_string(n);
    ::benchmark::RegisterBenchmark(
        benchmark_name, [n, kernel, a_config, b_config](::benchmark::State &state) {
            run_elementwise_vv_benchmark<T>(state, n, kernel, a_config, b_config);
        });
}

template <typename T, typename KernelFn>
inline void register_elementwise_vs_benchmark(const char *name, KernelFn kernel, T scalar_b,
                                              const ElementwiseBenchmarkRandomConfig &a_config = {})
{
    size_t n = global_gemm_shape().n;
    if (n == 0) {
        n = 4096;
    }
    const std::string benchmark_name = std::string(name) + "/N=" + std::to_string(n);
    ::benchmark::RegisterBenchmark(
        benchmark_name, [n, kernel, scalar_b, a_config](::benchmark::State &state) {
            run_elementwise_vs_benchmark<T>(state, n, kernel, scalar_b, a_config);
        });
}

}  // namespace benchmark
}  // namespace torq_tile
