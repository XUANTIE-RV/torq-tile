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

enum class ReductionBenchmarkAxis
{
    kRow,
    kColumn,
};

template <typename T, typename KernelFn>
inline void run_reduction_benchmark(::benchmark::State &state, size_t m, size_t n,
                                    ReductionBenchmarkAxis axis, KernelFn kernel)
{
    const size_t output_length = axis == ReductionBenchmarkAxis::kRow ? m : n;
    std::vector<T> input(m * n);
    std::vector<T> output(output_length);

    fill_random(input.data(), m * n);

    for (auto _ : state) {
        kernel(m, n, input.data(), n, output.data(), nullptr);
    }

    const double bytes_per_iter =
        (static_cast<double>(m) * static_cast<double>(n) + static_cast<double>(output_length)) *
        sizeof(T);
    state.counters["GB/s"] =
        ::benchmark::Counter(bytes_per_iter, ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::kIs1024);
}

template <typename T, typename KernelFn>
inline void register_reduction_benchmark(const char *name, ReductionBenchmarkAxis axis,
                                         KernelFn kernel)
{
    size_t m = global_gemm_shape().m;
    size_t n = global_gemm_shape().n;
    if (m == 0) {
        m = 128;
    }
    if (n == 0) {
        n = 128;
    }

    const std::string suffix = "/M=" + std::to_string(m) + "/N=" + std::to_string(n);
    ::benchmark::RegisterBenchmark(std::string(name) + suffix,
                                   [m, n, axis, kernel](::benchmark::State &state) {
                                       run_reduction_benchmark<T>(state, m, n, axis, kernel);
                                   });
}

}  // namespace benchmark
}  // namespace torq_tile
