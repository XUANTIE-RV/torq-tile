//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/elementwise_benchmark_runner.h"
#include "src/elementwise/div_f32_f32/divvs_f32_f32_f32/tqt_divvs_f32_f32_f32_8vl_rvv.h"
#include "src/elementwise/div_f32_f32/divvv_f32_f32_f32/tqt_divvv_f32_f32_f32_8vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_div_f32_f32_rvv_benchmarks()
{
    // Keep divisors away from zero to avoid unstable division results.
    const ElementwiseBenchmarkRandomConfig divisor_config{1.0f, 10.0f, 42};
    register_elementwise_vv_benchmark<float>("tqt_divvv_f32_f32_f32_8vl_rvv",
                                             tqt_run_divvv_f32_f32_f32_8vl_rvv, {}, divisor_config);
    register_elementwise_vs_benchmark<float>("tqt_divvs_f32_f32_f32_8vl_rvv",
                                             tqt_run_divvs_f32_f32_f32_8vl_rvv, 1.5f);
}

}  // namespace benchmark
}  // namespace torq_tile
