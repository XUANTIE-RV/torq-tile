//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/elementwise_benchmark_runner.h"
#include "src/elementwise/div_f16_f16/divvs_f16_f16_f16/tqt_divvs_f16_f16_f16_8vl_rvv.h"
#include "src/elementwise/div_f16_f16/divvv_f16_f16_f16/tqt_divvv_f16_f16_f16_8vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_div_f16_f16_rvv_benchmarks()
{
    // Keep divisors away from zero to avoid unstable division results.
    const ElementwiseBenchmarkRandomConfig divisor_config{1.0f, 10.0f, 42};
    register_elementwise_vv_benchmark<float16_t>(
        "tqt_divvv_f16_f16_f16_8vl_rvv", tqt_run_divvv_f16_f16_f16_8vl_rvv, {}, divisor_config);
    register_elementwise_vs_benchmark<float16_t>("tqt_divvs_f16_f16_f16_8vl_rvv",
                                                 tqt_run_divvs_f16_f16_f16_8vl_rvv,
                                                 static_cast<float16_t>(1.5f));
}

}  // namespace benchmark
}  // namespace torq_tile
