//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/reduction_benchmark_runner.h"
#include "src/reduction/sum_f16_f16/sumcol_f16_f16/tqt_sumcol_f16_f16_2x4vl_rvv.h"
#include "src/reduction/sum_f16_f16/sumrow_f16_f16/tqt_sumrow_f16_f16_4x2vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_sum_f16_f16_rvv_benchmarks()
{
    register_reduction_benchmark<float16_t>("tqt_sumrow_f16_f16_4x2vl_rvv",
                                            ReductionBenchmarkAxis::kRow,
                                            tqt_run_sumrow_f16_f16_4x2vl_rvv);
    register_reduction_benchmark<float16_t>("tqt_sumcol_f16_f16_2x4vl_rvv",
                                            ReductionBenchmarkAxis::kColumn,
                                            tqt_run_sumcol_f16_f16_2x4vl_rvv);
}

}  // namespace benchmark
}  // namespace torq_tile
