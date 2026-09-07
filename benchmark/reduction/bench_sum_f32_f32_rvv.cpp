//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/reduction_benchmark_runner.h"
#include "src/reduction/sum_f32_f32/sumcol_f32_f32/tqt_sumcol_f32_f32_2x8vl_rvv.h"
#include "src/reduction/sum_f32_f32/sumrow_f32_f32/tqt_sumrow_f32_f32_4x4vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_sum_f32_f32_rvv_benchmarks()
{
    register_reduction_benchmark<float>("tqt_sumrow_f32_f32_4x4vl_rvv",
                                        ReductionBenchmarkAxis::kRow,
                                        tqt_run_sumrow_f32_f32_4x4vl_rvv);
    register_reduction_benchmark<float>("tqt_sumcol_f32_f32_2x8vl_rvv",
                                        ReductionBenchmarkAxis::kColumn,
                                        tqt_run_sumcol_f32_f32_2x8vl_rvv);
}

}  // namespace benchmark
}  // namespace torq_tile
