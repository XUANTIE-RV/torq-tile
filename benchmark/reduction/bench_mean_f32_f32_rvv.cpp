//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/reduction_benchmark_runner.h"
#include "src/reduction/mean_f32_f32/meancol_f32_f32/tqt_meancol_f32_f32_2x8vl_rvv.h"
#include "src/reduction/mean_f32_f32/meanrow_f32_f32/tqt_meanrow_f32_f32_4x4vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_mean_f32_f32_rvv_benchmarks()
{
    register_reduction_benchmark<float>("tqt_meanrow_f32_f32_4x4vl_rvv",
                                        ReductionBenchmarkAxis::kRow,
                                        tqt_run_meanrow_f32_f32_4x4vl_rvv);
    register_reduction_benchmark<float>("tqt_meancol_f32_f32_2x8vl_rvv",
                                        ReductionBenchmarkAxis::kColumn,
                                        tqt_run_meancol_f32_f32_2x8vl_rvv);
}

}  // namespace benchmark
}  // namespace torq_tile
