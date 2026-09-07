//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/reduction_benchmark_runner.h"
#include "src/reduction/mean_f16_f16/meancol_f16_f16/tqt_meancol_f16_f16_2x4vl_rvv.h"
#include "src/reduction/mean_f16_f16/meanrow_f16_f16/tqt_meanrow_f16_f16_4x2vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_mean_f16_f16_rvv_benchmarks()
{
    register_reduction_benchmark<float16_t>("tqt_meanrow_f16_f16_4x2vl_rvv",
                                            ReductionBenchmarkAxis::kRow,
                                            tqt_run_meanrow_f16_f16_4x2vl_rvv);
    register_reduction_benchmark<float16_t>("tqt_meancol_f16_f16_2x4vl_rvv",
                                            ReductionBenchmarkAxis::kColumn,
                                            tqt_run_meancol_f16_f16_2x4vl_rvv);
}

}  // namespace benchmark
}  // namespace torq_tile
