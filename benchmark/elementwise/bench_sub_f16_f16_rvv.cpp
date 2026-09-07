//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/elementwise_benchmark_runner.h"
#include "src/elementwise/sub_f16_f16/subvs_f16_f16_f16/tqt_subvs_f16_f16_f16_8vl_rvv.h"
#include "src/elementwise/sub_f16_f16/subvv_f16_f16_f16/tqt_subvv_f16_f16_f16_8vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_sub_f16_f16_rvv_benchmarks()
{
    register_elementwise_vv_benchmark<float16_t>("tqt_subvv_f16_f16_f16_8vl_rvv",
                                                 tqt_run_subvv_f16_f16_f16_8vl_rvv);
    register_elementwise_vs_benchmark<float16_t>("tqt_subvs_f16_f16_f16_8vl_rvv",
                                                 tqt_run_subvs_f16_f16_f16_8vl_rvv,
                                                 static_cast<float16_t>(1.5f));
}

}  // namespace benchmark
}  // namespace torq_tile
