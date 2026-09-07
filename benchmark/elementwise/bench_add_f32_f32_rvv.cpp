//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/elementwise_benchmark_runner.h"
#include "src/elementwise/add_f32_f32/addvs_f32_f32_f32/tqt_addvs_f32_f32_f32_8vl_rvv.h"
#include "src/elementwise/add_f32_f32/addvv_f32_f32_f32/tqt_addvv_f32_f32_f32_8vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_add_f32_f32_rvv_benchmarks()
{
    register_elementwise_vv_benchmark<float>("tqt_addvv_f32_f32_f32_8vl_rvv",
                                             tqt_run_addvv_f32_f32_f32_8vl_rvv);
    register_elementwise_vs_benchmark<float>("tqt_addvs_f32_f32_f32_8vl_rvv",
                                             tqt_run_addvs_f32_f32_f32_8vl_rvv, 1.5f);
}

}  // namespace benchmark
}  // namespace torq_tile
