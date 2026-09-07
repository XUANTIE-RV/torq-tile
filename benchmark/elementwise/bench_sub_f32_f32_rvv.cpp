//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/elementwise_benchmark_runner.h"
#include "src/elementwise/sub_f32_f32/subvs_f32_f32_f32/tqt_subvs_f32_f32_f32_8vl_rvv.h"
#include "src/elementwise/sub_f32_f32/subvv_f32_f32_f32/tqt_subvv_f32_f32_f32_8vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_sub_f32_f32_rvv_benchmarks()
{
    register_elementwise_vv_benchmark<float>("tqt_subvv_f32_f32_f32_8vl_rvv",
                                             tqt_run_subvv_f32_f32_f32_8vl_rvv);
    register_elementwise_vs_benchmark<float>("tqt_subvs_f32_f32_f32_8vl_rvv",
                                             tqt_run_subvs_f32_f32_f32_8vl_rvv, 1.5f);
}

}  // namespace benchmark
}  // namespace torq_tile
