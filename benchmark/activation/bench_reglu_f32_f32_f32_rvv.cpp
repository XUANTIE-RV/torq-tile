//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/activation_benchmark_runner.h"
#include "src/activation/reglu_f32_f32_f32/reglu_f32_f32_f32/tqt_reglu_f32_f32_f32_1x8vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_reglu_f32_f32_f32_rvv_benchmarks()
{
    register_gated_activation_benchmark<float>("tqt_reglu_f32_f32_f32_1x8vl_rvv",
                                               tqt_run_reglu_f32_f32_f32_1x8vl_rvv);
}

}  // namespace benchmark
}  // namespace torq_tile
