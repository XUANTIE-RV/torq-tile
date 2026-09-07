//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/activation_benchmark_runner.h"
#include "src/activation/silu_f32_f32/silu_f32_f32/tqt_silu_f32_f32_1x4vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_silu_f32_f32_rvv_benchmarks()
{
    register_activation_benchmark<float>("tqt_silu_f32_f32_1x4vl_rvv",
                                         tqt_run_silu_f32_f32_1x4vl_rvv);
}

}  // namespace benchmark
}  // namespace torq_tile
