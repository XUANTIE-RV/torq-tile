//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/activation_benchmark_runner.h"
#include "src/activation/relu_f32_f32/relu_f32_f32/tqt_relu_f32_f32_1x8vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_relu_f32_f32_rvv_benchmarks()
{
    register_activation_benchmark<float>("tqt_relu_f32_f32_1x8vl_rvv",
                                         tqt_run_relu_f32_f32_1x8vl_rvv);
}

}  // namespace benchmark
}  // namespace torq_tile
