//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/activation_benchmark_runner.h"
#include "src/activation/relu_f16_f16/relu_f16_f16/tqt_relu_f16_f16_1x8vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_relu_f16_f16_rvv_benchmarks()
{
    register_activation_benchmark<float16_t>("tqt_relu_f16_f16_1x8vl_rvv",
                                             tqt_run_relu_f16_f16_1x8vl_rvv);
}

}  // namespace benchmark
}  // namespace torq_tile
