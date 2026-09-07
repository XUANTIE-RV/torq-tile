//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/elementwise_benchmark_runner.h"
#include "src/elementwise/mul_f16_f16/mulvs_f16_f16_f16/tqt_mulvs_f16_f16_f16_8vl_rvv.h"
#include "src/elementwise/mul_f16_f16/mulvv_f16_f16_f16/tqt_mulvv_f16_f16_f16_8vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

void register_mul_f16_f16_rvv_benchmarks()
{
    register_elementwise_vv_benchmark<float16_t>("tqt_mulvv_f16_f16_f16_8vl_rvv",
                                                 tqt_run_mulvv_f16_f16_f16_8vl_rvv);
    register_elementwise_vs_benchmark<float16_t>("tqt_mulvs_f16_f16_f16_8vl_rvv",
                                                 tqt_run_mulvs_f16_f16_f16_8vl_rvv,
                                                 static_cast<float16_t>(1.5f));
}

}  // namespace benchmark
}  // namespace torq_tile
