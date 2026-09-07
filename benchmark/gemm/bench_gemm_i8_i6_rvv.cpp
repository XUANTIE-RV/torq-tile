//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_i8_i6/gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

static tqt_qsi8d256_qsi6c16_params make_qsi8d256_qsi6c16_params(const float *scale_a, size_t ldsa,
                                                                const int16_t *bsums_a,
                                                                size_t ldbsuma,
                                                                const float *scale_b, const float *,
                                                                size_t ldsb)
{
    return {scale_a, ldsa, bsums_a, ldbsuma, scale_b, ldsb};
}

void register_gemm_i8_i6_rvv_benchmarks()
{
    const PackedQsi8BlockQuantizedGemmBenchmarkFuncs<tqt_qsi8d256_qsi6c16_params> funcs = {
        256,
        16,
        Qsi8BlockWeightEncoding::Qsi6,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        make_qsi8d256_qsi6c16_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
    };
    register_packed_qsi8_block_quantized_gemm_benchmark(
        "tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv", funcs);
}

}  // namespace benchmark
}  // namespace torq_tile
