//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_i8_b1/gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

static tqt_qsi8d32_qsb1c128_params make_qsb1c128_params(const float *scale_a, size_t ldsa,
                                                        const int16_t *, size_t,
                                                        const float *scale_b, const float *,
                                                        size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

void register_gemm_i8_b1_rvv_benchmarks()
{
    const PackedQsi8BlockQuantizedGemmBenchmarkFuncs<tqt_qsi8d32_qsb1c128_params> funcs = {
        32,
        128,
        Qsi8BlockWeightEncoding::Qsb1,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        make_qsb1c128_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv};
    register_packed_qsi8_block_quantized_gemm_benchmark(
        "tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv", funcs);
}

}  // namespace benchmark
}  // namespace torq_tile
