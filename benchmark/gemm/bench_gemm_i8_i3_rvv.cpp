//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_i8_i3/gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

static tqt_qsi8d256_qsi3c16_params make_qsi3c16_params(const float *scale_a, size_t ldsa,
                                                       const int16_t *, size_t,
                                                       const int8_t *qscale_b, const uint8_t *,
                                                       size_t ldqb, const float *super_scale_b,
                                                       const float *, size_t ldssb)
{
    return {scale_a, ldsa, qscale_b, ldqb, super_scale_b, ldssb};
}

static tqt_qsi8d256_qsi4c16_params make_qsi4c16_params(const float *scale_a, size_t ldsa,
                                                       const int16_t *, size_t,
                                                       const int8_t *qscale_b, const uint8_t *,
                                                       size_t ldqb, const float *super_scale_b,
                                                       const float *, size_t ldssb)
{
    return {scale_a, ldsa, qscale_b, ldqb, super_scale_b, ldssb};
}

void register_gemm_i8_i3_rvv_benchmarks()
{
    const PackedQsi8KBlockGemmBenchmarkFuncs<tqt_qsi8d256_qsi3c16_params> compact = {
        Qsi8KBlockBenchmarkEncoding::Qsi3,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
        make_qsi3c16_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv};
    const PackedQsi8KBlockGemmBenchmarkFuncs<tqt_qsi8d256_qsi4c16_params> expanded = {
        Qsi8KBlockBenchmarkEncoding::Qsi4,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
        make_qsi4c16_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv};
    register_packed_qsi8_kblock_gemm_benchmark(
        "tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv", compact);
    register_packed_qsi8_kblock_gemm_benchmark(
        "tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv", expanded);
}

}  // namespace benchmark
}  // namespace torq_tile
