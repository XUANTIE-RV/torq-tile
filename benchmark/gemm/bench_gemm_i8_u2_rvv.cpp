//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_i8_u2/gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_u2/gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_u4/gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

static tqt_qsi8d256_qau2c16_params make_qau2c16_params(const float *scale_a, size_t ldsa,
                                                       const int16_t *bsums_a, size_t ldbsuma,
                                                       const int8_t *qscale_b,
                                                       const uint8_t *qmin_b, size_t ldqb,
                                                       const float *super_scale_b,
                                                       const float *super_min_b, size_t ldssb)
{
    return {scale_a, ldsa, bsums_a,       ldbsuma,     reinterpret_cast<const uint8_t *>(qscale_b),
            qmin_b,  ldqb, super_scale_b, super_min_b, ldssb};
}

static tqt_qsi8d256_qau4c16_params make_qau4c16_params(const float *scale_a, size_t ldsa,
                                                       const int16_t *bsums_a, size_t ldbsuma,
                                                       const int8_t *qscale_b,
                                                       const uint8_t *qmin_b, size_t ldqb,
                                                       const float *super_scale_b,
                                                       const float *super_min_b, size_t ldssb)
{
    return {scale_a, ldsa, bsums_a,       ldbsuma,     reinterpret_cast<const uint8_t *>(qscale_b),
            qmin_b,  ldqb, super_scale_b, super_min_b, ldssb};
}

static tqt_qsi8d32_qau2z1c64_params make_qau2z1c64_params(const float *scale_a, size_t ldsa,
                                                          const int16_t *, size_t,
                                                          const float *scale_b, const float *,
                                                          size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

void register_gemm_i8_u2_rvv_benchmarks()
{
    const PackedQsi8KBlockGemmBenchmarkFuncs<tqt_qsi8d256_qau2c16_params> compact = {
        Qsi8KBlockBenchmarkEncoding::Qau2,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
        make_qau2c16_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv};
    const PackedQsi8KBlockGemmBenchmarkFuncs<tqt_qsi8d256_qau4c16_params> expanded = {
        Qsi8KBlockBenchmarkEncoding::Qau4,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
        make_qau4c16_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv};
    register_packed_qsi8_kblock_gemm_benchmark(
        "tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv", compact);
    register_packed_qsi8_kblock_gemm_benchmark(
        "tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv", expanded);

    const PackedQsi8BlockQuantizedGemmBenchmarkFuncs<tqt_qsi8d32_qau2z1c64_params> q2_0 = {
        32,
        64,
        Qsi8BlockWeightEncoding::Qau2,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        make_qau2z1c64_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv};
    register_packed_qsi8_block_quantized_gemm_benchmark(
        "tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv", q2_0);
}

}  // namespace benchmark
}  // namespace torq_tile
