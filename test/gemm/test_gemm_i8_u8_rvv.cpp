//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_u8/gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

static tqt_qsi8d256_qau8c32_params make_qsi8d256_qau8c32_test_params(
    const float *scale_a, size_t ldsa, const int16_t *bsums_a, size_t ldbsuma, const float *scale_b,
    const float *real_zero_b, size_t ldsb)
{
    return {scale_a, ldsa, bsums_a, ldbsuma, scale_b, real_zero_b, ldsb};
}

static PackedQsi8BlockQuantizedGemmTestFuncs<tqt_qsi8d256_qau8c32_params>
make_qsi8d256_qau8c32_test_funcs()
{
    return {
        256,
        32,
        true,
        31,
        Qsi8BlockWeightEncoding::Qau8,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        make_qsi8d256_qau8c32_test_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
    };
}

TEST(GemmI8U8Rvv, Qsi8d256Qau8c32)
{
    const auto funcs = make_qsi8d256_qau8c32_test_funcs();
    const size_t n = funcs.get_n_step() + 3;
    run_packed_qsi8_block_quantized_gemm_test(5, n, 512, funcs);
}
