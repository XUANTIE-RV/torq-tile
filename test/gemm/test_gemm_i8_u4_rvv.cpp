//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_u4/gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

static tqt_qsi8d256_qau4c32_params make_qsi8d256_qau4c32_test_params(
    const float *scale_a, size_t ldsa, const int16_t *bsums_a, size_t ldbsuma, const float *scale_b,
    const float *real_zero_b, size_t ldsb)
{
    return {scale_a, ldsa, bsums_a, ldbsuma, scale_b, real_zero_b, ldsb};
}

static PackedQsi8BlockQuantizedGemmTestFuncs<tqt_qsi8d256_qau4c32_params>
make_qsi8d256_qau4c32_test_funcs()
{
    return {
        256,
        32,
        true,
        15,
        Qsi8BlockWeightEncoding::Qau4,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv,
        make_qsi8d256_qau4c32_test_params,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

TEST(GemmI8U4Rvv, Qsi8d256Qau4c32)
{
    const auto funcs = make_qsi8d256_qau4c32_test_funcs();
    for (size_t rows = 1; rows <= 5; ++rows) {
        for (size_t columns : {size_t{1}, funcs.get_n_step(), funcs.get_n_step() + 3}) {
            run_packed_qsi8_block_quantized_gemm_test(rows, columns, 512, funcs);
        }
    }
}

TEST(GemmI8U4Rvv, ClampSpecialValues_Qsi8d256Qau4c32)
{
    ScopedGemmClampProbe probe;
    const auto funcs = make_qsi8d256_qau4c32_test_funcs();
    for (size_t rows = 1;
         rows <= tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv() + 1; ++rows) {
        for (size_t columns :
             {size_t{1}, tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv(),
              tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv() + 3}) {
            run_packed_qsi8_block_quantized_gemm_test(rows, columns, 512, funcs);
        }
    }
    const size_t m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv();
    const size_t n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_4x1vl_rvv();
    for (size_t tail = 1; tail <= m_step; ++tail) {
        for (size_t n_tail : {size_t{1}, size_t{3}, n_step}) {
            run_packed_qsi8_block_quantized_gemm_test(m_step + tail, n_step + n_tail, 512, funcs);
        }
    }
}

TEST(GemmI8U4Rvv, NarrowAccumulationExtremes)
{
    run_packed_qsi8_block_extreme_test(make_qsi8d256_qau4c32_test_funcs(), 0, 15);
}
