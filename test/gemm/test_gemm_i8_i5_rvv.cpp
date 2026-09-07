//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_i5/gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

static tqt_qsi8d32_qsi5c32_params make_qsi8d32_qsi5c32_params(const float *scale_a, size_t ldsa,
                                                              const float *scale_b, size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

TEST(GemmI8I5Rvv, Q5_0SourceAwareTailsAndSplitK)
{
    const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qsi5c32_params> funcs = {
        Qx0SourceEncoding::I5,
        5,
        32,
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        make_qsi8d32_qsi5c32_params,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>()};
    run_packed_qsi8_qx0_source_test(5, funcs.get_n_step() + 3, 64, funcs);
}

TEST(GemmI8I5Rvv, ClampSpecialValues_Q5_0SourceAwareTailsAndSplitK)
{
    ScopedGemmClampProbe probe;
    const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qsi5c32_params> funcs = {
        Qx0SourceEncoding::I5,
        5,
        32,
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        make_qsi8d32_qsi5c32_params,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>()};
    for (size_t rows = 1;
         rows <= tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv() + 1; ++rows) {
        for (size_t columns :
             {size_t{1}, tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(),
              tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv() + 3}) {
            run_packed_qsi8_qx0_source_test(rows, columns, 64, funcs);
        }
    }
}
