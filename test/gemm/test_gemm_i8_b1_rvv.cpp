//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_b1/gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

static tqt_qsi8d32_qsb1c128_params make_qsi8d32_qsb1c128_params(const float *scale_a, size_t ldsa,
                                                                const float *scale_b, size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

TEST(GemmI8B1Rvv, Q1_0SourceAwareTailsAndSplitK)
{
    const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qsb1c128_params> funcs = {
        Qx0SourceEncoding::Bipolar1,
        1,
        128,
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv,
        make_qsi8d32_qsb1c128_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv};
    run_packed_qsi8_qx0_source_test(5, funcs.get_n_step() + 3, 256, funcs);
}
