//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cfloat>

#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32t/tqt_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_mx1bias_clamp_f32_f32p_f32p/tqt_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv.h"
#include "test/common/gemm_test_runner.h"

using namespace torq_tile::test;

static auto common_test_values()
{
    return testing::Values(
        GemmTestParams(32, 32, 32),   //
        GemmTestParams(5, 11, 10),    //
        GemmTestParams(7, 3, 10),     //
        GemmTestParams(17, 19, 23),   //
        GemmTestParams(31, 35, 63),   //
        GemmTestParams(60, 61, 127),  //
        GemmTestParams(32, 32, 32, "WithBias").set_bias(true),
        GemmTestParams(31, 35, 63, "WithBias").set_bias(true),
        GemmTestParams(32, 32, 32, "WithClamp").set_clamp(-100.0f, 150.0f),
        GemmTestParams(31, 35, 63, "WithClamp").set_clamp(-100.0f, 150.0f),
        GemmTestParams(32, 32, 32, "WithC").set_c(true),
        GemmTestParams(31, 35, 63, "WithC").set_c(true),
        GemmTestParams(60, 61, 127, "WithC_WithBias_WithClamp")
            .set_c(true)
            .set_bias(true)
            .set_clamp(-100.0f, 150.0f),
        GemmTestParams(127, 127, 255, "Tile_M32_N96_K128").set_tile(32, 96, 128),
        GemmTestParams(15, 17, 256, "Tile_K64").set_tile(0, 0, 64),
        GemmTestParams(31, 35, 63, "OrderMKN").set_loop_order(LoopDim::M, LoopDim::K, LoopDim::N),
        GemmTestParams(31, 35, 63, "OrderNKM").set_loop_order(LoopDim::N, LoopDim::K, LoopDim::M),
        GemmTestParams(31, 35, 63, "OrderKMN").set_loop_order(LoopDim::K, LoopDim::M, LoopDim::N));
}

// ============================================================================
// With 1xN bias, With clamp, NonPacked, B RowMajor, Tile=8x3vl
// ============================================================================

static NonPackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    };
}

class Test_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
                         common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, NonPacked, B Transposed, Tile=8x3vl
// ============================================================================

static NonPackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
    };
}

class Test_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float, BLayout::kTransposed, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
                         common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, Packed, Tile=8x3vl
// ============================================================================

static PackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    };
}

class Test_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
                         common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// With Mx1 bias, With clamp, Packed, Tile=8x3vl
// ============================================================================

static PackedGemmFuncs<float> make_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_a_packed_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_b_packed_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_c_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_bias_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_d_size_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_a_packed_size_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_b_packed_size_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_a_pack_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_b_pack_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_bt_pack_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
    };
}

class Test_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float, BiasMode::kMx1>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
                         common_test_values(), GemmTestParamNameGenerator());
