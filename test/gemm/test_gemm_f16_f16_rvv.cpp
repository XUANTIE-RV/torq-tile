//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cfloat>

#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16p/tqt_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16_f16/tqt_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16_f16/tqt_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16p_f16p/tqt_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16p_f16p/tqt_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
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

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv, float16_t, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                      funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                      funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, NonPacked, B Transposed, Tile=8x3vl
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv, float16_t, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With Mx1 bias, With clamp, NonPacked, B RowMajor, Tile=8x3vl
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_a_offset_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_b_offset_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_c_offset_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_bias_offset_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_d_size_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        clamp_test_run<tqt_run_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv, float16_t, 13, 11, 12, 14,
                       ClampSource::kRowBias, false, 5, true>(),
    };
}

class Test_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::kMx1>(GetParam(),
                                                                                      funcs);
}

TEST_P(Test_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::kMx1>(GetParam(),
                                                                                      funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(),
                          tqt_get_n_step_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, Packed, Tile=8x3vl
// ============================================================================

static PackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv, float16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, B-Only-Packed, Tile=8x3vl
// ============================================================================

static BOnlyPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv, float16_t, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv_funcs();
    run_b_only_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv_funcs();
    run_b_only_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With Mx1 bias, With clamp, Packed, Tile=8x3vl
// ============================================================================

static PackedGemmFuncs<float16_t> make_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_packed_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_c_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_bias_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_d_size_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_packed_size_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_size_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_a_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_b_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_bt_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        clamp_test_run<tqt_run_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv, float16_t, 13, 11, 12,
                       14, ClampSource::kRowBias, false, 5, true>(),
    };
}

class Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::kMx1>(GetParam(), funcs);
}

TEST_P(Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::kMx1>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(),
                          tqt_get_n_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// FP16 4x4vl (LMUL=4)
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv, float16_t, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                      funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                      funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv(), 4, true,
                          false),
    GemmTestParamNameGenerator());

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv, float16_t, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv(), 4, true,
                          false),
    GemmTestParamNameGenerator());

static PackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv, float16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv(), 4, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// FP16 2x8vl (LMUL=8)
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv, float16_t, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                      funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                      funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv(), 8, true,
                          false),
    GemmTestParamNameGenerator());

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv, float16_t, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv(), 8, true,
                          false),
    GemmTestParamNameGenerator());

static PackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv, float16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(), 8, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// FP16 8x1vl with fp32 widening accumulator (B Transposed)
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());

// ============================================================================
// FP16 8x1vl with fp32 widening accumulator, Packed (A+B)
// ============================================================================

static PackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());

// FP16 widening variants for row-major B and Mx1 bias.

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                      funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                      funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

static NonPackedGemmFuncs<float16_t> make_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_a_offset_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_b_offset_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_c_offset_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_bias_offset_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_d_size_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        clamp_test_run<tqt_run_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kRowBias, false, 5, true>(),
    };
}

class Test_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv, Correctness)
{
    auto funcs = make_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::kMx1>(GetParam(),
                                                                                      funcs);
}

TEST_P(Test_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::kMx1>(GetParam(),
                                                                                      funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(),
                          tqt_get_n_step_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

static PackedGemmFuncs<float16_t> make_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_a_packed_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_b_packed_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_c_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_bias_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_d_size_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_a_packed_size_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_b_packed_size_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_a_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_b_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_bt_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        clamp_test_run<tqt_run_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kRowBias, false, 5, true>(),
    };
}

class Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, Correctness)
{
    auto funcs = make_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::kMx1>(GetParam(), funcs);
}

TEST_P(Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_funcs();
    run_packed_gemm_test<float16_t, float16_t, BiasMode::kMx1>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                          tqt_get_n_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());

TEST(Test_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_Precision, AccumulationAndEpilogue)
{
    check_f16_f32acc_nonpacked(tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(),
                               tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(),
                               BLayout::kRowMajor, BiasMode::k1xN,
                               tqt_run_gemm_1xnbias_clamp_f16_f16_f16_8x1vl_f32acc_rvv);
}

TEST(Test_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv_Precision, AccumulationAndEpilogue)
{
    check_f16_f32acc_nonpacked(tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv(),
                               tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv(),
                               BLayout::kTransposed, BiasMode::k1xN,
                               tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv);
}

TEST(Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_Precision, AccumulationAndEpilogue)
{
    check_f16_f32acc_packed(tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                            tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                            BiasMode::k1xN,
                            tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv);
}

TEST(Test_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_Precision, AccumulationAndEpilogue)
{
    check_f16_f32acc_nonpacked(tqt_get_m_step_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(),
                               tqt_get_n_step_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(),
                               BLayout::kRowMajor, BiasMode::kMx1,
                               tqt_run_gemm_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv);
}

TEST(Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_Precision, AccumulationAndEpilogue)
{
    check_f16_f32acc_packed(tqt_get_m_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                            tqt_get_n_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                            BiasMode::kMx1,
                            tqt_run_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_a_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_b_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_bt_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv);
}

// Exercise nonzero packed K origins with padded storage, internal/external M/N
// tiling, and separate K calls whose intermediate result is stored as FP16.
static void check_f32acc_packed_offsets(const PackedGemmFuncs<float16_t> &funcs, BiasMode bias_mode)
{
    const size_t ms = funcs.get_m_step(), ns = funcs.get_n_step();
    const size_t m = 2 * ms + 3, n = 2 * ns + 3, k = 19;
    const size_t origin = 3, kp = k + 7, guard = 8;
    const size_t lda = k + 5, ldb = n + 3, ldbt = k + 2;
    const size_t ldc = n + 4, ldd = n + 7;
    const float16_t sentinel = static_cast<float16_t>(-123);
    std::vector<float16_t> a(m * lda), b(k * ldb), bt(n * ldbt), c(m * ldc);
    std::vector<float16_t> bias(bias_mode == BiasMode::kMx1 ? m : n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p)
            a[i * lda + p] = static_cast<float16_t>((int((i * 7 + p * 3) % 17) - 8) / 8.0f);
        for (size_t j = 0; j < n; ++j)
            c[i * ldc + j] = static_cast<float16_t>(2048 + 4 * i + 2 * j);
    }
    for (size_t p = 0; p < k; ++p) {
        for (size_t j = 0; j < n; ++j) {
            b[p * ldb + j] = static_cast<float16_t>((int((p * 5 + j * 3) % 19) - 9) / 8.0f);
            bt[j * ldbt + p] = b[p * ldb + j];
        }
    }
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = static_cast<float16_t>(-2048 + 8 * int(i % 7));
    ASSERT_EQ(funcs.get_a_packed_size(m, kp), m * kp * sizeof(float16_t));
    ASSERT_EQ(funcs.get_b_packed_size(n, kp), n * kp * sizeof(float16_t));
    std::vector<float16_t> ap(2 * guard + m * kp, sentinel), bp(2 * guard + n * kp, sentinel);
    auto ap_ref = ap, bp_ref = bp, btp = bp;
    const auto a_offset = [&](size_t i, size_t p, size_t rows) {
        return funcs.get_a_packed_offset(i, p, kp, rows) / sizeof(float16_t);
    };
    const auto b_offset = [&](size_t j, size_t p, size_t cols) {
        return funcs.get_b_packed_offset(j, p, kp, cols) / sizeof(float16_t);
    };
    funcs.run_a_pack(m, k, lda, kp, origin, a.data(), ap.data() + guard + a_offset(0, origin, m));
    funcs.run_b_pack(n, k, ldb, kp, origin, b.data(), bp.data() + guard + b_offset(0, origin, n));
    funcs.run_bt_pack(n, k, ldbt, kp, origin, bt.data(),
                      btp.data() + guard + b_offset(0, origin, n));
    for (size_t i = 0; i < m; i += ms) {
        const size_t rows = std::min(ms, m - i);
        EXPECT_EQ(a_offset(i, origin, rows), i * kp + origin * rows);
        for (size_t p = 0; p < k; ++p)
            for (size_t r = 0; r < rows; ++r)
                ap_ref[guard + i * kp + (origin + p) * rows + r] = a[(i + r) * lda + p];
    }
    for (size_t j = 0; j < n; j += ns) {
        const size_t cols = std::min(ns, n - j);
        EXPECT_EQ(b_offset(j, origin, cols), j * kp + origin * cols);
        for (size_t p = 0; p < k; ++p)
            for (size_t q = 0; q < cols; ++q)
                bp_ref[guard + j * kp + (origin + p) * cols + q] = b[p * ldb + j + q];
    }
    ASSERT_EQ(std::memcmp(ap.data(), ap_ref.data(), ap.size() * sizeof(float16_t)), 0);
    ASSERT_EQ(std::memcmp(bp.data(), bp_ref.data(), bp.size() * sizeof(float16_t)), 0);
    ASSERT_EQ(std::memcmp(btp.data(), bp_ref.data(), bp.size() * sizeof(float16_t)), 0);
    for (int mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE(::testing::Message() << "mode=" << mode);
        std::vector<float16_t> d(2 * guard + m * ldd, sentinel);
        auto expected = d;
        const size_t mt = mode == 1 ? ms : m, nt = mode == 1 ? ns : n;
        const size_t kt = mode == 2 ? 7 : k;
        for (size_t p = 0; p < k; p += kt) {
            const size_t count = std::min(kt, k - p);
            const bool last = p + count == k;
            for (size_t i = 0; i < m; i += mt) {
                const size_t rows = std::min(mt, m - i);
                for (size_t j = 0; j < n; j += nt) {
                    const size_t cols = std::min(nt, n - j);
                    const size_t bi = bias_mode == BiasMode::kMx1 ? i : j;
                    EXPECT_EQ(funcs.get_c_offset(i, j, ldc), (i * ldc + j) * sizeof(float16_t));
                    EXPECT_EQ(funcs.get_d_offset(i, j, ldd), (i * ldd + j) * sizeof(float16_t));
                    EXPECT_EQ(funcs.get_bias_offset(bi), bi * sizeof(float16_t));
                    funcs.run_gemm(
                        rows, cols, count, ap.data() + guard + a_offset(i, origin + p, rows), kp,
                        origin + p, bp.data() + guard + b_offset(j, origin + p, cols), kp,
                        origin + p,
                        p == 0 ? c.data() + i * ldc + j : d.data() + guard + i * ldd + j,
                        p == 0 ? ldc : ldd, d.data() + guard + i * ldd + j, ldd,
                        last ? bias.data() + bi : nullptr, static_cast<float16_t>(-INFINITY),
                        static_cast<float16_t>(INFINITY));
                }
            }
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float acc = p == 0 ? static_cast<float>(c[i * ldc + j])
                                       : static_cast<float>(expected[guard + i * ldd + j]);
                    for (size_t q = p; q < p + count; ++q)
                        acc = std::fma(static_cast<float>(a[i * lda + q]),
                                       static_cast<float>(b[q * ldb + j]), acc);
                    if (last)
                        acc += static_cast<float>(bias[bias_mode == BiasMode::kMx1 ? i : j]);
                    expected[guard + i * ldd + j] = static_cast<float16_t>(acc);
                }
            }
        }
        for (size_t i = 0; i < d.size(); ++i)
            ASSERT_EQ(static_cast<float>(d[i]), static_cast<float>(expected[i])) << "index=" << i;
    }
}

TEST(Test_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_Packing, OffsetsAndSplitK)
{
    check_f32acc_packed_offsets(make_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_funcs(),
                                BiasMode::k1xN);
}

TEST(Test_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_Packing, OffsetsAndSplitK)
{
    check_f32acc_packed_offsets(make_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_funcs(),
                                BiasMode::kMx1);
}
