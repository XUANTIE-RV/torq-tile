//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cfloat>

#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32p/tqt_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32t/tqt_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32t/tqt_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32t/tqt_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32p_f32p/tqt_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_mx1bias_clamp_f32_f32_f32/tqt_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/gemm_mx1bias_clamp_f32_f32p_f32p/tqt_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv.h"
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
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

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
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kTransposed, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kTransposed, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

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
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float, float, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float, float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, B-Only-Packed, Tile=8x3vl
// ============================================================================

static BOnlyPackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv_funcs();
    run_b_only_packed_gemm_test<float, float, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv_funcs();
    run_b_only_packed_gemm_test<float, float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With Mx1 bias, With clamp, NonPacked, B RowMajor, Tile=8x3vl
// ============================================================================

static NonPackedGemmFuncs<float> make_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_a_offset_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_b_offset_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_c_offset_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_bias_offset_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_d_size_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        clamp_test_run<tqt_run_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kRowBias, false, 5, true>(),
    };
}

class Test_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kRowMajor, BiasMode::kMx1>(GetParam(), funcs);
}

TEST_P(Test_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kRowMajor, BiasMode::kMx1>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv(),
                          tqt_get_n_step_gemm_mx1bias_clamp_f32_f32_f32_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

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
        clamp_test_run<tqt_run_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kRowBias, false, 5, true>(),
    };
}

class Test_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv, Correctness)
{
    auto funcs = make_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float, float, BiasMode::kMx1>(GetParam(), funcs);
}

TEST_P(Test_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv_funcs();
    run_packed_gemm_test<float, float, BiasMode::kMx1>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(),
                          tqt_get_n_step_gemm_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv(), 3, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, NonPacked, B RowMajor, Tile=4x4vl (LMUL=4)
// ============================================================================

static NonPackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_4x4vl_rvv(), 4, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, NonPacked, B Transposed, Tile=4x4vl (LMUL=4)
// ============================================================================

static NonPackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kTransposed, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kTransposed, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(), 4, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, Packed, Tile=4x4vl (LMUL=4)
// ============================================================================

static PackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv_funcs();
    run_packed_gemm_test<float, float, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv_funcs();
    run_packed_gemm_test<float, float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(), 4, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, NonPacked, B RowMajor, Tile=2x8vl (LMUL=8)
// ============================================================================

static NonPackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_2x8vl_rvv(), 8, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, NonPacked, B Transposed, Tile=2x8vl (LMUL=8)
// ============================================================================

static NonPackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kTransposed, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv_funcs();
    run_nonpacked_gemm_test<float, float, BLayout::kTransposed, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32t_2x8vl_rvv(), 8, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, Packed, Tile=2x8vl (LMUL=8)
// ============================================================================

static PackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv_funcs();
    run_packed_gemm_test<float, float, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv_funcs();
    run_packed_gemm_test<float, float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_2x8vl_rvv(), 8, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// SGEMM (BLAS-style): D = alpha * op(A) * op(B) + beta * C
// ============================================================================

#include "src/gemm/gemm_f32_f32/sgemm_f32_f32_f32/tqt_sgemm_f32_f32_f32_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/sgemm_f32_f32_f32t/tqt_sgemm_f32_f32_f32t_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/sgemm_f32_f32t_f32/tqt_sgemm_f32_f32t_f32_8x3vl_rvv.h"
#include "src/gemm/gemm_f32_f32/sgemm_f32_f32t_f32t/tqt_sgemm_f32_f32t_f32t_8x3vl_rvv.h"
#include "test/common/sgemm_test_runner.h"

using torq_tile::test::ALayout;
using torq_tile::test::NonPackedSgemmFuncs;
using torq_tile::test::SgemmTestParamNameGenerator;
using torq_tile::test::SgemmTestParams;

static auto sgemm_test_values()
{
    return testing::Values(
        SgemmTestParams(32, 32, 32, "Default"), SgemmTestParams(17, 19, 23, "Default"),
        SgemmTestParams(60, 61, 127, "Default"),
        SgemmTestParams(32, 32, 32, "AB1_AddC").set_beta(1.0f).set_c(true),
        SgemmTestParams(31, 35, 63, "AB1_AddC").set_beta(1.0f).set_c(true),
        SgemmTestParams(32, 32, 32, "A1_B2p5").set_beta(2.5f).set_c(true),
        SgemmTestParams(32, 32, 32, "A2p5_B0").set_alpha(2.5f),
        SgemmTestParams(32, 32, 32, "A2p5_B1").set_alpha(2.5f).set_beta(1.0f).set_c(true),
        SgemmTestParams(60, 61, 127, "GenAB").set_alpha(-1.5f).set_beta(0.75f).set_c(true),
        SgemmTestParams(15, 17, 256, "TileK64")
            .set_tile(0, 0, 64)
            .set_alpha(2.0f)
            .set_beta(0.5f)
            .set_c(true),
        SgemmTestParams(31, 35, 63, "OrderKMN")
            .set_loop_order(LoopDim::K, LoopDim::M, LoopDim::N)
            .set_alpha(1.5f)
            .set_beta(0.5f)
            .set_c(true));
}

// ============================================================================
// SGEMM A=RowMajor, B=RowMajor, Tile=8x3vl
// ============================================================================

static NonPackedSgemmFuncs<float> make_sgemm_f32_f32_f32_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_sgemm_f32_f32_f32_8x3vl_rvv,   tqt_get_n_step_sgemm_f32_f32_f32_8x3vl_rvv,
        tqt_get_a_offset_sgemm_f32_f32_f32_8x3vl_rvv, tqt_get_b_offset_sgemm_f32_f32_f32_8x3vl_rvv,
        tqt_get_c_offset_sgemm_f32_f32_f32_8x3vl_rvv, tqt_get_d_offset_sgemm_f32_f32_f32_8x3vl_rvv,
        tqt_get_d_size_sgemm_f32_f32_f32_8x3vl_rvv,   tqt_run_sgemm_f32_f32_f32_8x3vl_rvv,
    };
}

class Test_sgemm_f32_f32_f32_8x3vl_rvv : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_sgemm_f32_f32_f32_8x3vl_rvv, Correctness)
{
    auto funcs = make_sgemm_f32_f32_f32_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_test<float, float, ALayout::kRowMajor, BLayout::kRowMajor>(GetParam(),
                                                                                   funcs);
}

INSTANTIATE_TEST_SUITE_P(SgemmTests, Test_sgemm_f32_f32_f32_8x3vl_rvv, sgemm_test_values(),
                         SgemmTestParamNameGenerator());

// ============================================================================
// SGEMM A=RowMajor, B=Transposed, Tile=8x3vl
// ============================================================================

static NonPackedSgemmFuncs<float> make_sgemm_f32_f32_f32t_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_sgemm_f32_f32_f32t_8x3vl_rvv,
        tqt_get_n_step_sgemm_f32_f32_f32t_8x3vl_rvv,
        tqt_get_a_offset_sgemm_f32_f32_f32t_8x3vl_rvv,
        tqt_get_b_offset_sgemm_f32_f32_f32t_8x3vl_rvv,
        tqt_get_c_offset_sgemm_f32_f32_f32t_8x3vl_rvv,
        tqt_get_d_offset_sgemm_f32_f32_f32t_8x3vl_rvv,
        tqt_get_d_size_sgemm_f32_f32_f32t_8x3vl_rvv,
        tqt_run_sgemm_f32_f32_f32t_8x3vl_rvv,
    };
}

class Test_sgemm_f32_f32_f32t_8x3vl_rvv : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_sgemm_f32_f32_f32t_8x3vl_rvv, Correctness)
{
    auto funcs = make_sgemm_f32_f32_f32t_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_test<float, float, ALayout::kRowMajor, BLayout::kTransposed>(GetParam(),
                                                                                     funcs);
}

INSTANTIATE_TEST_SUITE_P(SgemmTests, Test_sgemm_f32_f32_f32t_8x3vl_rvv, sgemm_test_values(),
                         SgemmTestParamNameGenerator());

// ============================================================================
// SGEMM A=Transposed, B=RowMajor, Tile=8x3vl
// ============================================================================

static NonPackedSgemmFuncs<float> make_sgemm_f32_f32t_f32_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_sgemm_f32_f32t_f32_8x3vl_rvv,
        tqt_get_n_step_sgemm_f32_f32t_f32_8x3vl_rvv,
        tqt_get_a_offset_sgemm_f32_f32t_f32_8x3vl_rvv,
        tqt_get_b_offset_sgemm_f32_f32t_f32_8x3vl_rvv,
        tqt_get_c_offset_sgemm_f32_f32t_f32_8x3vl_rvv,
        tqt_get_d_offset_sgemm_f32_f32t_f32_8x3vl_rvv,
        tqt_get_d_size_sgemm_f32_f32t_f32_8x3vl_rvv,
        tqt_run_sgemm_f32_f32t_f32_8x3vl_rvv,
    };
}

class Test_sgemm_f32_f32t_f32_8x3vl_rvv : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_sgemm_f32_f32t_f32_8x3vl_rvv, Correctness)
{
    auto funcs = make_sgemm_f32_f32t_f32_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_test<float, float, ALayout::kTransposed, BLayout::kRowMajor>(GetParam(),
                                                                                     funcs);
}

INSTANTIATE_TEST_SUITE_P(SgemmTests, Test_sgemm_f32_f32t_f32_8x3vl_rvv, sgemm_test_values(),
                         SgemmTestParamNameGenerator());

// ============================================================================
// SGEMM A=Transposed, B=Transposed, Tile=8x3vl
// ============================================================================

static NonPackedSgemmFuncs<float> make_sgemm_f32_f32t_f32t_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_sgemm_f32_f32t_f32t_8x3vl_rvv,
        tqt_get_n_step_sgemm_f32_f32t_f32t_8x3vl_rvv,
        tqt_get_a_offset_sgemm_f32_f32t_f32t_8x3vl_rvv,
        tqt_get_b_offset_sgemm_f32_f32t_f32t_8x3vl_rvv,
        tqt_get_c_offset_sgemm_f32_f32t_f32t_8x3vl_rvv,
        tqt_get_d_offset_sgemm_f32_f32t_f32t_8x3vl_rvv,
        tqt_get_d_size_sgemm_f32_f32t_f32t_8x3vl_rvv,
        tqt_run_sgemm_f32_f32t_f32t_8x3vl_rvv,
    };
}

class Test_sgemm_f32_f32t_f32t_8x3vl_rvv : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_sgemm_f32_f32t_f32t_8x3vl_rvv, Correctness)
{
    auto funcs = make_sgemm_f32_f32t_f32t_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_test<float, float, ALayout::kTransposed, BLayout::kTransposed>(GetParam(),
                                                                                       funcs);
}

INSTANTIATE_TEST_SUITE_P(SgemmTests, Test_sgemm_f32_f32t_f32t_8x3vl_rvv, sgemm_test_values(),
                         SgemmTestParamNameGenerator());

// ============================================================================
// SGEMM C==D alias tests (all 4 layout variants)
// ============================================================================

static auto sgemm_alias_test_values()
{
    return testing::Values(
        SgemmTestParams(32, 32, 32, "Alias_B1").set_alpha(1.0f).set_beta(1.0f).set_c(true),
        SgemmTestParams(17, 19, 23, "Alias_B1").set_alpha(1.0f).set_beta(1.0f).set_c(true),
        SgemmTestParams(60, 61, 127, "Alias_GenAB").set_alpha(-1.5f).set_beta(0.75f).set_c(true),
        SgemmTestParams(32, 32, 32, "Alias_A2p5_B2p5").set_alpha(2.5f).set_beta(2.5f).set_c(true));
}

class Test_sgemm_f32_f32_f32_8x3vl_rvv_Alias : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_sgemm_f32_f32_f32_8x3vl_rvv_Alias, CDAlias)
{
    auto funcs = make_sgemm_f32_f32_f32_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_alias_test<float, float, ALayout::kRowMajor, BLayout::kRowMajor>(GetParam(),
                                                                                         funcs);
}

INSTANTIATE_TEST_SUITE_P(SgemmAliasTests, Test_sgemm_f32_f32_f32_8x3vl_rvv_Alias,
                         sgemm_alias_test_values(), SgemmTestParamNameGenerator());

class Test_sgemm_f32_f32_f32t_8x3vl_rvv_Alias : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_sgemm_f32_f32_f32t_8x3vl_rvv_Alias, CDAlias)
{
    auto funcs = make_sgemm_f32_f32_f32t_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_alias_test<float, float, ALayout::kRowMajor, BLayout::kTransposed>(
        GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(SgemmAliasTests, Test_sgemm_f32_f32_f32t_8x3vl_rvv_Alias,
                         sgemm_alias_test_values(), SgemmTestParamNameGenerator());

class Test_sgemm_f32_f32t_f32_8x3vl_rvv_Alias : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_sgemm_f32_f32t_f32_8x3vl_rvv_Alias, CDAlias)
{
    auto funcs = make_sgemm_f32_f32t_f32_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_alias_test<float, float, ALayout::kTransposed, BLayout::kRowMajor>(
        GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(SgemmAliasTests, Test_sgemm_f32_f32t_f32_8x3vl_rvv_Alias,
                         sgemm_alias_test_values(), SgemmTestParamNameGenerator());

class Test_sgemm_f32_f32t_f32t_8x3vl_rvv_Alias : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_sgemm_f32_f32t_f32t_8x3vl_rvv_Alias, CDAlias)
{
    auto funcs = make_sgemm_f32_f32t_f32t_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_alias_test<float, float, ALayout::kTransposed, BLayout::kTransposed>(
        GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(SgemmAliasTests, Test_sgemm_f32_f32t_f32t_8x3vl_rvv_Alias,
                         sgemm_alias_test_values(), SgemmTestParamNameGenerator());
