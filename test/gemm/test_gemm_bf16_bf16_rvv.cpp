//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cfloat>

#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16p/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16t/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16p_bf16p/tqt_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv.h"
#include "torq_tile/tqt_compile_config.h"
#if TORQ_TILE_HAS_RVV_BF16_TO_F16
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbiasf32_clamp_f16_bf16_bf16/tqt_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbiasf32_clamp_f16_bf16_bf16t/tqt_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbiasf32_clamp_f16_bf16p_bf16p/tqt_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv.h"
#endif
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16_bf16/tqt_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16_bf16t/tqt_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16p_bf16p/tqt_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_mx1bias_clamp_bf16_bf16_bf16/tqt_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_mx1bias_clamp_bf16_bf16p_bf16p/tqt_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv.h"
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
// With 1xN bias, With clamp, NonPacked, B RowMajor, Tile=8x1vl
// ============================================================================

static NonPackedGemmFuncs<bfloat16_t> make_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv, bfloat16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, bfloat16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, bfloat16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                        funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, NonPacked, B Transposed, Tile=8x1vl
// ============================================================================

static NonPackedGemmFuncs<bfloat16_t> make_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv, bfloat16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, bfloat16_t, BLayout::kTransposed, BiasMode::k1xN>(
        GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, bfloat16_t, BLayout::kTransposed, BiasMode::k1xN>(
        GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, Packed, Tile=8x1vl
// ============================================================================

static PackedGemmFuncs<bfloat16_t> make_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv, bfloat16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, bfloat16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, bfloat16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With 1xN bias, With clamp, B-Only-Packed, Tile=8x1vl
// ============================================================================

static BOnlyPackedGemmFuncs<bfloat16_t> make_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv, bfloat16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv_funcs();
    run_b_only_packed_gemm_test<bfloat16_t, bfloat16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv_funcs();
    run_b_only_packed_gemm_test<bfloat16_t, bfloat16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

#if TORQ_TILE_HAS_RVV_BF16_TO_F16
// ============================================================================
// D=f16, With 1xN bias, With clamp, NonPacked, B RowMajor, Tile=8x1vl
// ============================================================================

static NonPackedGemmFuncs<float16_t, float>
make_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv, float16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true, float>(),
    };
}

class Test_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                       funcs);
}

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                       funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// D=f16, With 1xN bias, With clamp, NonPacked, B Transposed, Tile=8x1vl
// ============================================================================

static NonPackedGemmFuncs<float16_t, float>
make_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true, float>(),
    };
}

class Test_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                         funcs);
}

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                         funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// D=f16, With 1xN bias, With clamp, Packed, Tile=8x1vl
// ============================================================================

static PackedGemmFuncs<float16_t, float>
make_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_b_pack_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true, float>(),
    };
}

class Test_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

#endif  // TORQ_TILE_HAS_RVV_BF16_TO_F16

// ============================================================================
// D=f32, With 1xN bias, With clamp, NonPacked, B RowMajor, Tile=8x1vl
// ============================================================================

static NonPackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                   funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                   funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// D=f32, With 1xN bias, With clamp, NonPacked, B Transposed, Tile=8x1vl
// ============================================================================

static NonPackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                     funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                     funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// D=f32, With 1xN bias, With clamp, Packed, Tile=8x1vl
// ============================================================================

static PackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, float, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With Mx1 bias, With clamp, NonPacked, B Plain, Tile=8x1vl
// ============================================================================

static NonPackedGemmFuncs<bfloat16_t> make_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_a_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_b_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_c_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_bias_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_size_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv, bfloat16_t, 13, 11, 12,
                       14, ClampSource::kRowBias, false, 5, true>(),
    };
}

class Test_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, bfloat16_t, BLayout::kRowMajor, BiasMode::kMx1>(GetParam(),
                                                                                        funcs);
}

TEST_P(Test_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, bfloat16_t, BLayout::kRowMajor, BiasMode::kMx1>(GetParam(),
                                                                                        funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(),
                          tqt_get_n_step_gemm_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ============================================================================
// With Mx1 bias, With clamp, Packed, Tile=8x1vl
// ============================================================================

static PackedGemmFuncs<bfloat16_t> make_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_c_offset_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_bias_offset_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_size_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_b_pack_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv, bfloat16_t, 13, 11,
                       12, 14, ClampSource::kRowBias, false, 5, true>(),
    };
}

class Test_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, bfloat16_t, BiasMode::kMx1>(GetParam(), funcs);
}

TEST_P(Test_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, bfloat16_t, BiasMode::kMx1>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv(),
                          tqt_get_n_step_gemm_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

#if TORQ_TILE_HAS_RVV_BF16_TO_F16

TEST(F32BiasContract, gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv)
{
    const size_t ms = tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv();
    const size_t m = ms + 1, n = ns + 1, full_m = ms + m, full_n = ns + n;
    const size_t full_k = 96, ki = 32, k = 32, ld = 128, bl = 16, lds = 8;
    std::vector<bfloat16_t> a(full_m * ld, static_cast<bfloat16_t>(0));
    std::vector<bfloat16_t> b(full_k * full_n, static_cast<bfloat16_t>(0));
    for (size_t i = ms; i < full_m; ++i) {
        for (size_t p = ki; p < ki + k; ++p)
            a[i * ld + p] = 4;
    }
    for (size_t j = ns; j < full_n; ++j) {
        for (size_t p = ki; p < ki + k; ++p)
            b[p * full_n + j] = 512;
    }
    const void *aw = reinterpret_cast<const uint8_t *>(a.data()) +
                     tqt_get_a_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv(ms, ki, ld);
    const void *bw = reinterpret_cast<const uint8_t *>(b.data()) +
                     tqt_get_b_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv(ns, ki, full_n);
    EXPECT_EQ(tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv(ns),
              ns * sizeof(float));
    check_f32_bias_epilogue<float16_t>(
        m, n, 65536.0f, true,
        [&](const float16_t *c, size_t stride, const float *bias, float16_t *d, float16_t lo,
            float16_t hi) {
            std::vector<float> full_bias(full_n, 999.0f);
            if (bias)
                std::copy(bias, bias + n, full_bias.begin() + ns);
            const float *bw_bias =
                bias ? full_bias.data() +
                           tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv(ns) /
                               sizeof(float)
                     : nullptr;
            tqt_run_gemm_1xnbiasf32_clamp_f16_bf16_bf16_8x1vl_rvv(
                m, n, k, aw, ld, ki, bw, full_n, ki, c, stride, d, stride, bw_bias, lo, hi);
        });
}
#endif

#if TORQ_TILE_HAS_RVV_BF16_TO_F16

TEST(F32BiasContract, gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv)
{
    const size_t ms = tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv();
    const size_t m = ms + 1, n = ns + 1, full_m = ms + m, full_n = ns + n;
    const size_t full_k = 96, ki = 32, k = 32, ld = 128, bl = 16, lds = 8;
    std::vector<bfloat16_t> a(full_m * ld, static_cast<bfloat16_t>(0));
    std::vector<bfloat16_t> b(full_n * ld, static_cast<bfloat16_t>(0));
    for (size_t i = ms; i < full_m; ++i) {
        for (size_t p = ki; p < ki + k; ++p)
            a[i * ld + p] = 4;
    }
    for (size_t j = ns; j < full_n; ++j) {
        for (size_t p = ki; p < ki + k; ++p)
            b[j * ld + p] = 512;
    }
    const void *aw = reinterpret_cast<const uint8_t *>(a.data()) +
                     tqt_get_a_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv(ms, ki, ld);
    const void *bw = reinterpret_cast<const uint8_t *>(b.data()) +
                     tqt_get_b_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv(ns, ki, ld);
    EXPECT_EQ(tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv(ns),
              ns * sizeof(float));
    check_f32_bias_epilogue<float16_t>(
        m, n, 65536.0f, true,
        [&](const float16_t *c, size_t stride, const float *bias, float16_t *d, float16_t lo,
            float16_t hi) {
            std::vector<float> full_bias(full_n, 999.0f);
            if (bias)
                std::copy(bias, bias + n, full_bias.begin() + ns);
            const float *bw_bias =
                bias ? full_bias.data() +
                           tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv(ns) /
                               sizeof(float)
                     : nullptr;
            tqt_run_gemm_1xnbiasf32_clamp_f16_bf16_bf16t_8x1vl_rvv(
                m, n, k, aw, ld, ki, bw, ld, ki, c, stride, d, stride, bw_bias, lo, hi);
        });
}
#endif

#if TORQ_TILE_HAS_RVV_BF16_TO_F16

TEST(F32BiasContract, gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv)
{
    const size_t ms = tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv();
    const size_t m = ms + 1, n = ns + 1, full_m = ms + m, full_n = ns + n;
    const size_t full_k = 96, ki = 32, k = 32, ld = 128, bl = 16, lds = 8;
    std::vector<bfloat16_t> a(full_m * ld, static_cast<bfloat16_t>(0));
    std::vector<bfloat16_t> b(full_n * ld, static_cast<bfloat16_t>(0));
    for (size_t i = ms; i < full_m; ++i) {
        for (size_t p = ki; p < ki + k; ++p)
            a[i * ld + p] = 4;
    }
    for (size_t j = ns; j < full_n; ++j) {
        for (size_t p = ki; p < ki + k; ++p)
            b[j * ld + p] = 512;
    }
    std::vector<uint8_t> ap(
        tqt_get_a_packed_size_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(full_m, full_k));
    std::vector<uint8_t> bp(
        tqt_get_b_packed_size_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(full_n, full_k));
    tqt_run_a_pack_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(full_m, full_k, ld, full_k, 0,
                                                                   a.data(), ap.data());
    tqt_run_bt_pack_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(full_n, full_k, ld, full_k, 0,
                                                                    b.data(), bp.data());
    const void *aw =
        ap.data() +
        tqt_get_a_packed_offset_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(ms, ki, full_k, m);
    const void *bw =
        bp.data() +
        tqt_get_b_packed_offset_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(ns, ki, full_k, n);
    EXPECT_EQ(tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(ns),
              ns * sizeof(float));
    check_f32_bias_epilogue<float16_t>(
        m, n, 65536.0f, true,
        [&](const float16_t *c, size_t stride, const float *bias, float16_t *d, float16_t lo,
            float16_t hi) {
            std::vector<float> full_bias(full_n, 999.0f);
            if (bias)
                std::copy(bias, bias + n, full_bias.begin() + ns);
            const float *bw_bias =
                bias ? full_bias.data() +
                           tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(ns) /
                               sizeof(float)
                     : nullptr;
            tqt_run_gemm_1xnbiasf32_clamp_f16_bf16p_bf16p_8x1vl_rvv(
                m, n, k, aw, full_k, ki, bw, full_k, ki, c, stride, d, stride, bw_bias, lo, hi);
        });
}
#endif

#if TORQ_TILE_HAS_RVV_BF16_TO_F16

// Original narrow-bias entry points remain covered after adding biasf32 variants.
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f16_bf16_bf16/tqt_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f16_bf16_bf16t/tqt_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f16_bf16p_bf16p/tqt_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv.h"

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv, float16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                       funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(GetParam(),
                                                                                       funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv, float16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                         funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv_funcs();
    run_nonpacked_gemm_test<bfloat16_t, float16_t, BLayout::kTransposed, BiasMode::k1xN>(GetParam(),
                                                                                         funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

static PackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv, float16_t, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv_funcs();
    run_packed_gemm_test<bfloat16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
    with_clamp_dimensions(common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());
#endif
