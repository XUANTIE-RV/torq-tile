//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/gemm/gemm_f32_f16/gemm_1xnbias_clamp_f32_f32_f16t/tqt_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv.h"
#include "src/gemm/gemm_f32_f16/gemm_1xnbias_clamp_f32_f32p_f16p/tqt_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/gemm_test_runner.h"

using namespace torq_tile::test;

static NonPackedGemmFuncs<float> make_direct_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

static PackedGemmFuncs<float> make_packed_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv, float, 13, 11, 12, 14,
                       ClampSource::kColumnBias, false, 5, true>(),
    };
}

static auto test_values()
{
    return testing::Values(
        GemmTestParams(1, 1, 1), GemmTestParams(7, 3, 5), GemmTestParams(9, 19, 17),
        GemmTestParams(17, 35, 63, "WithC_Bias_Clamp")
            .set_c(true)
            .set_bias(true)
            .set_clamp(-3.0f, 4.0f),
        GemmTestParams(15, 17, 31, "SplitK").set_c(true).set_bias(true).set_tile(0, 0, 7));
}

class DirectF32F16RvvTest : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(DirectF32F16RvvTest, Correctness)
{
    run_nonpacked_gemm_test<float, float, BLayout::kTransposed, BiasMode::k1xN, float, float16_t>(
        GetParam(), make_direct_funcs(), 1.0e-3f, 0.99999f);
}

TEST_P(DirectF32F16RvvTest, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_nonpacked_gemm_test<float, float, BLayout::kTransposed, BiasMode::k1xN, float, float16_t>(
        GetParam(), make_direct_funcs(), 1.0e-3f, 0.99999f);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, DirectF32F16RvvTest,
    with_clamp_dimensions(test_values(), tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

class PackedF32F16RvvTest : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(PackedF32F16RvvTest, Correctness)
{
    run_packed_gemm_test<float, float, BiasMode::k1xN, float, float16_t>(
        GetParam(), make_packed_funcs(), 1.0e-3f, 0.99999f);
}

TEST_P(PackedF32F16RvvTest, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_packed_gemm_test<float, float, BiasMode::k1xN, float, float16_t>(
        GetParam(), make_packed_funcs(), 1.0e-3f, 0.99999f);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, PackedF32F16RvvTest,
    with_clamp_dimensions(test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

TEST(GemmF32F16RvvEdgeCases, ZeroKNullInputsAndInPlaceC)
{
    run_gemm_zero_k_test<float, float>(9, 19, make_direct_funcs());
    run_gemm_zero_k_test<float, float>(9, 19, make_packed_funcs());
}

TEST(GemmF32F16RvvEdgeCases, ClampSpecialValues_ZeroKNullInputsAndInPlaceC)
{
    ScopedGemmClampProbe probe;
    run_gemm_zero_k_test<float, float>(9, 19, make_direct_funcs());
    run_gemm_zero_k_test<float, float>(9, 19, make_packed_funcs());
}

TEST(GemmF32F16RvvEdgeCases, FullBufferSplitKWithNonzeroOffsets)
{
    const auto funcs = make_packed_funcs();
    run_packed_gemm_full_buffer_split_k_test<float, float, BiasMode::k1xN, float, float16_t>(
        funcs.get_m_step() + 3, funcs.get_n_step() + 2, 19, 7, funcs, 1.0e-3f, 0.99999f);
}

TEST(GemmF32F16RvvEdgeCases, ClampSpecialValues_FullBufferSplitKWithNonzeroOffsets)
{
    ScopedGemmClampProbe probe;
    const auto funcs = make_packed_funcs();
    run_packed_gemm_full_buffer_split_k_test<float, float, BiasMode::k1xN, float, float16_t>(
        funcs.get_m_step() + 3, funcs.get_n_step() + 2, 19, 7, funcs, 1.0e-3f, 0.99999f);
}
