//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/conv/conv_f32_f32/convcf_mx1bias_clamp_f32_f32_f32/tqt_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv.h"
#include "src/conv/conv_f32_f32/convcf_mx1bias_clamp_f32_f32p_f32p/tqt_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv.h"
#include "src/conv/conv_f32_f32/convcl_1xnbias_clamp_f32_f32_f32t/tqt_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.h"
#include "src/conv/conv_f32_f32/convcl_1xnbias_clamp_f32_f32p_f32p/tqt_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.h"
#include "test/common/conv_test_runner.h"

using namespace torq_tile::test;

// ============================================================================
// Common test cases
// ============================================================================

static auto common_conv_test_values()
{
    return testing::Values(
        // Basic 2D convolutions
        ConvTestParams(3, 16, 1, 8, 8, 3, 3),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "NoBias").set_bias(false),
        ConvTestParams(16, 32, 1, 8, 8, 3, 3), ConvTestParams(16, 32, 1, 8, 8, 1, 1),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Pad1").set_pad(1, 1),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Stride2").set_stride(2, 2),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Dilation2").set_dilation(2, 2).set_pad(2, 2),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Clamp").set_clamp(-0.5f, 0.5f),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "PadClampBias").set_pad(1, 1).set_clamp(-0.5f, 0.5f),
        // Non-square
        ConvTestParams(8, 16, 1, 7, 11, 3, 5),
        // 1x1 conv
        ConvTestParams(64, 128, 1, 4, 4, 1, 1),
        // Larger
        ConvTestParams(3, 32, 1, 16, 16, 3, 3, "Large").set_pad(1, 1));
}

// ============================================================================
// ConvCF NonPacked (mx1bias, f32_f32_f32)
// ============================================================================

static NonPackedConvCfFuncs<float> make_convcf_nonpacked_funcs()
{
    return {
        tqt_get_m_step_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_n_step_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_get_b_im2col_size_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_run_im2col_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_run_conv_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    };
}

class Test_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv, Correctness)
{
    run_nonpacked_convcf_test<float, float>(GetParam(), make_convcf_nonpacked_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCL NonPacked (1xnbias, f32_f32_f32t)
// ============================================================================

static NonPackedConvClFuncs<float> make_convcl_nonpacked_funcs()
{
    return {
        tqt_get_m_step_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_n_step_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_a_im2col_size_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_run_im2col_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_run_conv_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
    };
}

class Test_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv, Correctness)
{
    run_nonpacked_convcl_test<float, float>(GetParam(), make_convcl_nonpacked_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCF Packed (mx1bias, f32_f32p_f32p)
// ============================================================================

static PackedConvCfFuncs<float> make_convcf_packed_funcs()
{
    return {
        tqt_get_m_step_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_n_step_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_a_packed_size_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_b_im2colpack_size_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_a_pack_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_b_im2colpack_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_conv_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
    };
}

class Test_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv, Correctness)
{
    run_packed_convcf_test<float, float>(GetParam(), make_convcf_packed_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCL Packed (1xnbias, f32_f32p_f32p)
// ============================================================================

static PackedConvClFuncs<float> make_convcl_packed_funcs()
{
    return {
        tqt_get_m_step_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_n_step_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_a_im2colpack_size_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_b_packed_size_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_bt_pack_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_a_im2colpack_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_conv_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    };
}

class Test_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv, Correctness)
{
    run_packed_convcl_test<float, float>(GetParam(), make_convcl_packed_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());
