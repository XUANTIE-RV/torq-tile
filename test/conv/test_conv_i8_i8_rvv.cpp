//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/conv/conv_i8_i8/convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.h"
#include "src/conv/conv_i8_i8/convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p/tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv.h"
#include "src/conv/conv_i8_i8/convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.h"
#include "src/conv/conv_i8_i8/convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp/tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv.h"
#include "test/common/quantized_conv_test_runner.h"

using namespace torq_tile::test;

// ============================================================================
// Common test cases for int8 conv
// ============================================================================

static auto i8_common_conv_test_values()
{
    return testing::Values(
        // Basic 2D convolutions
        ConvTestParams(3, 16, 1, 8, 8, 3, 3), ConvTestParams(16, 32, 1, 8, 8, 3, 3),
        ConvTestParams(16, 32, 1, 8, 8, 1, 1),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Pad1").set_pad(1, 1),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Stride2").set_stride(2, 2),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Dilation2").set_dilation(2, 2).set_pad(2, 2),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "PadStride").set_pad(1, 1).set_stride(2, 2),
        // Non-square
        ConvTestParams(8, 16, 1, 7, 11, 3, 5),
        // 1x1 conv
        ConvTestParams(64, 128, 1, 4, 4, 1, 1),
        // Larger
        ConvTestParams(3, 32, 1, 16, 16, 3, 3, "Large").set_pad(1, 1),
        // Small IC/OC
        ConvTestParams(3, 8, 1, 6, 6, 3, 3),
        // Larger IC
        ConvTestParams(64, 32, 1, 4, 4, 3, 3, "LargeIC").set_pad(1, 1));
}

// ============================================================================
// ConvCF NonPacked (mx1bias, reqi32toi8, qai8_qsi8dx_qai8) - RVV 4x1vl
// ============================================================================

static NonPackedRequantizedConvCfFuncs<tqt_qai8_qsi8dx_qai8_params>
make_convcf_nonpacked_reqi32toi8_funcs()
{
    return {
        tqt_get_m_step_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_n_step_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_b_im2col_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_run_im2col_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_run_conv_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    };
}

class Test_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv, Correctness)
{
    run_nonpacked_convcf_reqi32toi8_test(GetParam(), make_convcf_nonpacked_reqi32toi8_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
                         i8_common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCL NonPacked (1xnbias, reqi32toi8, qai8_qai8_qsi8cxt) - RVV 4x1vl
// ============================================================================

static NonPackedRequantizedConvClFuncs<tqt_qai8_qai8_qsi8cx_params>
make_convcl_nonpacked_reqi32toi8_funcs()
{
    return {
        tqt_get_m_step_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_n_step_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_a_im2col_size_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_run_im2col_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_run_conv_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
    };
}

class Test_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv, Correctness)
{
    run_nonpacked_convcl_reqi32toi8_test(GetParam(), make_convcl_nonpacked_reqi32toi8_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests,
                         Test_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
                         i8_common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCF Packed (mx1bias, reqi32toi8, qai8_qsi8dxp_qai8p) - RVV 4x1vl
// ============================================================================

static PackedRequantizedConvCfFuncs<tqt_qai8_qsi8dx_qai8_params>
make_convcf_packed_reqi32toi8_funcs()
{
    return {
        tqt_get_m_step_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_n_step_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_a_packed_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_b_im2colpack_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_a_pack_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_b_im2colpack_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_conv_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
    };
}

class Test_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv, Correctness)
{
    run_packed_convcf_reqi32toi8_test(GetParam(), make_convcf_packed_reqi32toi8_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests,
                         Test_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
                         i8_common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCL Packed (1xnbias, reqi32toi8, qai8_qai8p_qsi8cxp) - RVV 4x1vl
// ============================================================================

static PackedRequantizedConvClFuncs<tqt_qai8_qai8_qsi8cx_params>
make_convcl_packed_reqi32toi8_funcs()
{
    return {
        tqt_get_m_step_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_a_im2colpack_size_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_a_im2colpack_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_conv_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    };
}

class Test_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv, Correctness)
{
    run_packed_convcl_reqi32toi8_test(GetParam(), make_convcl_packed_reqi32toi8_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests,
                         Test_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
                         i8_common_conv_test_values(), ConvTestParamNameGenerator());
