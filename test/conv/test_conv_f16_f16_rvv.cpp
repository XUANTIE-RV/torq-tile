//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16_f16/tqt_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv.h"
#include "src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16_f16/tqt_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv.h"
#include "src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16p_f16p/tqt_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv.h"
#include "src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16p_f16p/tqt_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv.h"
#include "src/conv/conv_f16_f16/convcl_1xnbias_clamp_f16_f16_f16p/tqt_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.h"
#include "src/conv/conv_f16_f16/convcl_1xnbias_clamp_f16_f16_f16t/tqt_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv.h"
#include "src/conv/conv_f16_f16/convcl_1xnbias_clamp_f16_f16_f16t/tqt_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h"
#include "src/conv/conv_f16_f16/convcl_1xnbias_clamp_f16_f16p_f16p/tqt_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv.h"
#include "src/conv/conv_f16_f16/convcl_1xnbias_clamp_f16_f16p_f16p/tqt_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16p_f16p/tqt_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv.h"
#include "test/common/conv_test_runner.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/gemm_test_runner.h"

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
// ConvCF NonPacked (mx1bias, f16_f16_f16)
// ============================================================================

static NonPackedConvCfFuncs<float16_t> make_convcf_nonpacked_funcs()
{
    return {
        tqt_get_m_step_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_n_step_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_b_im2col_size_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_run_im2col_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        clamp_test_run<tqt_run_conv_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kRowBias>(),
    };
}

class Test_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv, Correctness)
{
    run_nonpacked_convcf_test<float16_t, float16_t>(GetParam(), make_convcf_nonpacked_funcs());
}

TEST_P(Test_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_nonpacked_convcf_test<float16_t, float16_t>(GetParam(), make_convcf_nonpacked_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCL NonPacked (1xnbias, f16_f16_f16t)
// ============================================================================

static NonPackedConvClFuncs<float16_t> make_convcl_nonpacked_funcs()
{
    return {
        tqt_get_m_step_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_n_step_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_a_im2col_size_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_run_im2col_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        clamp_test_run<tqt_run_conv_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias>(),
    };
}

class Test_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv, Correctness)
{
    run_nonpacked_convcl_test<float16_t, float16_t>(GetParam(), make_convcl_nonpacked_funcs());
}

TEST_P(Test_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_nonpacked_convcl_test<float16_t, float16_t>(GetParam(), make_convcl_nonpacked_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCF Packed (mx1bias, f16_f16p_f16p)
// ============================================================================

static PackedConvCfFuncs<float16_t> make_convcf_packed_funcs()
{
    return {
        tqt_get_m_step_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_n_step_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_packed_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_im2colpack_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_a_pack_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_b_im2colpack_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        clamp_test_run<tqt_run_conv_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kRowBias>(),
    };
}

class Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv, Correctness)
{
    run_packed_convcf_test<float16_t, float16_t>(GetParam(), make_convcf_packed_funcs());
}

TEST_P(Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_packed_convcf_test<float16_t, float16_t>(GetParam(), make_convcf_packed_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCL Packed (1xnbias, f16_f16p_f16p)
// ============================================================================

static PackedConvClFuncs<float16_t> make_convcl_packed_funcs()
{
    return {
        tqt_get_m_step_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_n_step_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_im2colpack_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_bt_pack_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_a_im2colpack_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        clamp_test_run<tqt_run_conv_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias>(),
    };
}

class Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv, Correctness)
{
    run_packed_convcl_test<float16_t, float16_t>(GetParam(), make_convcl_packed_funcs());
}

TEST_P(Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_packed_convcl_test<float16_t, float16_t>(GetParam(), make_convcl_packed_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// ConvCL MixedPacked (1xnbias, f16_f16_f16p) — A im2col only, B packed
// ============================================================================

static MixedPackedConvClFuncs<float16_t> make_convcl_mixed_packed_funcs()
{
    return {
        tqt_get_m_step_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_n_step_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_a_im2col_size_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_b_packed_size_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_run_bt_pack_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_run_im2col_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        clamp_test_run<tqt_run_conv_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias>(),
    };
}

class Test_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv, Correctness)
{
    run_mixed_packed_convcl_test<float16_t, float16_t>(GetParam(),
                                                       make_convcl_mixed_packed_funcs());
}

TEST_P(Test_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_mixed_packed_convcl_test<float16_t, float16_t>(GetParam(),
                                                       make_convcl_mixed_packed_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

// FP16 convolution with FP32 accumulation.

static NonPackedConvCfFuncs<float16_t> make_convcf_nonpacked_f32acc_funcs()
{
    return {
        tqt_get_m_step_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_n_step_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_get_b_im2col_size_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        tqt_run_im2col_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
        clamp_test_run<tqt_run_conv_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv, float16_t,
                       13, 11, 12, 14, ClampSource::kRowBias>(),
    };
}

class Test_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv, Correctness)
{
    run_nonpacked_convcf_test<float16_t, float16_t>(GetParam(),
                                                    make_convcf_nonpacked_f32acc_funcs());
}

TEST_P(Test_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_nonpacked_convcf_test<float16_t, float16_t>(GetParam(),
                                                    make_convcf_nonpacked_f32acc_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

static PackedConvCfFuncs<float16_t> make_convcf_packed_f32acc_funcs()
{
    return {
        tqt_get_m_step_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_n_step_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_a_packed_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_b_im2colpack_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_a_pack_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_b_im2colpack_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        clamp_test_run<tqt_run_conv_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, float16_t,
                       13, 11, 12, 14, ClampSource::kRowBias>(),
    };
}

class Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, Correctness)
{
    run_packed_convcf_test<float16_t, float16_t>(GetParam(), make_convcf_packed_f32acc_funcs());
}

TEST_P(Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_packed_convcf_test<float16_t, float16_t>(GetParam(), make_convcf_packed_f32acc_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

static NonPackedConvClFuncs<float16_t> make_convcl_nonpacked_f32acc_funcs()
{
    return {
        tqt_get_m_step_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_get_n_step_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_get_a_im2col_size_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        tqt_run_im2col_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
        clamp_test_run<tqt_run_conv_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv, float16_t,
                       13, 11, 12, 14, ClampSource::kColumnBias>(),
    };
}

class Test_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv, Correctness)
{
    run_nonpacked_convcl_test<float16_t, float16_t>(GetParam(),
                                                    make_convcl_nonpacked_f32acc_funcs());
}

TEST_P(Test_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_nonpacked_convcl_test<float16_t, float16_t>(GetParam(),
                                                    make_convcl_nonpacked_f32acc_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

static PackedConvClFuncs<float16_t> make_convcl_packed_f32acc_funcs()
{
    return {
        tqt_get_m_step_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_n_step_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_a_im2colpack_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_get_b_packed_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_bt_pack_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        tqt_run_a_im2colpack_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
        clamp_test_run<tqt_run_conv_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, float16_t,
                       13, 11, 12, 14, ClampSource::kColumnBias>(),
    };
}

class Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, Correctness)
{
    run_packed_convcl_test<float16_t, float16_t>(GetParam(), make_convcl_packed_f32acc_funcs());
}

TEST_P(Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_packed_convcl_test<float16_t, float16_t>(GetParam(), make_convcl_packed_f32acc_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                         common_conv_test_values(), ConvTestParamNameGenerator());

TEST(Test_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_Precision, AccumulationAndEpilogue)
{
    check_f16_f32acc_nonpacked(tqt_get_m_step_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(),
                               tqt_get_n_step_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(),
                               BLayout::kRowMajor, BiasMode::kMx1,
                               tqt_run_conv_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv);
}

TEST(Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_Precision, AccumulationAndEpilogue)
{
    check_f16_f32acc_packed(tqt_get_m_step_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                            tqt_get_n_step_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                            BiasMode::kMx1,
                            tqt_run_conv_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_a_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_b_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_bt_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv);
}

TEST(Test_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv_Precision, AccumulationAndEpilogue)
{
    check_f16_f32acc_nonpacked(tqt_get_m_step_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv(),
                               tqt_get_n_step_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv(),
                               BLayout::kTransposed, BiasMode::k1xN,
                               tqt_run_conv_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv);
}

TEST(Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_Precision, AccumulationAndEpilogue)
{
    check_f16_f32acc_packed(tqt_get_m_step_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                            tqt_get_n_step_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(),
                            BiasMode::k1xN,
                            tqt_run_conv_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv,
                            tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv);
}

// Compare fused im2col packing with the plain layout at a nonzero K origin,
// then validate both groups against an independent convolution reference.
template <bool kChannelLast, bool kPacked>
static void check_f32acc_grouped_conv()
{
    const size_t ms = 8;
    const size_t ns = tqt_get_n_step_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv();
    const size_t guard = 8, origin = kPacked ? 3 : 0;
    const float16_t sentinel = static_cast<float16_t>(-123);
    for (size_t stride : {1u, 2u}) {
        auto params = ConvTestParams(6, 2 * (ns + 3), 2, 5, ns + 5, 3, 3)
                          .set_pad(2, 2)
                          .set_dilation(2, 2)
                          .set_stride(stride, stride)
                          .to_conv_params();
        const size_t spatial = params.output_shape[1] * params.output_shape[2];
        const size_t oc = params.output_channel / params.groups;
        const size_t m = kChannelLast ? spatial : oc, n = kChannelLast ? oc : spatial;
        const size_t k = 3 * 3 * 3, kp = k + 7, ldd = n + 5;
        std::vector<float16_t> activation(params.input_channel * params.input_shape[1] *
                                          params.input_shape[2]);
        for (size_t i = 0; i < activation.size(); ++i)
            activation[i] = static_cast<float16_t>((int((i * 7) % 23) - 11) / 16.0f);
        for (size_t group = 0; group < params.groups; ++group) {
            SCOPED_TRACE(::testing::Message() << "group=" << group << " stride=" << stride);
            std::vector<float16_t> weight(oc * k), bias(oc), reference(m * n);
            for (size_t i = 0; i < weight.size(); ++i)
                weight[i] = static_cast<float16_t>((int((i * 3 + group) % 17) - 8) / 16.0f);
            for (size_t i = 0; i < bias.size(); ++i)
                bias[i] = static_cast<float16_t>((int((i + group) % 5) - 2) / 8.0f);
            const float16_t lo = static_cast<float16_t>(-0.75f), hi = static_cast<float16_t>(0.75f);
            reference_conv<float16_t>(
                group, activation.data(), weight.data(), bias.data(), reference.data(), &params,
                kChannelLast ? ConvLayout::kChannelLast : ConvLayout::kChannelFirst, lo, hi,
                ConvClampPolicy::kSkipUnbounded);
            std::vector<float16_t> plain((kChannelLast ? m : n) * k);
            if constexpr (kChannelLast) {
                EXPECT_EQ(
                    tqt_get_a_im2col_size_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv(m, k),
                    plain.size() * sizeof(float16_t));
                tqt_run_im2col_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv(
                    group, activation.data(), plain.data(), &params);
            } else {
                EXPECT_EQ(
                    tqt_get_b_im2col_size_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(k, n),
                    plain.size() * sizeof(float16_t));
                tqt_run_im2col_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv(
                    group, activation.data(), plain.data(), &params);
            }
            std::vector<float16_t> ap(2 * guard + m * kp, sentinel),
                bp(2 * guard + n * kp, sentinel);
            auto ap_ref = ap, bp_ref = bp;
            const size_t ao = origin * std::min(m, ms), bo = origin * std::min(n, ns);
            const float16_t *a = kChannelLast ? plain.data() : weight.data();
            const float16_t *b = kChannelLast ? weight.data() : plain.data();
            if constexpr (kPacked) {
                if constexpr (kChannelLast) {
                    EXPECT_EQ(
                        tqt_get_a_im2colpack_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                            m, kp),
                        m * kp * sizeof(float16_t));
                    EXPECT_EQ(
                        tqt_get_b_packed_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                            n, kp),
                        n * kp * sizeof(float16_t));
                    tqt_run_a_im2colpack_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                        group, activation.data(), kp, origin, ap.data() + guard + ao, &params);
                    tqt_run_bt_pack_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                        n, k, k, kp, origin, weight.data(), bp.data() + guard + bo);
                } else {
                    EXPECT_EQ(
                        tqt_get_a_packed_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                            m, kp),
                        m * kp * sizeof(float16_t));
                    EXPECT_EQ(
                        tqt_get_b_im2colpack_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                            n, kp),
                        n * kp * sizeof(float16_t));
                    tqt_run_a_pack_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                        m, k, k, kp, origin, weight.data(), ap.data() + guard + ao);
                    tqt_run_b_im2colpack_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                        group, activation.data(), kp, origin, bp.data() + guard + bo, &params);
                }
                for (size_t i = 0; i < m; i += ms) {
                    const size_t rows = std::min(ms, m - i);
                    size_t offset;
                    if constexpr (kChannelLast)
                        offset =
                            tqt_get_a_im2colpack_offset_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                                i, origin, kp, rows);
                    else
                        offset =
                            tqt_get_a_packed_offset_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                                i, origin, kp, rows);
                    EXPECT_EQ(offset, (i * kp + origin * rows) * sizeof(float16_t));
                    for (size_t p = 0; p < k; ++p)
                        for (size_t r = 0; r < rows; ++r)
                            ap_ref[guard + i * kp + (origin + p) * rows + r] = a[(i + r) * k + p];
                }
                for (size_t j = 0; j < n; j += ns) {
                    const size_t cols = std::min(ns, n - j);
                    size_t offset;
                    if constexpr (kChannelLast)
                        offset =
                            tqt_get_b_packed_offset_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                                j, origin, kp, cols);
                    else
                        offset =
                            tqt_get_b_im2colpack_offset_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv(
                                j, origin, kp, cols);
                    EXPECT_EQ(offset, (j * kp + origin * cols) * sizeof(float16_t));
                    for (size_t p = 0; p < k; ++p)
                        for (size_t q = 0; q < cols; ++q)
                            bp_ref[guard + j * kp + (origin + p) * cols + q] =
                                kChannelLast ? b[(j + q) * k + p] : b[p * n + j + q];
                }
                ASSERT_EQ(std::memcmp(ap.data(), ap_ref.data(), ap.size() * sizeof(float16_t)), 0);
                ASSERT_EQ(std::memcmp(bp.data(), bp_ref.data(), bp.size() * sizeof(float16_t)), 0);
                a = ap.data() + guard + ao;
                b = bp.data() + guard + bo;
            }
            std::vector<float16_t> actual(2 * guard + m * ldd, sentinel);
            auto expected = actual;
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j)
                    expected[guard + i * ldd + j] = reference[i * n + j];
            auto run =
                kChannelLast
                    ? (kPacked ? tqt_run_conv_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv
                               : tqt_run_conv_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv)
                    : (kPacked ? tqt_run_conv_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv
                               : tqt_run_conv_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv);
            run(m, n, k, a, kPacked ? kp : k, origin, b, kPacked ? kp : (kChannelLast ? k : n),
                origin, nullptr, ldd, actual.data() + guard, ldd, bias.data(), lo, hi);
            for (size_t i = 0; i < actual.size(); ++i)
                ASSERT_EQ(static_cast<float>(actual[i]), static_cast<float>(expected[i]))
                    << "index=" << i;
        }
    }
}

TEST(Test_convcf_mx1bias_clamp_f16_f16_f16_8x1vl_f32acc_rvv_Integration, GroupsAndTails)
{
    check_f32acc_grouped_conv<false, false>();
}

TEST(Test_convcf_mx1bias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_Integration, GroupsAndPackedKOrigin)
{
    check_f32acc_grouped_conv<false, true>();
}

TEST(Test_convcl_1xnbias_clamp_f16_f16_f16t_8x1vl_f32acc_rvv_Integration, GroupsAndTails)
{
    check_f32acc_grouped_conv<true, false>();
}

TEST(Test_convcl_1xnbias_clamp_f16_f16p_f16p_8x1vl_f32acc_rvv_Integration, GroupsAndPackedKOrigin)
{
    check_f32acc_grouped_conv<true, true>();
}
