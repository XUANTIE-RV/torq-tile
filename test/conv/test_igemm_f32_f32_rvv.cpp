//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/conv/igemm_f32_f32/igemmcf_mx1bias_clamp_f32_f32_f32/tqt_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv.h"
#include "src/conv/igemm_f32_f32/igemmcf_mx1bias_clamp_f32_f32p_f32/tqt_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv.h"
#include "src/conv/igemm_f32_f32/igemmcl_1xnbias_clamp_f32_f32_f32p/tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv.h"
#include "src/conv/igemm_f32_f32/igemmcl_1xnbias_clamp_f32_f32_f32t/tqt_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.h"
#include "test/common/conv_clamp_test_utils.h"

using namespace torq_tile::test;

// ============================================================================
// Common test cases
// ============================================================================

static auto common_igemmcf_test_values()
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
        ConvTestParams(3, 32, 1, 16, 16, 3, 3, "Large").set_pad(1, 1),
        // Grouped convolution
        ConvTestParams(16, 32, 2, 8, 8, 3, 3, "Group2").set_pad(1, 1),
        ConvTestParams(32, 64, 4, 8, 8, 3, 3, "Group4").set_pad(1, 1));
}

// ============================================================================
// Igemmcf NonPacked (mx1bias, f32_f32_f32)
// ============================================================================

static NonPackedIgemmcfFuncs<float> make_igemmcf_nonpacked_funcs()
{
    return {
        tqt_get_m_step_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv,
        tqt_get_n_step_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv,
        tqt_run_igemm_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv,
    };
}

class Test_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv, Correctness)
{
    run_nonpacked_igemmcf_test<float, float>(GetParam(), make_igemmcf_nonpacked_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv,
                         common_igemmcf_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// Igemmcf Packed (mx1bias, f32_f32p_f32)
// ============================================================================

static PackedIgemmcfFuncs<float> make_igemmcf_packed_funcs()
{
    return {
        tqt_get_m_step_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv,
        tqt_get_n_step_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv,
        tqt_get_a_packed_size_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv,
        tqt_run_a_pack_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv,
        tqt_run_igemm_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv,
        tqt_get_a_packed_offset_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv,
    };
}

class Test_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv, Correctness)
{
    run_packed_igemmcf_test<float, float>(GetParam(), make_igemmcf_packed_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv,
                         common_igemmcf_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// Common test cases for igemmcl
// ============================================================================

static auto common_igemmcl_test_values()
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
        ConvTestParams(3, 32, 1, 16, 16, 3, 3, "Large").set_pad(1, 1),
        // Grouped convolution
        ConvTestParams(16, 32, 2, 8, 8, 3, 3, "Group2").set_pad(1, 1),
        ConvTestParams(32, 64, 4, 8, 8, 3, 3, "Group4").set_pad(1, 1));
}

// ============================================================================
// Igemmcl NonPacked (1xnbias, f32_f32_f32t)
// ============================================================================

static NonPackedIgemmclFuncs<float> make_igemmcl_nonpacked_funcs()
{
    return {
        tqt_get_m_step_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_get_n_step_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_run_igemm_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
    };
}

class Test_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv, Correctness)
{
    run_nonpacked_igemmcl_test<float, float>(GetParam(), make_igemmcl_nonpacked_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_igemmcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
                         common_igemmcl_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// Igemmcl Packed (1xnbias, f32_f32_f32p)
// ============================================================================

static PackedIgemmclFuncs<float> make_igemmcl_packed_funcs()
{
    return {
        tqt_get_m_step_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_n_step_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_b_packed_size_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_run_bt_pack_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_run_igemm_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
        tqt_get_b_packed_offset_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    };
}

class Test_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv, Correctness)
{
    run_packed_igemmcl_test<float, float>(GetParam(), make_igemmcl_packed_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests, Test_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
                         common_igemmcl_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// Nonzero K-origin (split-K) coverage for the packed weight paths
//
// The parameterized suites above always pack and run at k_idx = 0. These shapes
// force a panel past the first one to use the 3vl tail split, plus a single
// partial panel below vlmax, which is where the origin-relative packed
// addressing differs from a plain per-panel offset.
// ============================================================================

static std::vector<size_t> nonzero_k_origin_channels(size_t vlmax)
{
    std::vector<size_t> channels = {4 * vlmax + 1};
    if (vlmax > 1) {
        channels.push_back(vlmax - 1);
    }
    return channels;
}

TEST(IgemmF32F32Rvv, PackedNonzeroKOrigin)
{
    const auto cl_funcs = make_igemmcl_packed_funcs();
    for (size_t oc : nonzero_k_origin_channels(cl_funcs.get_n_step() / 3)) {
        ConvTestParams tp(2, oc, 1, 5, 5, 3, 3, "OC" + std::to_string(oc));
        tp.input_shape[0] = 2;   // ID, so that KD = 2 gives two sliceable K planes
        tp.kernel_shape[0] = 2;  // KD
        run_packed_igemmcl_nonzero_k_origin_test<float, float>(tp, cl_funcs);
    }

    const auto cf_funcs = make_igemmcf_packed_funcs();
    for (size_t oc : nonzero_k_origin_channels(cf_funcs.get_m_step() / 3)) {
        ConvTestParams tp(2, oc, 1, 5, 5, 3, 3, "OC" + std::to_string(oc));
        run_packed_igemmcf_nonzero_k_origin_test<float, float>(tp, cf_funcs);
    }
}

TEST(test_igemm_f32_f32_rvv_make_igemmcf_nonpacked_funcs, ClampSpecialValues)
{
    check_igemm_clamp<float>(make_igemmcf_nonpacked_funcs());
}

TEST(test_igemm_f32_f32_rvv_make_igemmcf_packed_funcs, ClampSpecialValues)
{
    check_igemm_clamp<float>(make_igemmcf_packed_funcs());
}

TEST(test_igemm_f32_f32_rvv_make_igemmcl_nonpacked_funcs, ClampSpecialValues)
{
    check_igemm_clamp<float>(make_igemmcl_nonpacked_funcs());
}

TEST(test_igemm_f32_f32_rvv_make_igemmcl_packed_funcs, ClampSpecialValues)
{
    check_igemm_clamp<float>(make_igemmcl_packed_funcs());
}
