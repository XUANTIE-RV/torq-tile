//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/conv/conv_f32_f32/convcf_mx1bias_clamp_f32_f32_f32/tqt_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv.h"
#include "src/conv/igemm_f32_f32/igemmcf_mx1bias_clamp_f32_f32_f32/tqt_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv.h"

// Generic kernels (4)
#include "src/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32_f32_f32/tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl_1xnbias_clamp_f32_f32_f32t/tqt_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv.h"
// Specialized kernels (6)
#include "src/dwconv/dwconv_f32_f32/dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p/tqt_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.h"
#include "test/common/conv_clamp_test_utils.h"
#include "test/common/dwconv_test_runner.h"

using namespace torq_tile::test;

// ============================================================================
// Test param sets
//
// dwconv constraint: groups == IC. We use 5 categories of test params:
//   - generic_basic : dm=1 + various spatial / KxK / stride / dilation / pad / 1D / 3D
//   - generic_dm    : dm > 1 cases (dm=2, dm=3) and tail_vl (IC not multiple of vlmax)
//   - spec_3x3s1    : KH=KW=3, stride=1, dilation=1, 2D only
//   - spec_3x3s2    : KH=KW=3, stride=2, dilation=1, 2D only
//   - spec_5x5s1    : KH=KW=5, stride=1, dilation=1, 2D only
// ============================================================================

static auto generic_basic_params()
{
    return testing::Values(
        // Basic 3x3 dwconv, dm=1
        ConvTestParams(8, 8, /*g=*/8, 16, 16, 3, 3, "Basic"),
        ConvTestParams(8, 8, 8, 16, 16, 3, 3, "NoBias").set_bias(false),
        ConvTestParams(8, 8, 8, 16, 16, 3, 3, "Pad1").set_pad(1, 1),
        ConvTestParams(8, 8, 8, 16, 16, 3, 3, "Stride2").set_stride(2, 2),
        ConvTestParams(8, 8, 8, 16, 16, 3, 3, "Clamp").set_clamp(0.0f, 1.0f),
        ConvTestParams(8, 8, 8, 16, 16, 5, 5, "5x5"), ConvTestParams(8, 8, 8, 16, 16, 1, 1, "1x1"),
        ConvTestParams(8, 8, 8, 16, 16, 3, 3, "Dilation2").set_dilation(2, 2).set_pad(2, 2),
        // Tail vl (IC=15, vlmax=4 -> tail_vl=3 if VLEN=128)
        ConvTestParams(15, 15, 15, 8, 8, 3, 3, "Tail"));
}

static auto generic_dm_params()
{
    return testing::Values(
        // dm=2: OC = 2*IC
        ConvTestParams(8, 16, 8, 8, 8, 3, 3, "DM2"),
        ConvTestParams(15, 30, 15, 8, 8, 3, 3, "DM2Tail"),
        ConvTestParams(8, 16, 8, 8, 8, 3, 3, "DM2Pad").set_pad(1, 1),
        // dm=3
        ConvTestParams(8, 24, 8, 8, 8, 3, 3, "DM3"));
}

static auto spec_3x3s1_params()
{
    return testing::Values(
        ConvTestParams(8, 8, 8, 16, 16, 3, 3, "Spec3x3s1Basic").set_pad(1, 1),
        ConvTestParams(8, 8, 8, 32, 32, 3, 3, "Spec3x3s1Pad1").set_pad(1, 1),
        ConvTestParams(8, 8, 8, 16, 16, 3, 3, "Spec3x3s1NoPad"),
        ConvTestParams(8, 8, 8, 16, 16, 3, 3, "Spec3x3s1NoBias").set_bias(false).set_pad(1, 1),
        ConvTestParams(15, 15, 15, 16, 16, 3, 3, "Spec3x3s1Tail").set_pad(1, 1),
        ConvTestParams(8, 16, 8, 16, 16, 3, 3, "Spec3x3s1DM2").set_pad(1, 1),
        ConvTestParams(8, 8, 8, 4, 1, 3, 3, "Spec3x3s1OW1").set_pad(1, 1),
        ConvTestParams(8, 8, 8, 5, 2, 3, 3, "Spec3x3s1OW2").set_pad(1, 1));
}

static auto spec_3x3s2_params()
{
    return testing::Values(
        ConvTestParams(8, 8, 8, 16, 16, 3, 3, "Spec3x3s2").set_stride(2, 2).set_pad(1, 1),
        ConvTestParams(8, 8, 8, 32, 32, 3, 3, "Spec3x3s2Big").set_stride(2, 2).set_pad(1, 1),
        ConvTestParams(15, 15, 15, 16, 16, 3, 3, "Spec3x3s2Tail").set_stride(2, 2).set_pad(1, 1));
}

static auto spec_5x5s1_params()
{
    return testing::Values(ConvTestParams(8, 8, 8, 16, 16, 5, 5, "Spec5x5s1").set_pad(2, 2),
                           ConvTestParams(8, 8, 8, 32, 32, 5, 5, "Spec5x5s1Big").set_pad(2, 2),
                           ConvTestParams(15, 15, 15, 16, 16, 5, 5, "Spec5x5s1Tail").set_pad(2, 2));
}

// ============================================================================
// Kernel #1: dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv (generic non-packed)
// ============================================================================

static NonPackedDwconvFuncs<float> make_funcs_dwconvcf_generic()
{
    return {
        tqt_get_output_size_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv,
        tqt_run_dwconv_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv,
    };
}

class Test_dwconvcf_generic_f32 : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_dwconvcf_generic_f32, BasicDM1)
{
    run_nonpacked_dwconv_test<float>(make_funcs_dwconvcf_generic(), GetParam(),
                                     ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(DwconvBasic, Test_dwconvcf_generic_f32, generic_basic_params(),
                         ConvTestParamNameGenerator());
INSTANTIATE_TEST_SUITE_P(DwconvDM, Test_dwconvcf_generic_f32, generic_dm_params(),
                         ConvTestParamNameGenerator());

// ============================================================================
// Kernel #2: dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv (generic non-packed)
// ============================================================================

static NonPackedDwconvFuncs<float> make_funcs_dwconvcl_generic()
{
    return {
        tqt_get_output_size_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv,
        tqt_run_dwconv_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv,
    };
}

class Test_dwconvcl_generic_f32 : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_dwconvcl_generic_f32, BasicDM1)
{
    run_nonpacked_dwconv_test<float>(make_funcs_dwconvcl_generic(), GetParam(),
                                     ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(DwconvBasic, Test_dwconvcl_generic_f32, generic_basic_params(),
                         ConvTestParamNameGenerator());
INSTANTIATE_TEST_SUITE_P(DwconvDM, Test_dwconvcl_generic_f32, generic_dm_params(),
                         ConvTestParamNameGenerator());

// ============================================================================
// Kernel #3: dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv (generic packed)
// ============================================================================

static GenericPackedDwconvFuncs<float> make_funcs_dwconvcf_generic_packed()
{
    return {
        tqt_get_output_size_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_pack_input_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_dwconv_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
    };
}

class Test_dwconvcf_generic_packed_f32 : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_dwconvcf_generic_packed_f32, BasicDM1)
{
    run_generic_packed_dwconv_test<float>(make_funcs_dwconvcf_generic_packed(), GetParam(),
                                          ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(DwconvBasic, Test_dwconvcf_generic_packed_f32, generic_basic_params(),
                         ConvTestParamNameGenerator());
INSTANTIATE_TEST_SUITE_P(DwconvDM, Test_dwconvcf_generic_packed_f32, generic_dm_params(),
                         ConvTestParamNameGenerator());

// ============================================================================
// Kernel #4: dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv (generic packed)
// ============================================================================

static GenericPackedDwconvFuncs<float> make_funcs_dwconvcl_generic_packed()
{
    return {
        tqt_get_output_size_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_pack_input_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_dwconv_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
    };
}

class Test_dwconvcl_generic_packed_f32 : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_dwconvcl_generic_packed_f32, BasicDM1)
{
    run_generic_packed_dwconv_test<float>(make_funcs_dwconvcl_generic_packed(), GetParam(),
                                          ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(DwconvBasic, Test_dwconvcl_generic_packed_f32, generic_basic_params(),
                         ConvTestParamNameGenerator());
INSTANTIATE_TEST_SUITE_P(DwconvDM, Test_dwconvcl_generic_packed_f32, generic_dm_params(),
                         ConvTestParamNameGenerator());

// ============================================================================
// Specialized kernels: macro-generated test fixtures
//
// Each spec kernel has identical scaffolding modulo function names, layout, and
// param set. We write per-kernel boilerplate explicitly for clarity.
// ============================================================================

// --- Kernel #5: dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv ---
static SpecPackedDwconvFuncs<float> make_funcs_dwconvcf3x3s1()
{
    return {
        tqt_get_output_size_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
    };
}
class Test_dwconvcf3x3s1 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcf3x3s1, Correctness)
{
    run_spec_packed_dwconv_test<float>(make_funcs_dwconvcf3x3s1(), GetParam(),
                                       ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcf3x3s1, spec_3x3s1_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #6: dwconvcf3x3s2 ---
static SpecPackedDwconvFuncs<float> make_funcs_dwconvcf3x3s2()
{
    return {
        tqt_get_output_size_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
    };
}
class Test_dwconvcf3x3s2 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcf3x3s2, Correctness)
{
    run_spec_packed_dwconv_test<float>(make_funcs_dwconvcf3x3s2(), GetParam(),
                                       ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcf3x3s2, spec_3x3s2_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #7: dwconvcf5x5s1 ---
static SpecPackedDwconvFuncs<float> make_funcs_dwconvcf5x5s1()
{
    return {
        tqt_get_output_size_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
    };
}
class Test_dwconvcf5x5s1 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcf5x5s1, Correctness)
{
    run_spec_packed_dwconv_test<float>(make_funcs_dwconvcf5x5s1(), GetParam(),
                                       ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcf5x5s1, spec_5x5s1_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #8: dwconvcl3x3s1 ---
static SpecPackedDwconvFuncs<float> make_funcs_dwconvcl3x3s1()
{
    return {
        tqt_get_output_size_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
    };
}
class Test_dwconvcl3x3s1 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcl3x3s1, Correctness)
{
    run_spec_packed_dwconv_test<float>(make_funcs_dwconvcl3x3s1(), GetParam(),
                                       ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcl3x3s1, spec_3x3s1_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #9: dwconvcl3x3s2 ---
static SpecPackedDwconvFuncs<float> make_funcs_dwconvcl3x3s2()
{
    return {
        tqt_get_output_size_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
    };
}
class Test_dwconvcl3x3s2 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcl3x3s2, Correctness)
{
    run_spec_packed_dwconv_test<float>(make_funcs_dwconvcl3x3s2(), GetParam(),
                                       ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcl3x3s2, spec_3x3s2_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #10: dwconvcl5x5s1 ---
static SpecPackedDwconvFuncs<float> make_funcs_dwconvcl5x5s1()
{
    return {
        tqt_get_output_size_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
    };
}
class Test_dwconvcl5x5s1 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcl5x5s1, Correctness)
{
    run_spec_packed_dwconv_test<float>(make_funcs_dwconvcl5x5s1(), GetParam(),
                                       ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcl5x5s1, spec_5x5s1_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #11: dwconvcl3x3s1 weight-packed (input/output raw NHWC) ---
static SpecWeightPackedDwconvFuncs<float> make_funcs_dwconvcl3x3s1_f32_f32_f32p()
{
    return {
        tqt_get_input_size_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv,
        tqt_get_output_size_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl3x3s1_1xnbias_clamp_f32_f32_f32p_4x1x1vl_rvv,
    };
}
class Test_dwconvcl3x3s1_f32_f32_f32p : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcl3x3s1_f32_f32_f32p, Correctness)
{
    run_spec_weight_packed_dwconv_test<float>(make_funcs_dwconvcl3x3s1_f32_f32_f32p(), GetParam(),
                                              ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcl3x3s1_f32_f32_f32p, spec_3x3s1_params(),
                         ConvTestParamNameGenerator());

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf_generic, ClampSpecialValues)
{
    const size_t vl = __riscv_vsetvlmax_e32m1();
    RecordProperty("vlmax", static_cast<int>(vl));
    // Raw CF vectorizes output width, so exercise its runtime vector boundaries.
    check_dwconv_clamp<float>(
        make_funcs_dwconvcf_generic(), ConvLayout::kChannelFirst, 3, 3, 1, vl,
        std::array<size_t, 5>{1, std::max(size_t(1), vl - 1), vl, vl + 1, 2 * vl + 1});
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf_generic, MultiplierOrdering)
{
    const size_t vl = __riscv_vsetvlmax_e32m1();
    RecordProperty("vlmax", static_cast<int>(vl));
    // Raw CF vectorizes output width, so exercise its runtime vector boundaries.
    check_dwconv_multiplier<float>(
        make_funcs_dwconvcf_generic(), ConvLayout::kChannelFirst, 3, 3, 1, vl,
        std::array<size_t, 5>{1, std::max(size_t(1), vl - 1), vl, vl + 1, 2 * vl + 1});
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl_generic, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcl_generic(), ConvLayout::kChannelLast, 3, 3, 1,
                              __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl_generic, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcl_generic(), ConvLayout::kChannelLast, 3, 3, 1,
                                   __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf_generic_packed, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcf_generic_packed(), ConvLayout::kChannelFirst, 3, 3,
                              1, __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf_generic_packed, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcf_generic_packed(), ConvLayout::kChannelFirst,
                                   3, 3, 1, __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl_generic_packed, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcl_generic_packed(), ConvLayout::kChannelLast, 3, 3,
                              1, __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl_generic_packed, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcl_generic_packed(), ConvLayout::kChannelLast,
                                   3, 3, 1, __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf3x3s1, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcf3x3s1(), ConvLayout::kChannelFirst, 3, 3, 1,
                              __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf3x3s1, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcf3x3s1(), ConvLayout::kChannelFirst, 3, 3, 1,
                                   __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf3x3s2, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcf3x3s2(), ConvLayout::kChannelFirst, 3, 3, 2,
                              __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf3x3s2, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcf3x3s2(), ConvLayout::kChannelFirst, 3, 3, 2,
                                   __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf5x5s1, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcf5x5s1(), ConvLayout::kChannelFirst, 5, 5, 1,
                              __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcf5x5s1, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcf5x5s1(), ConvLayout::kChannelFirst, 5, 5, 1,
                                   __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl3x3s1, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcl3x3s1(), ConvLayout::kChannelLast, 3, 3, 1,
                              __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl3x3s1, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcl3x3s1(), ConvLayout::kChannelLast, 3, 3, 1,
                                   __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl3x3s2, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcl3x3s2(), ConvLayout::kChannelLast, 3, 3, 2,
                              __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl3x3s2, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcl3x3s2(), ConvLayout::kChannelLast, 3, 3, 2,
                                   __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl5x5s1, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcl5x5s1(), ConvLayout::kChannelLast, 5, 5, 1,
                              __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl5x5s1, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcl5x5s1(), ConvLayout::kChannelLast, 5, 5, 1,
                                   __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl3x3s1_f32_f32_f32p, ClampSpecialValues)
{
    check_dwconv_clamp<float>(make_funcs_dwconvcl3x3s1_f32_f32_f32p(), ConvLayout::kChannelLast, 3,
                              3, 1, __riscv_vsetvlmax_e32m1(), 4);
}

TEST(test_dwconv_f32_f32_rvv_make_funcs_dwconvcl3x3s1_f32_f32_f32p, MultiplierOrdering)
{
    check_dwconv_multiplier<float>(make_funcs_dwconvcl3x3s1_f32_f32_f32p(),
                                   ConvLayout::kChannelLast, 3, 3, 1, __riscv_vsetvlmax_e32m1(), 4);
}

// Cross-check every group against independent im2col and implicit-GEMM paths.
TEST(Test_dwconv_multiplier, GroupedConvAndIgemm)
{
    for (size_t channels : {size_t(1), size_t(3), size_t(9)}) {
        for (size_t multiplier : {size_t(1), size_t(2), size_t(3)}) {
            auto tp = ConvTestParams(channels, channels * multiplier, channels, 5, 7, 3, 3);
            tp.set_pad(1, 1);
            const auto p = tp.to_conv_params();
            const size_t spatial = p.output_shape[1] * p.output_shape[2], taps = 9;
            std::vector<float> input(channels * 5 * 7), weight(p.output_channel * taps);
            std::vector<float> bias(p.output_channel), dw(p.output_channel * spatial);
            std::vector<float> conv(dw.size()), igemm(dw.size()), im2col(taps * spatial);
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = float(int(i % 11) - 5);
            for (size_t i = 0; i < weight.size(); ++i)
                weight[i] = float(int(i % 7) - 3);
            for (size_t i = 0; i < bias.size(); ++i)
                bias[i] = float(int(i % 5) - 2);
            for (bool has_bias : {false, true}) {
                for (bool clamp : {false, true}) {
                    SCOPED_TRACE(testing::Message() << "IC=" << channels << " M=" << multiplier
                                                    << " bias=" << has_bias << " clamp=" << clamp);
                    const float lo = clamp ? -3 : -INFINITY, hi = clamp ? 13 : INFINITY;
                    make_funcs_dwconvcf_generic().run_dwconv(input.data(), weight.data(),
                                                             has_bias ? bias.data() : nullptr,
                                                             dw.data(), &p, lo, hi);
                    for (size_t group = 0; group < channels; ++group) {
                        const size_t oc = group * multiplier;
                        const float *w = weight.data() + oc * taps;
                        const float *b = has_bias ? bias.data() + oc : nullptr;
                        tqt_run_im2col_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv(
                            group, input.data(), im2col.data(), &p);
                        tqt_run_conv_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv(
                            multiplier, spatial, taps, w, taps, 0, im2col.data(), spatial, 0,
                            nullptr, spatial, conv.data() + oc * spatial, spatial, b, lo, hi);
                        tqt_run_igemm_igemmcf_mx1bias_clamp_f32_f32_f32_3vlx8_rvv(
                            multiplier, spatial, taps, input.data(), group, 0, 0, w, taps, 0,
                            nullptr, spatial, igemm.data() + oc * spatial, spatial, b, lo, hi, &p);
                    }
                    ASSERT_EQ(dw, conv);
                    ASSERT_EQ(dw, igemm);
                }
            }
        }
    }
}
