//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

// Generic kernels (4)
#include "src/dwconv/dwconv_f16_f16/dwconvcf_mx1bias_clamp_f16_f16_f16/tqt_dwconvcf_mx1bias_clamp_f16_f16_f16_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcf_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl_1xnbias_clamp_f16_f16_f16t/tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv.h"
// Specialized kernels (6)
#include "src/dwconv/dwconv_f16_f16/dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.h"
#include "test/common/dwconv_test_runner.h"

using namespace torq_tile::test;

// ============================================================================
// Test param sets
//
// dwconv constraint: groups == IC. We use 4 categories of test params:
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
        ConvTestParams(8, 16, 8, 16, 16, 3, 3, "Spec3x3s1DM2").set_pad(1, 1));
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
// Kernel #1: dwconvcf_mx1bias_clamp_f16_f16_f16_1x1vl_rvv (generic non-packed)
// ============================================================================

static NonPackedDwconvFuncs<float16_t> make_funcs_dwconvcf_generic()
{
    return {
        tqt_get_output_size_dwconvcf_mx1bias_clamp_f16_f16_f16_1x1vl_rvv,
        tqt_run_dwconv_dwconvcf_mx1bias_clamp_f16_f16_f16_1x1vl_rvv,
    };
}

class Test_dwconvcf_generic_f16 : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_dwconvcf_generic_f16, BasicDM1)
{
    run_nonpacked_dwconv_test<float16_t>(make_funcs_dwconvcf_generic(), GetParam(),
                                         ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(DwconvBasic, Test_dwconvcf_generic_f16, generic_basic_params(),
                         ConvTestParamNameGenerator());
INSTANTIATE_TEST_SUITE_P(DwconvDM, Test_dwconvcf_generic_f16, generic_dm_params(),
                         ConvTestParamNameGenerator());

// ============================================================================
// Kernel #2: dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv (generic non-packed)
// ============================================================================

static NonPackedDwconvFuncs<float16_t> make_funcs_dwconvcl_generic()
{
    return {
        tqt_get_output_size_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv,
        tqt_run_dwconv_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv,
    };
}

class Test_dwconvcl_generic_f16 : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_dwconvcl_generic_f16, BasicDM1)
{
    run_nonpacked_dwconv_test<float16_t>(make_funcs_dwconvcl_generic(), GetParam(),
                                         ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(DwconvBasic, Test_dwconvcl_generic_f16, generic_basic_params(),
                         ConvTestParamNameGenerator());
INSTANTIATE_TEST_SUITE_P(DwconvDM, Test_dwconvcl_generic_f16, generic_dm_params(),
                         ConvTestParamNameGenerator());

// ============================================================================
// Kernel #3: dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv (generic packed)
// ============================================================================

static GenericPackedDwconvFuncs<float16_t> make_funcs_dwconvcf_generic_packed()
{
    return {
        tqt_get_output_size_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_pack_input_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_dwconv_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
    };
}

class Test_dwconvcf_generic_packed_f16 : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_dwconvcf_generic_packed_f16, BasicDM1)
{
    run_generic_packed_dwconv_test<float16_t>(make_funcs_dwconvcf_generic_packed(), GetParam(),
                                              ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(DwconvBasic, Test_dwconvcf_generic_packed_f16, generic_basic_params(),
                         ConvTestParamNameGenerator());
INSTANTIATE_TEST_SUITE_P(DwconvDM, Test_dwconvcf_generic_packed_f16, generic_dm_params(),
                         ConvTestParamNameGenerator());

// ============================================================================
// Kernel #4: dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv (generic packed)
// ============================================================================

static GenericPackedDwconvFuncs<float16_t> make_funcs_dwconvcl_generic_packed()
{
    return {
        tqt_get_output_size_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_pack_input_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_dwconv_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
    };
}

class Test_dwconvcl_generic_packed_f16 : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_dwconvcl_generic_packed_f16, BasicDM1)
{
    run_generic_packed_dwconv_test<float16_t>(make_funcs_dwconvcl_generic_packed(), GetParam(),
                                              ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(DwconvBasic, Test_dwconvcl_generic_packed_f16, generic_basic_params(),
                         ConvTestParamNameGenerator());
INSTANTIATE_TEST_SUITE_P(DwconvDM, Test_dwconvcl_generic_packed_f16, generic_dm_params(),
                         ConvTestParamNameGenerator());

// ============================================================================
// Specialized kernels: macro-generated test fixtures
//
// Each spec kernel has identical scaffolding modulo function names, layout, and
// param set. We write per-kernel boilerplate explicitly for clarity.
// ============================================================================

// --- Kernel #5: dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv ---
static SpecPackedDwconvFuncs<float16_t> make_funcs_dwconvcf3x3s1()
{
    return {
        tqt_get_output_size_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
    };
}
class Test_dwconvcf3x3s1 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcf3x3s1, Correctness)
{
    run_spec_packed_dwconv_test<float16_t>(make_funcs_dwconvcf3x3s1(), GetParam(),
                                           ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcf3x3s1, spec_3x3s1_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #6: dwconvcf3x3s2 ---
static SpecPackedDwconvFuncs<float16_t> make_funcs_dwconvcf3x3s2()
{
    return {
        tqt_get_output_size_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
    };
}
class Test_dwconvcf3x3s2 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcf3x3s2, Correctness)
{
    run_spec_packed_dwconv_test<float16_t>(make_funcs_dwconvcf3x3s2(), GetParam(),
                                           ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcf3x3s2, spec_3x3s2_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #7: dwconvcf5x5s1 ---
static SpecPackedDwconvFuncs<float16_t> make_funcs_dwconvcf5x5s1()
{
    return {
        tqt_get_output_size_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
    };
}
class Test_dwconvcf5x5s1 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcf5x5s1, Correctness)
{
    run_spec_packed_dwconv_test<float16_t>(make_funcs_dwconvcf5x5s1(), GetParam(),
                                           ConvLayout::kChannelFirst);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcf5x5s1, spec_5x5s1_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #8: dwconvcl3x3s1 ---
static SpecPackedDwconvFuncs<float16_t> make_funcs_dwconvcl3x3s1()
{
    return {
        tqt_get_output_size_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
    };
}
class Test_dwconvcl3x3s1 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcl3x3s1, Correctness)
{
    run_spec_packed_dwconv_test<float16_t>(make_funcs_dwconvcl3x3s1(), GetParam(),
                                           ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcl3x3s1, spec_3x3s1_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #9: dwconvcl3x3s2 ---
static SpecPackedDwconvFuncs<float16_t> make_funcs_dwconvcl3x3s2()
{
    return {
        tqt_get_output_size_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
    };
}
class Test_dwconvcl3x3s2 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcl3x3s2, Correctness)
{
    run_spec_packed_dwconv_test<float16_t>(make_funcs_dwconvcl3x3s2(), GetParam(),
                                           ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcl3x3s2, spec_3x3s2_params(),
                         ConvTestParamNameGenerator());

// --- Kernel #10: dwconvcl5x5s1 ---
static SpecPackedDwconvFuncs<float16_t> make_funcs_dwconvcl5x5s1()
{
    return {
        tqt_get_output_size_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_input_size_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
    };
}
class Test_dwconvcl5x5s1 : public testing::TestWithParam<ConvTestParams>
{
};
TEST_P(Test_dwconvcl5x5s1, Correctness)
{
    run_spec_packed_dwconv_test<float16_t>(make_funcs_dwconvcl5x5s1(), GetParam(),
                                           ConvLayout::kChannelLast);
}
INSTANTIATE_TEST_SUITE_P(Dwconv, Test_dwconvcl5x5s1, spec_5x5s1_params(),
                         ConvTestParamNameGenerator());
