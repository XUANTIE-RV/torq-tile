//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/activation/swiglu_f32_f32_f32/swiglu_f32_f32_f32/tqt_swiglu_f32_f32_f32_1x4vl_rvv.h"
#include "test/common/activation_test_runner.h"

using namespace torq_tile::test;

// The SiLU term comes from the cephes polynomial exp in src/common/tqt_math_rvv.h
// followed by a correctly-rounded divide, and is then multiplied by the linear
// operand, so the absolute error of the result is the SiLU error scaled by |b|.
// The SiLU error over the [-10, 10] range used by these tests is at most 1.1e-6,
// which the |b| <= 10 factor lifts to ~1.1e-5; the rounding of the product adds up
// to |c| * 2^-24 = 6e-6. Over this shape set the measured maximum is 1.5e-5.
//
// Preserve the established acceptance budget rather than tightening it to
// this range-specific measurement.
static constexpr float max_abs_error = 5e-4f;
static constexpr float min_cosine = 0.999999f;

// ============================================================================
// swiglu_f32_f32_f32_1x4vl_rvv
// ============================================================================

class Test_swiglu_f32_f32_f32_1x4vl_rvv : public testing::TestWithParam<ActivationTestParams>
{
};

TEST_P(Test_swiglu_f32_f32_f32_1x4vl_rvv, Correctness)
{
    run_gated_activation_test<float>(tqt_run_swiglu_f32_f32_f32_1x4vl_rvv,
                                     GatedActivationKind::Swiglu, GetParam(), max_abs_error,
                                     min_cosine);
}

TEST_P(Test_swiglu_f32_f32_f32_1x4vl_rvv, InPlaceAliasA)
{
    run_gated_activation_inplace_test<float>(tqt_run_swiglu_f32_f32_f32_1x4vl_rvv,
                                             GatedActivationKind::Swiglu, GatedAliasOperand::A,
                                             GetParam(), max_abs_error, min_cosine);
}

TEST_P(Test_swiglu_f32_f32_f32_1x4vl_rvv, InPlaceAliasB)
{
    run_gated_activation_inplace_test<float>(tqt_run_swiglu_f32_f32_f32_1x4vl_rvv,
                                             GatedActivationKind::Swiglu, GatedAliasOperand::B,
                                             GetParam(), max_abs_error, min_cosine);
}

// The fused [M, 2N] gate-and-up layout is the reason this kernel takes two
// independent leading dimensions instead of one buffer, so it gets its own check:
// addressing the two halves through strides must match two separate buffers
// exactly.
TEST(Test_swiglu_f32_f32_f32_1x4vl_rvv, FusedGateUpLayout)
{
    run_gated_activation_fused_layout_test<float>(tqt_run_swiglu_f32_f32_f32_1x4vl_rvv, 17, 129);
    run_gated_activation_fused_layout_test<float>(tqt_run_swiglu_f32_f32_f32_1x4vl_rvv, 4, 1024);
    run_gated_activation_fused_layout_test<float>(tqt_run_swiglu_f32_f32_f32_1x4vl_rvv, 1, 1);
}

TEST(Test_swiglu_f32_f32_f32_1x4vl_rvv, EdgeValues)
{
    run_gated_activation_edge_values_test<float>(tqt_run_swiglu_f32_f32_f32_1x4vl_rvv,
                                                 GatedActivationKind::Swiglu, max_abs_error,
                                                 min_cosine);
}

TEST(Test_swiglu_f32_f32_f32_1x4vl_rvv, Steps)
{
    EXPECT_EQ(tqt_get_m_step_swiglu_f32_f32_f32_1x4vl_rvv(), 1u);
    EXPECT_EQ(tqt_get_n_step_swiglu_f32_f32_f32_1x4vl_rvv(), 4u * (csrr_vlenb() / sizeof(float)));
}

INSTANTIATE_TEST_SUITE_P(GatedActivationTests, Test_swiglu_f32_f32_f32_1x4vl_rvv,
                         gated_activation_test_values(), ActivationTestParamNameGenerator());
