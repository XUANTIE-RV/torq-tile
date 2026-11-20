//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/activation/swiglu_f16_f16_f16/swiglu_f16_f16_f16/tqt_swiglu_f16_f16_f16_1x2vl_rvv.h"
#include "test/common/activation_test_runner.h"

using namespace torq_tile::test;

// Keep the established absolute-error and cosine acceptance budgets.
// Activation accuracy includes the divide,
// the optional gate/up multiply, and (for FP16) the final narrowing conversion.
static constexpr float max_abs_error = 1e-1f;
static constexpr float min_cosine = 0.9999f;

// ============================================================================
// swiglu_f16_f16_f16_1x2vl_rvv
// ============================================================================

class Test_swiglu_f16_f16_f16_1x2vl_rvv : public testing::TestWithParam<ActivationTestParams>
{
};

TEST_P(Test_swiglu_f16_f16_f16_1x2vl_rvv, Correctness)
{
    run_gated_activation_test<float16_t>(tqt_run_swiglu_f16_f16_f16_1x2vl_rvv,
                                         GatedActivationKind::Swiglu, GetParam(), max_abs_error,
                                         min_cosine);
}

TEST_P(Test_swiglu_f16_f16_f16_1x2vl_rvv, InPlaceAliasA)
{
    run_gated_activation_inplace_test<float16_t>(tqt_run_swiglu_f16_f16_f16_1x2vl_rvv,
                                                 GatedActivationKind::Swiglu, GatedAliasOperand::A,
                                                 GetParam(), max_abs_error, min_cosine);
}

TEST_P(Test_swiglu_f16_f16_f16_1x2vl_rvv, InPlaceAliasB)
{
    run_gated_activation_inplace_test<float16_t>(tqt_run_swiglu_f16_f16_f16_1x2vl_rvv,
                                                 GatedActivationKind::Swiglu, GatedAliasOperand::B,
                                                 GetParam(), max_abs_error, min_cosine);
}

TEST(Test_swiglu_f16_f16_f16_1x2vl_rvv, FusedGateUpLayout)
{
    run_gated_activation_fused_layout_test<float16_t>(tqt_run_swiglu_f16_f16_f16_1x2vl_rvv, 17,
                                                      129);
    run_gated_activation_fused_layout_test<float16_t>(tqt_run_swiglu_f16_f16_f16_1x2vl_rvv, 4,
                                                      1024);
    run_gated_activation_fused_layout_test<float16_t>(tqt_run_swiglu_f16_f16_f16_1x2vl_rvv, 1, 1);
}

TEST(Test_swiglu_f16_f16_f16_1x2vl_rvv, EdgeValues)
{
    run_gated_activation_edge_values_test<float16_t>(tqt_run_swiglu_f16_f16_f16_1x2vl_rvv,
                                                     GatedActivationKind::Swiglu, max_abs_error,
                                                     min_cosine);
}

TEST(Test_swiglu_f16_f16_f16_1x2vl_rvv, Steps)
{
    EXPECT_EQ(tqt_get_m_step_swiglu_f16_f16_f16_1x2vl_rvv(), 1u);
    EXPECT_EQ(tqt_get_n_step_swiglu_f16_f16_f16_1x2vl_rvv(), 4u * (csrr_vlenb() / sizeof(float)));
}

INSTANTIATE_TEST_SUITE_P(GatedActivationTests, Test_swiglu_f16_f16_f16_1x2vl_rvv,
                         gated_activation_test_values(), ActivationTestParamNameGenerator());

TEST(Test_swiglu_f16_f16_f16_1x2vl_rvv, SpecialValues)
{
    run_swiglu_special_values_test<float16_t>(tqt_run_swiglu_f16_f16_f16_1x2vl_rvv,
                                              tqt_get_n_step_swiglu_f16_f16_f16_1x2vl_rvv(),
                                              max_abs_error);
}
