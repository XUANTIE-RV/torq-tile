//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/activation/geglu_f32_f32_f32/geglu_f32_f32_f32/tqt_geglu_f32_f32_f32_1x4vl_rvv.h"
#include "test/common/activation_test_runner.h"

using namespace torq_tile::test;

// Sweeping 200,001 gate values over [-10, 10] and integer b in [-10, 10]
// measures a 1.04e-5 maximum absolute error.
static constexpr float max_abs_error = 5e-5f;
static constexpr float min_cosine = 0.999999f;

class Test_geglu_f32_f32_f32_1x4vl_rvv : public testing::TestWithParam<ActivationTestParams>
{
};

TEST_P(Test_geglu_f32_f32_f32_1x4vl_rvv, Correctness)
{
    run_gated_activation_test<float>(tqt_run_geglu_f32_f32_f32_1x4vl_rvv,
                                     GatedActivationKind::Geglu, GetParam(), max_abs_error,
                                     min_cosine);
}

TEST_P(Test_geglu_f32_f32_f32_1x4vl_rvv, InPlaceAliasA)
{
    run_gated_activation_inplace_test<float>(tqt_run_geglu_f32_f32_f32_1x4vl_rvv,
                                             GatedActivationKind::Geglu, GatedAliasOperand::A,
                                             GetParam(), max_abs_error, min_cosine);
}

TEST_P(Test_geglu_f32_f32_f32_1x4vl_rvv, InPlaceAliasB)
{
    run_gated_activation_inplace_test<float>(tqt_run_geglu_f32_f32_f32_1x4vl_rvv,
                                             GatedActivationKind::Geglu, GatedAliasOperand::B,
                                             GetParam(), max_abs_error, min_cosine);
}

TEST(Test_geglu_f32_f32_f32_1x4vl_rvv, FusedGateUpLayout)
{
    run_gated_activation_fused_layout_test<float>(tqt_run_geglu_f32_f32_f32_1x4vl_rvv, 17, 129);
    run_gated_activation_fused_layout_test<float>(tqt_run_geglu_f32_f32_f32_1x4vl_rvv, 4, 1024);
    run_gated_activation_fused_layout_test<float>(tqt_run_geglu_f32_f32_f32_1x4vl_rvv, 1, 1);
}

TEST(Test_geglu_f32_f32_f32_1x4vl_rvv, EdgeValues)
{
    run_gated_activation_edge_values_test<float>(
        tqt_run_geglu_f32_f32_f32_1x4vl_rvv, GatedActivationKind::Geglu, max_abs_error, min_cosine);
}

TEST(Test_geglu_f32_f32_f32_1x4vl_rvv, Steps)
{
    EXPECT_EQ(tqt_get_m_step_geglu_f32_f32_f32_1x4vl_rvv(), 1u);
    EXPECT_EQ(tqt_get_n_step_geglu_f32_f32_f32_1x4vl_rvv(), 4u * (csrr_vlenb() / sizeof(float)));
}

INSTANTIATE_TEST_SUITE_P(GatedActivationTests, Test_geglu_f32_f32_f32_1x4vl_rvv,
                         gated_activation_test_values(), ActivationTestParamNameGenerator());
