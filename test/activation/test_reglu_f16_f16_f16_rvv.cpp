//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/activation/reglu_f16_f16_f16/reglu_f16_f16_f16/tqt_reglu_f16_f16_f16_1x8vl_rvv.h"
#include "test/common/activation_test_runner.h"

using namespace torq_tile::test;

static constexpr float max_abs_error = 0.0f;
static constexpr float min_cosine = 0.999999f;

class Test_reglu_f16_f16_f16_1x8vl_rvv : public testing::TestWithParam<ActivationTestParams>
{
};

TEST_P(Test_reglu_f16_f16_f16_1x8vl_rvv, Correctness)
{
    run_gated_activation_test<float16_t>(tqt_run_reglu_f16_f16_f16_1x8vl_rvv,
                                         GatedActivationKind::Reglu, GetParam(), max_abs_error,
                                         min_cosine);
}

TEST_P(Test_reglu_f16_f16_f16_1x8vl_rvv, InPlaceAliasA)
{
    run_gated_activation_inplace_test<float16_t>(tqt_run_reglu_f16_f16_f16_1x8vl_rvv,
                                                 GatedActivationKind::Reglu, GatedAliasOperand::A,
                                                 GetParam(), max_abs_error, min_cosine);
}

TEST_P(Test_reglu_f16_f16_f16_1x8vl_rvv, InPlaceAliasB)
{
    run_gated_activation_inplace_test<float16_t>(tqt_run_reglu_f16_f16_f16_1x8vl_rvv,
                                                 GatedActivationKind::Reglu, GatedAliasOperand::B,
                                                 GetParam(), max_abs_error, min_cosine);
}

TEST(Test_reglu_f16_f16_f16_1x8vl_rvv, FusedGateUpLayout)
{
    run_gated_activation_fused_layout_test<float16_t>(tqt_run_reglu_f16_f16_f16_1x8vl_rvv, 17, 129);
    run_gated_activation_fused_layout_test<float16_t>(tqt_run_reglu_f16_f16_f16_1x8vl_rvv, 4, 1024);
    run_gated_activation_fused_layout_test<float16_t>(tqt_run_reglu_f16_f16_f16_1x8vl_rvv, 1, 1);
}

TEST(Test_reglu_f16_f16_f16_1x8vl_rvv, EdgeValues)
{
    run_gated_activation_edge_values_test<float16_t>(
        tqt_run_reglu_f16_f16_f16_1x8vl_rvv, GatedActivationKind::Reglu, max_abs_error, min_cosine);
}

TEST(Test_reglu_f16_f16_f16_1x8vl_rvv, Steps)
{
    EXPECT_EQ(tqt_get_m_step_reglu_f16_f16_f16_1x8vl_rvv(), 1u);
    EXPECT_EQ(tqt_get_n_step_reglu_f16_f16_f16_1x8vl_rvv(),
              8u * (csrr_vlenb() / sizeof(float16_t)));
}

INSTANTIATE_TEST_SUITE_P(GatedActivationTests, Test_reglu_f16_f16_f16_1x8vl_rvv,
                         gated_activation_test_values(), ActivationTestParamNameGenerator());
