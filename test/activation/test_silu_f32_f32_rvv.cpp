//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/activation/silu_f32_f32/silu_f32_f32/tqt_silu_f32_f32_1x4vl_rvv.h"
#include "test/common/activation_test_runner.h"

using namespace torq_tile::test;

// Keep the established absolute-error and cosine acceptance budgets.
// Activation accuracy includes the divide,
// the optional gate/up multiply, and (for FP16) the final narrowing conversion.
static constexpr float max_abs_error = 5e-5f;
static constexpr float min_cosine = 0.999999f;

// ============================================================================
// silu_f32_f32_1x4vl_rvv
// ============================================================================

class Test_silu_f32_f32_1x4vl_rvv : public testing::TestWithParam<ActivationTestParams>
{
};

TEST_P(Test_silu_f32_f32_1x4vl_rvv, Correctness)
{
    run_activation_test<float>(tqt_run_silu_f32_f32_1x4vl_rvv, ActivationKind::Silu, GetParam(),
                               max_abs_error, min_cosine);
}

TEST_P(Test_silu_f32_f32_1x4vl_rvv, InPlace)
{
    run_activation_inplace_test<float>(tqt_run_silu_f32_f32_1x4vl_rvv, ActivationKind::Silu,
                                       GetParam(), max_abs_error, min_cosine);
}

TEST(Test_silu_f32_f32_1x4vl_rvv, Steps)
{
    EXPECT_EQ(tqt_get_m_step_silu_f32_f32_1x4vl_rvv(), 1u);
    EXPECT_EQ(tqt_get_n_step_silu_f32_f32_1x4vl_rvv(), 4u * (csrr_vlenb() / sizeof(float)));
}

INSTANTIATE_TEST_SUITE_P(ActivationTests, Test_silu_f32_f32_1x4vl_rvv, activation_test_values(),
                         ActivationTestParamNameGenerator());

TEST(Test_silu_f32_f32_1x4vl_rvv, SpecialValues)
{
    run_silu_special_values_test<float>(tqt_run_silu_f32_f32_1x4vl_rvv,
                                        tqt_get_n_step_silu_f32_f32_1x4vl_rvv(), max_abs_error);
}
