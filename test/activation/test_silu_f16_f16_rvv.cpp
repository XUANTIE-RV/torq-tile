//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/activation/silu_f16_f16/silu_f16_f16/tqt_silu_f16_f16_1x2vl_rvv.h"
#include "test/common/activation_test_runner.h"

using namespace torq_tile::test;

// Keep the established absolute-error and cosine acceptance budgets.
// Activation accuracy includes the divide,
// the optional gate/up multiply, and (for FP16) the final narrowing conversion.
static constexpr float max_abs_error = 1e-2f;
static constexpr float min_cosine = 0.9999f;

// ============================================================================
// silu_f16_f16_1x2vl_rvv
// ============================================================================

class Test_silu_f16_f16_1x2vl_rvv : public testing::TestWithParam<ActivationTestParams>
{
};

TEST_P(Test_silu_f16_f16_1x2vl_rvv, Correctness)
{
    run_activation_test<float16_t>(tqt_run_silu_f16_f16_1x2vl_rvv, ActivationKind::Silu, GetParam(),
                                   max_abs_error, min_cosine);
}

TEST_P(Test_silu_f16_f16_1x2vl_rvv, InPlace)
{
    run_activation_inplace_test<float16_t>(tqt_run_silu_f16_f16_1x2vl_rvv, ActivationKind::Silu,
                                           GetParam(), max_abs_error, min_cosine);
}

TEST(Test_silu_f16_f16_1x2vl_rvv, Steps)
{
    EXPECT_EQ(tqt_get_m_step_silu_f16_f16_1x2vl_rvv(), 1u);
    EXPECT_EQ(tqt_get_n_step_silu_f16_f16_1x2vl_rvv(), 4u * (csrr_vlenb() / sizeof(float)));
}

INSTANTIATE_TEST_SUITE_P(ActivationTests, Test_silu_f16_f16_1x2vl_rvv, activation_test_values(),
                         ActivationTestParamNameGenerator());

TEST(Test_silu_f16_f16_1x2vl_rvv, SpecialValues)
{
    run_silu_special_values_test<float16_t>(tqt_run_silu_f16_f16_1x2vl_rvv,
                                            tqt_get_n_step_silu_f16_f16_1x2vl_rvv(), max_abs_error);
}

TEST(Test_silu_f16_f16_1x2vl_rvv, AllFp16Inputs)
{
    run_silu_all_fp16_values_test<float16_t>(tqt_run_silu_f16_f16_1x2vl_rvv, max_abs_error);
}
