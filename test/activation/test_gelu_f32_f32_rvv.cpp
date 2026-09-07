//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/activation/gelu_f32_f32/gelu_f32_f32/tqt_gelu_f32_f32_1x4vl_rvv.h"
#include "test/common/activation_test_runner.h"

using namespace torq_tile::test;

// A 200,001-point sweep over [-10, 10] measures a 9.2e-7 maximum absolute error.
static constexpr float max_abs_error = 5e-6f;
static constexpr float min_cosine = 0.999999f;

class Test_gelu_f32_f32_1x4vl_rvv : public testing::TestWithParam<ActivationTestParams>
{
};

TEST_P(Test_gelu_f32_f32_1x4vl_rvv, Correctness)
{
    run_activation_test<float>(tqt_run_gelu_f32_f32_1x4vl_rvv, ActivationKind::Gelu, GetParam(),
                               max_abs_error, min_cosine);
}

TEST_P(Test_gelu_f32_f32_1x4vl_rvv, InPlace)
{
    run_activation_inplace_test<float>(tqt_run_gelu_f32_f32_1x4vl_rvv, ActivationKind::Gelu,
                                       GetParam(), max_abs_error, min_cosine);
}

TEST(Test_gelu_f32_f32_1x4vl_rvv, EdgeValues)
{
    run_activation_edge_values_test<float>(tqt_run_gelu_f32_f32_1x4vl_rvv, ActivationKind::Gelu,
                                           max_abs_error, min_cosine);
}

TEST(Test_gelu_f32_f32_1x4vl_rvv, Steps)
{
    EXPECT_EQ(tqt_get_m_step_gelu_f32_f32_1x4vl_rvv(), 1u);
    EXPECT_EQ(tqt_get_n_step_gelu_f32_f32_1x4vl_rvv(), 4u * (csrr_vlenb() / sizeof(float)));
}

INSTANTIATE_TEST_SUITE_P(ActivationTests, Test_gelu_f32_f32_1x4vl_rvv, activation_test_values(),
                         ActivationTestParamNameGenerator());
