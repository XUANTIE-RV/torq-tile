//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/activation/relu_f32_f32/relu_f32_f32/tqt_relu_f32_f32_1x8vl_rvv.h"
#include "test/common/activation_test_runner.h"

using namespace torq_tile::test;

// ReLU is a plain maximum against zero, so the kernel must match the reference
// bit for bit: the absolute error budget is zero.
static constexpr float max_abs_error = 0.0f;
static constexpr float min_cosine = 0.999999f;

// ============================================================================
// relu_f32_f32_1x8vl_rvv
// ============================================================================

class Test_relu_f32_f32_1x8vl_rvv : public testing::TestWithParam<ActivationTestParams>
{
};

TEST_P(Test_relu_f32_f32_1x8vl_rvv, Correctness)
{
    run_activation_test<float>(tqt_run_relu_f32_f32_1x8vl_rvv, ActivationKind::Relu, GetParam(),
                               max_abs_error, min_cosine);
}

TEST_P(Test_relu_f32_f32_1x8vl_rvv, InPlace)
{
    run_activation_inplace_test<float>(tqt_run_relu_f32_f32_1x8vl_rvv, ActivationKind::Relu,
                                       GetParam(), max_abs_error, min_cosine);
}

TEST(Test_relu_f32_f32_1x8vl_rvv, Steps)
{
    EXPECT_EQ(tqt_get_m_step_relu_f32_f32_1x8vl_rvv(), 1u);
    EXPECT_EQ(tqt_get_n_step_relu_f32_f32_1x8vl_rvv(), 8u * (csrr_vlenb() / sizeof(float)));
}

INSTANTIATE_TEST_SUITE_P(ActivationTests, Test_relu_f32_f32_1x8vl_rvv, activation_test_values(),
                         ActivationTestParamNameGenerator());
