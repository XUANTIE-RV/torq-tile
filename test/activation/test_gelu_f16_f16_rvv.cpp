//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/activation/gelu_f16_f16/gelu_f16_f16/tqt_gelu_f16_f16_1x2vl_rvv.h"
#include "test/common/activation_test_runner.h"

using namespace torq_tile::test;

// Exhausting every finite FP16 input in [-10, 10] measures a 9.8e-4 maximum
// absolute error.
static constexpr float max_abs_error = 5e-3f;
static constexpr float min_cosine = 0.9999f;

class Test_gelu_f16_f16_1x2vl_rvv : public testing::TestWithParam<ActivationTestParams>
{
};

TEST_P(Test_gelu_f16_f16_1x2vl_rvv, Correctness)
{
    run_activation_test<float16_t>(tqt_run_gelu_f16_f16_1x2vl_rvv, ActivationKind::Gelu, GetParam(),
                                   max_abs_error, min_cosine);
}

TEST_P(Test_gelu_f16_f16_1x2vl_rvv, InPlace)
{
    run_activation_inplace_test<float16_t>(tqt_run_gelu_f16_f16_1x2vl_rvv, ActivationKind::Gelu,
                                           GetParam(), max_abs_error, min_cosine);
}

TEST(Test_gelu_f16_f16_1x2vl_rvv, EdgeValues)
{
    run_activation_edge_values_test<float16_t>(tqt_run_gelu_f16_f16_1x2vl_rvv, ActivationKind::Gelu,
                                               max_abs_error, min_cosine);
}

TEST(Test_gelu_f16_f16_1x2vl_rvv, Steps)
{
    EXPECT_EQ(tqt_get_m_step_gelu_f16_f16_1x2vl_rvv(), 1u);
    EXPECT_EQ(tqt_get_n_step_gelu_f16_f16_1x2vl_rvv(), 4u * (csrr_vlenb() / sizeof(float)));
}

INSTANTIATE_TEST_SUITE_P(ActivationTests, Test_gelu_f16_f16_1x2vl_rvv, activation_test_values(),
                         ActivationTestParamNameGenerator());
