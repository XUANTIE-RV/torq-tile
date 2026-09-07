//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/elementwise/div_f16_f16/divvs_f16_f16_f16/tqt_divvs_f16_f16_f16_8vl_rvv.h"
#include "src/elementwise/div_f16_f16/divvv_f16_f16_f16/tqt_divvv_f16_f16_f16_8vl_rvv.h"
#include "test/common/elementwise_test_runner.h"
#include "test/common/reference_elementwise.h"

using namespace torq_tile::test;

static constexpr ElementwiseRandomConfig kInputConfig{-10.0f, 10.0f, 42};
static constexpr ElementwiseTolerance kTolerance{1.0f, 0.9999f};
// Keep divisors away from zero to avoid unstable division results.
static constexpr ElementwiseRandomConfig kDivisorConfig{1.0f, 10.0f, 43};

class Test_divvv_f16_f16_f16_8vl_rvv : public testing::TestWithParam<ElementwiseTestParams>
{
};

TEST_P(Test_divvv_f16_f16_f16_8vl_rvv, Correctness)
{
    const ElementwiseVVTestConfig config{kInputConfig, kDivisorConfig, kTolerance};
    run_elementwise_vv_test<float16_t>(GetParam().n, reference_divvv<float16_t>,
                                       tqt_run_divvv_f16_f16_f16_8vl_rvv, config);
}

INSTANTIATE_TEST_SUITE_P(ElementwiseTests, Test_divvv_f16_f16_f16_8vl_rvv,
                         common_elementwise_test_values(), ElementwiseTestParamNameGenerator());

class Test_divvs_f16_f16_f16_8vl_rvv : public testing::TestWithParam<ElementwiseTestParams>
{
};

TEST_P(Test_divvs_f16_f16_f16_8vl_rvv, Correctness)
{
    const ElementwiseVSTestConfig<float16_t> config{kInputConfig, static_cast<float16_t>(3.14f),
                                                    kTolerance};
    run_elementwise_vs_test<float16_t>(GetParam().n, reference_divvs<float16_t>,
                                       tqt_run_divvs_f16_f16_f16_8vl_rvv, config);
}

INSTANTIATE_TEST_SUITE_P(ElementwiseTests, Test_divvs_f16_f16_f16_8vl_rvv,
                         common_elementwise_test_values(), ElementwiseTestParamNameGenerator());
