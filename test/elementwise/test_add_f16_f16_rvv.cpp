//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/elementwise/add_f16_f16/addvs_f16_f16_f16/tqt_addvs_f16_f16_f16_8vl_rvv.h"
#include "src/elementwise/add_f16_f16/addvv_f16_f16_f16/tqt_addvv_f16_f16_f16_8vl_rvv.h"
#include "test/common/elementwise_test_runner.h"
#include "test/common/reference_elementwise.h"

using namespace torq_tile::test;

static constexpr ElementwiseRandomConfig kInputConfig{-10.0f, 10.0f, 42};
static constexpr ElementwiseTolerance kTolerance{1.0f, 0.9999f};

class Test_addvv_f16_f16_f16_8vl_rvv : public testing::TestWithParam<ElementwiseTestParams>
{
};

TEST_P(Test_addvv_f16_f16_f16_8vl_rvv, Correctness)
{
    const ElementwiseVVTestConfig config{kInputConfig, std::nullopt, kTolerance};
    run_elementwise_vv_test<float16_t>(GetParam().n, reference_addvv<float16_t>,
                                       tqt_run_addvv_f16_f16_f16_8vl_rvv, config);
}

INSTANTIATE_TEST_SUITE_P(ElementwiseTests, Test_addvv_f16_f16_f16_8vl_rvv,
                         common_elementwise_test_values(), ElementwiseTestParamNameGenerator());

class Test_addvs_f16_f16_f16_8vl_rvv : public testing::TestWithParam<ElementwiseTestParams>
{
};

TEST_P(Test_addvs_f16_f16_f16_8vl_rvv, Correctness)
{
    const ElementwiseVSTestConfig<float16_t> config{kInputConfig, static_cast<float16_t>(3.14f),
                                                    kTolerance};
    run_elementwise_vs_test<float16_t>(GetParam().n, reference_addvs<float16_t>,
                                       tqt_run_addvs_f16_f16_f16_8vl_rvv, config);
}

INSTANTIATE_TEST_SUITE_P(ElementwiseTests, Test_addvs_f16_f16_f16_8vl_rvv,
                         common_elementwise_test_values(), ElementwiseTestParamNameGenerator());
