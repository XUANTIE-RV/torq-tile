//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/elementwise/div_f32_f32/divvs_f32_f32_f32/tqt_divvs_f32_f32_f32_8vl_rvv.h"
#include "src/elementwise/div_f32_f32/divvv_f32_f32_f32/tqt_divvv_f32_f32_f32_8vl_rvv.h"
#include "test/common/elementwise_test_runner.h"
#include "test/common/reference_elementwise.h"

using namespace torq_tile::test;

static constexpr ElementwiseRandomConfig kInputConfig{-10.0f, 10.0f, 42};
static constexpr ElementwiseTolerance kTolerance{1e-5f, 0.999999f};
// Keep divisors away from zero to avoid unstable division results.
static constexpr ElementwiseRandomConfig kDivisorConfig{1.0f, 10.0f, 43};

class Test_divvv_f32_f32_f32_8vl_rvv : public testing::TestWithParam<ElementwiseTestParams>
{
};

TEST_P(Test_divvv_f32_f32_f32_8vl_rvv, Correctness)
{
    const ElementwiseVVTestConfig config{kInputConfig, kDivisorConfig, kTolerance};
    run_elementwise_vv_test<float>(GetParam().n, reference_divvv<float>,
                                   tqt_run_divvv_f32_f32_f32_8vl_rvv, config);
}

INSTANTIATE_TEST_SUITE_P(ElementwiseTests, Test_divvv_f32_f32_f32_8vl_rvv,
                         common_elementwise_test_values(), ElementwiseTestParamNameGenerator());

class Test_divvs_f32_f32_f32_8vl_rvv : public testing::TestWithParam<ElementwiseTestParams>
{
};

TEST_P(Test_divvs_f32_f32_f32_8vl_rvv, Correctness)
{
    const ElementwiseVSTestConfig<float> config{kInputConfig, 3.14f, kTolerance};
    run_elementwise_vs_test<float>(GetParam().n, reference_divvs<float>,
                                   tqt_run_divvs_f32_f32_f32_8vl_rvv, config);
}

INSTANTIATE_TEST_SUITE_P(ElementwiseTests, Test_divvs_f32_f32_f32_8vl_rvv,
                         common_elementwise_test_values(), ElementwiseTestParamNameGenerator());

TEST(Test_divvs_f32_f32_f32_8vl_rvv, SubnormalDivisors)
{
    std::vector<float> divisors;
    for (uint32_t magnitude : {0x00000001u, 0x00000002u, 0x00200000u, 0x00400000u, 0x007fffffu}) {
        for (uint32_t sign : {0u, 0x80000000u}) {
            const float divisor = elementwise_test_value_from_bits<float>(magnitude | sign);
            ASSERT_NO_FATAL_FAILURE(
                run_divvs_subnormal_identity_test(divisor, tqt_run_divvs_f32_f32_f32_8vl_rvv));
            divisors.push_back(divisor);
        }
    }
    run_divvs_boundary_test(divisors, true, tqt_get_n_step_divvs_f32_f32_f32_8vl_rvv(),
                            elementwise_test_value_from_bits<float>(0x00000001u),
                            elementwise_test_value_from_bits<float>(0x7f7fffffu),
                            tqt_run_divvs_f32_f32_f32_8vl_rvv);
}

TEST(Test_divvs_f32_f32_f32_8vl_rvv, ReciprocalPathPreserved)
{
    std::vector<float> divisors;
    for (uint32_t magnitude : {0x00000000u, 0x00800000u, 0x00800001u, 0x3f800000u, 0x40400000u,
                               0x7f7fffffu, 0x7f800000u, 0x7fc00000u}) {
        for (uint32_t sign : {0u, 0x80000000u}) {
            divisors.push_back(elementwise_test_value_from_bits<float>(magnitude | sign));
        }
    }
    run_divvs_boundary_test(divisors, false, tqt_get_n_step_divvs_f32_f32_f32_8vl_rvv(),
                            elementwise_test_value_from_bits<float>(0x00000001u),
                            elementwise_test_value_from_bits<float>(0x7f7fffffu),
                            tqt_run_divvs_f32_f32_f32_8vl_rvv);
}
