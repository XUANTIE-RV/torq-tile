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

TEST(Test_divvs_f16_f16_f16_8vl_rvv, AllSubnormalDivisors)
{
    for (uint16_t magnitude = 1; magnitude < 0x0400; ++magnitude) {
        for (uint16_t sign : {0x0000, 0x8000}) {
            SCOPED_TRACE(testing::Message() << "divisor_bits=" << (magnitude | sign));
            const float16_t divisor = elementwise_test_value_from_bits<float16_t>(
                static_cast<uint16_t>(magnitude | sign));
            ASSERT_NO_FATAL_FAILURE(
                run_divvs_subnormal_identity_test(divisor, tqt_run_divvs_f16_f16_f16_8vl_rvv));
        }
    }
}

TEST(Test_divvs_f16_f16_f16_8vl_rvv, SubnormalBoundaries)
{
    std::vector<float16_t> divisors;
    for (uint16_t magnitude : {0x0001, 0x0002, 0x0100, 0x0200, 0x03ff}) {
        for (uint16_t sign : {0x0000, 0x8000}) {
            divisors.push_back(elementwise_test_value_from_bits<float16_t>(
                static_cast<uint16_t>(magnitude | sign)));
        }
    }
    run_divvs_boundary_test(divisors, true, tqt_get_n_step_divvs_f16_f16_f16_8vl_rvv(),
                            elementwise_test_value_from_bits<float16_t>(uint16_t{0x0001}),
                            elementwise_test_value_from_bits<float16_t>(uint16_t{0x7bff}),
                            tqt_run_divvs_f16_f16_f16_8vl_rvv);
}

TEST(Test_divvs_f16_f16_f16_8vl_rvv, ReciprocalPathPreserved)
{
    std::vector<float16_t> divisors;
    for (uint16_t magnitude : {0x0000, 0x0400, 0x0401, 0x3c00, 0x4200, 0x7bff, 0x7c00, 0x7e00}) {
        for (uint16_t sign : {0x0000, 0x8000}) {
            divisors.push_back(elementwise_test_value_from_bits<float16_t>(
                static_cast<uint16_t>(magnitude | sign)));
        }
    }
    run_divvs_boundary_test(divisors, false, tqt_get_n_step_divvs_f16_f16_f16_8vl_rvv(),
                            elementwise_test_value_from_bits<float16_t>(uint16_t{0x0001}),
                            elementwise_test_value_from_bits<float16_t>(uint16_t{0x7bff}),
                            tqt_run_divvs_f16_f16_f16_8vl_rvv);
}
