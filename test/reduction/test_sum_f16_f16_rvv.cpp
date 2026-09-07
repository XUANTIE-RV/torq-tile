//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/reduction/sum_f16_f16/sumcol_f16_f16/tqt_sumcol_f16_f16_2x4vl_rvv.h"
#include "src/reduction/sum_f16_f16/sumrow_f16_f16/tqt_sumrow_f16_f16_4x2vl_rvv.h"
#include "test/common/reduction_test_runner.h"
#include "test/common/reference_reduction.h"

using namespace torq_tile::test;

static const ReductionTestConfig<float16_t> kTestConfig{{-2.0f, 2.0f, 42},
                                                        static_cast<float16_t>(1.0f)};

static ReductionTolerance reduction_tolerance(size_t reduced_length)
{
    return {TQT_MAX(0.5f, reduced_length * 5e-4f), 0.999f};
}

class Test_sumrow_f16_f16_4x2vl_rvv : public testing::TestWithParam<ReductionTestParams>
{
};

TEST_P(Test_sumrow_f16_f16_4x2vl_rvv, WithoutInit)
{
    run_reduction_test<float16_t>(GetParam(), ReductionAxis::kRow, ReductionInit::kNone,
                                  reference_sumrow_f16, tqt_run_sumrow_f16_f16_4x2vl_rvv,
                                  kTestConfig, reduction_tolerance);
}

TEST_P(Test_sumrow_f16_f16_4x2vl_rvv, WithInit)
{
    run_reduction_test<float16_t>(GetParam(), ReductionAxis::kRow, ReductionInit::kWithInit,
                                  reference_sumrow_f16, tqt_run_sumrow_f16_f16_4x2vl_rvv,
                                  kTestConfig, reduction_tolerance);
}

INSTANTIATE_TEST_SUITE_P(ReductionTests, Test_sumrow_f16_f16_4x2vl_rvv,
                         common_reduction_test_values(), ReductionTestParamNameGenerator());

class Test_sumcol_f16_f16_2x4vl_rvv : public testing::TestWithParam<ReductionTestParams>
{
};

TEST_P(Test_sumcol_f16_f16_2x4vl_rvv, WithoutInit)
{
    run_reduction_test<float16_t>(GetParam(), ReductionAxis::kColumn, ReductionInit::kNone,
                                  reference_sumcol_f16, tqt_run_sumcol_f16_f16_2x4vl_rvv,
                                  kTestConfig, reduction_tolerance);
}

TEST_P(Test_sumcol_f16_f16_2x4vl_rvv, WithInit)
{
    run_reduction_test<float16_t>(GetParam(), ReductionAxis::kColumn, ReductionInit::kWithInit,
                                  reference_sumcol_f16, tqt_run_sumcol_f16_f16_2x4vl_rvv,
                                  kTestConfig, reduction_tolerance);
}

INSTANTIATE_TEST_SUITE_P(ReductionTests, Test_sumcol_f16_f16_2x4vl_rvv,
                         common_reduction_test_values(), ReductionTestParamNameGenerator());
