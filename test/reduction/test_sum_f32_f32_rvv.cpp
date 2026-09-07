//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/reduction/sum_f32_f32/sumcol_f32_f32/tqt_sumcol_f32_f32_2x8vl_rvv.h"
#include "src/reduction/sum_f32_f32/sumrow_f32_f32/tqt_sumrow_f32_f32_4x4vl_rvv.h"
#include "test/common/reduction_test_runner.h"
#include "test/common/reference_reduction.h"

using namespace torq_tile::test;

static constexpr ReductionTestConfig<float> kTestConfig{{-10.0f, 10.0f, 42}, 1.0f};

static ReductionTolerance reduction_tolerance(size_t reduced_length)
{
    return {TQT_MAX(1e-4f, reduced_length * 2e-6f), 0.999999f};
}

class Test_sumrow_f32_f32_4x4vl_rvv : public testing::TestWithParam<ReductionTestParams>
{
};

TEST_P(Test_sumrow_f32_f32_4x4vl_rvv, WithoutInit)
{
    run_reduction_test<float>(GetParam(), ReductionAxis::kRow, ReductionInit::kNone,
                              reference_sumrow_f32, tqt_run_sumrow_f32_f32_4x4vl_rvv, kTestConfig,
                              reduction_tolerance);
}

TEST_P(Test_sumrow_f32_f32_4x4vl_rvv, WithInit)
{
    run_reduction_test<float>(GetParam(), ReductionAxis::kRow, ReductionInit::kWithInit,
                              reference_sumrow_f32, tqt_run_sumrow_f32_f32_4x4vl_rvv, kTestConfig,
                              reduction_tolerance);
}

INSTANTIATE_TEST_SUITE_P(ReductionTests, Test_sumrow_f32_f32_4x4vl_rvv,
                         common_reduction_test_values(), ReductionTestParamNameGenerator());

class Test_sumcol_f32_f32_2x8vl_rvv : public testing::TestWithParam<ReductionTestParams>
{
};

TEST_P(Test_sumcol_f32_f32_2x8vl_rvv, WithoutInit)
{
    run_reduction_test<float>(GetParam(), ReductionAxis::kColumn, ReductionInit::kNone,
                              reference_sumcol_f32, tqt_run_sumcol_f32_f32_2x8vl_rvv, kTestConfig,
                              reduction_tolerance);
}

TEST_P(Test_sumcol_f32_f32_2x8vl_rvv, WithInit)
{
    run_reduction_test<float>(GetParam(), ReductionAxis::kColumn, ReductionInit::kWithInit,
                              reference_sumcol_f32, tqt_run_sumcol_f32_f32_2x8vl_rvv, kTestConfig,
                              reduction_tolerance);
}

INSTANTIATE_TEST_SUITE_P(ReductionTests, Test_sumcol_f32_f32_2x8vl_rvv,
                         common_reduction_test_values(), ReductionTestParamNameGenerator());
