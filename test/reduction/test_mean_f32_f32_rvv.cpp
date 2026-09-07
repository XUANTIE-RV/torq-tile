//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/reduction/mean_f32_f32/meancol_f32_f32/tqt_meancol_f32_f32_2x8vl_rvv.h"
#include "src/reduction/mean_f32_f32/meanrow_f32_f32/tqt_meanrow_f32_f32_4x4vl_rvv.h"
#include "test/common/reduction_test_runner.h"
#include "test/common/reference_reduction.h"

using namespace torq_tile::test;

static constexpr ReductionTestConfig<float> kTestConfig{{-10.0f, 10.0f, 42}, 1.0f};

static ReductionTolerance row_tolerance(size_t)
{
    return {1e-4f, 0.999999f};
}

static ReductionTolerance column_tolerance(size_t)
{
    return {1e-5f, 0.999999f};
}

class Test_meanrow_f32_f32_4x4vl_rvv : public testing::TestWithParam<ReductionTestParams>
{
};

TEST_P(Test_meanrow_f32_f32_4x4vl_rvv, WithoutInit)
{
    run_reduction_test<float>(GetParam(), ReductionAxis::kRow, ReductionInit::kNone,
                              reference_meanrow_f32, tqt_run_meanrow_f32_f32_4x4vl_rvv, kTestConfig,
                              row_tolerance);
}

TEST_P(Test_meanrow_f32_f32_4x4vl_rvv, WithInit)
{
    run_reduction_test<float>(GetParam(), ReductionAxis::kRow, ReductionInit::kWithInit,
                              reference_meanrow_f32, tqt_run_meanrow_f32_f32_4x4vl_rvv, kTestConfig,
                              row_tolerance);
}

INSTANTIATE_TEST_SUITE_P(ReductionTests, Test_meanrow_f32_f32_4x4vl_rvv,
                         common_reduction_test_values(), ReductionTestParamNameGenerator());

class Test_meancol_f32_f32_2x8vl_rvv : public testing::TestWithParam<ReductionTestParams>
{
};

TEST_P(Test_meancol_f32_f32_2x8vl_rvv, WithoutInit)
{
    run_reduction_test<float>(GetParam(), ReductionAxis::kColumn, ReductionInit::kNone,
                              reference_meancol_f32, tqt_run_meancol_f32_f32_2x8vl_rvv, kTestConfig,
                              column_tolerance);
}

TEST_P(Test_meancol_f32_f32_2x8vl_rvv, WithInit)
{
    run_reduction_test<float>(GetParam(), ReductionAxis::kColumn, ReductionInit::kWithInit,
                              reference_meancol_f32, tqt_run_meancol_f32_f32_2x8vl_rvv, kTestConfig,
                              column_tolerance);
}

INSTANTIATE_TEST_SUITE_P(ReductionTests, Test_meancol_f32_f32_2x8vl_rvv,
                         common_reduction_test_values(), ReductionTestParamNameGenerator());
