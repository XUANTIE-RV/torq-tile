//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cfloat>

#include "src/gemm/gemm_f64_f64/dgemm_f64_f64_f64/tqt_dgemm_f64_f64_f64_8x3vl_rvv.h"
#include "src/gemm/gemm_f64_f64/dgemm_f64_f64_f64t/tqt_dgemm_f64_f64_f64t_8x3vl_rvv.h"
#include "src/gemm/gemm_f64_f64/dgemm_f64_f64t_f64/tqt_dgemm_f64_f64t_f64_8x3vl_rvv.h"
#include "src/gemm/gemm_f64_f64/dgemm_f64_f64t_f64t/tqt_dgemm_f64_f64t_f64t_8x3vl_rvv.h"
#include "test/common/sgemm_test_runner.h"

using namespace torq_tile::test;

// ============================================================================
// DGEMM test parameter set
// ============================================================================

static auto dgemm_test_values()
{
    return testing::Values(
        SgemmTestParams(32, 32, 32, "Default"), SgemmTestParams(17, 19, 23, "Default"),
        SgemmTestParams(60, 61, 127, "Default"),
        SgemmTestParams(32, 32, 32, "AB1_AddC").set_beta(1.0f).set_c(true),
        SgemmTestParams(31, 35, 63, "AB1_AddC").set_beta(1.0f).set_c(true),
        SgemmTestParams(32, 32, 32, "A1_B2p5").set_beta(2.5f).set_c(true),
        SgemmTestParams(32, 32, 32, "A2p5_B0").set_alpha(2.5f),
        SgemmTestParams(32, 32, 32, "A2p5_B1").set_alpha(2.5f).set_beta(1.0f).set_c(true),
        SgemmTestParams(60, 61, 127, "GenAB").set_alpha(-1.5f).set_beta(0.75f).set_c(true),
        SgemmTestParams(15, 17, 256, "TileK64")
            .set_tile(0, 0, 64)
            .set_alpha(2.0f)
            .set_beta(0.5f)
            .set_c(true),
        SgemmTestParams(31, 35, 63, "OrderKMN")
            .set_loop_order(LoopDim::K, LoopDim::M, LoopDim::N)
            .set_alpha(1.5f)
            .set_beta(0.5f)
            .set_c(true));
}

// ============================================================================
// DGEMM A=RowMajor, B=RowMajor, Tile=8x3vl
// ============================================================================

static NonPackedSgemmFuncs<double> make_dgemm_f64_f64_f64_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_dgemm_f64_f64_f64_8x3vl_rvv,   tqt_get_n_step_dgemm_f64_f64_f64_8x3vl_rvv,
        tqt_get_a_offset_dgemm_f64_f64_f64_8x3vl_rvv, tqt_get_b_offset_dgemm_f64_f64_f64_8x3vl_rvv,
        tqt_get_c_offset_dgemm_f64_f64_f64_8x3vl_rvv, tqt_get_d_offset_dgemm_f64_f64_f64_8x3vl_rvv,
        tqt_get_d_size_dgemm_f64_f64_f64_8x3vl_rvv,   tqt_run_dgemm_f64_f64_f64_8x3vl_rvv,
    };
}

class Test_dgemm_f64_f64_f64_8x3vl_rvv : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_dgemm_f64_f64_f64_8x3vl_rvv, Correctness)
{
    auto funcs = make_dgemm_f64_f64_f64_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_test<double, double, ALayout::kRowMajor, BLayout::kRowMajor>(GetParam(),
                                                                                     funcs);
}

INSTANTIATE_TEST_SUITE_P(DgemmTests, Test_dgemm_f64_f64_f64_8x3vl_rvv, dgemm_test_values(),
                         SgemmTestParamNameGenerator());

// ============================================================================
// DGEMM A=RowMajor, B=Transposed, Tile=8x3vl
// ============================================================================

static NonPackedSgemmFuncs<double> make_dgemm_f64_f64_f64t_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_n_step_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_a_offset_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_b_offset_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_c_offset_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_d_offset_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_d_size_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_run_dgemm_f64_f64_f64t_8x3vl_rvv,
    };
}

class Test_dgemm_f64_f64_f64t_8x3vl_rvv : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_dgemm_f64_f64_f64t_8x3vl_rvv, Correctness)
{
    auto funcs = make_dgemm_f64_f64_f64t_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_test<double, double, ALayout::kRowMajor, BLayout::kTransposed>(GetParam(),
                                                                                       funcs);
}

INSTANTIATE_TEST_SUITE_P(DgemmTests, Test_dgemm_f64_f64_f64t_8x3vl_rvv, dgemm_test_values(),
                         SgemmTestParamNameGenerator());

// ============================================================================
// DGEMM A=Transposed, B=RowMajor, Tile=8x3vl
// ============================================================================

static NonPackedSgemmFuncs<double> make_dgemm_f64_f64t_f64_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_n_step_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_a_offset_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_b_offset_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_c_offset_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_d_offset_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_d_size_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_run_dgemm_f64_f64t_f64_8x3vl_rvv,
    };
}

class Test_dgemm_f64_f64t_f64_8x3vl_rvv : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_dgemm_f64_f64t_f64_8x3vl_rvv, Correctness)
{
    auto funcs = make_dgemm_f64_f64t_f64_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_test<double, double, ALayout::kTransposed, BLayout::kRowMajor>(GetParam(),
                                                                                       funcs);
}

INSTANTIATE_TEST_SUITE_P(DgemmTests, Test_dgemm_f64_f64t_f64_8x3vl_rvv, dgemm_test_values(),
                         SgemmTestParamNameGenerator());

// ============================================================================
// DGEMM A=Transposed, B=Transposed, Tile=8x3vl
// ============================================================================

static NonPackedSgemmFuncs<double> make_dgemm_f64_f64t_f64t_8x3vl_rvv_funcs()
{
    return {
        tqt_get_m_step_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_n_step_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_a_offset_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_b_offset_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_c_offset_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_d_offset_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_d_size_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_run_dgemm_f64_f64t_f64t_8x3vl_rvv,
    };
}

class Test_dgemm_f64_f64t_f64t_8x3vl_rvv : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_dgemm_f64_f64t_f64t_8x3vl_rvv, Correctness)
{
    auto funcs = make_dgemm_f64_f64t_f64t_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_test<double, double, ALayout::kTransposed, BLayout::kTransposed>(GetParam(),
                                                                                         funcs);
}

INSTANTIATE_TEST_SUITE_P(DgemmTests, Test_dgemm_f64_f64t_f64t_8x3vl_rvv, dgemm_test_values(),
                         SgemmTestParamNameGenerator());

// ============================================================================
// DGEMM C==D alias tests (all 4 layout variants)
// ============================================================================

static auto dgemm_alias_test_values()
{
    return testing::Values(
        SgemmTestParams(32, 32, 32, "Alias_B1").set_alpha(1.0f).set_beta(1.0f).set_c(true),
        SgemmTestParams(17, 19, 23, "Alias_B1").set_alpha(1.0f).set_beta(1.0f).set_c(true),
        SgemmTestParams(60, 61, 127, "Alias_GenAB").set_alpha(-1.5f).set_beta(0.75f).set_c(true),
        SgemmTestParams(32, 32, 32, "Alias_A2p5_B2p5").set_alpha(2.5f).set_beta(2.5f).set_c(true));
}

class Test_dgemm_f64_f64_f64_8x3vl_rvv_Alias : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_dgemm_f64_f64_f64_8x3vl_rvv_Alias, CDAlias)
{
    auto funcs = make_dgemm_f64_f64_f64_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_alias_test<double, double, ALayout::kRowMajor, BLayout::kRowMajor>(
        GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(DgemmAliasTests, Test_dgemm_f64_f64_f64_8x3vl_rvv_Alias,
                         dgemm_alias_test_values(), SgemmTestParamNameGenerator());

class Test_dgemm_f64_f64_f64t_8x3vl_rvv_Alias : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_dgemm_f64_f64_f64t_8x3vl_rvv_Alias, CDAlias)
{
    auto funcs = make_dgemm_f64_f64_f64t_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_alias_test<double, double, ALayout::kRowMajor, BLayout::kTransposed>(
        GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(DgemmAliasTests, Test_dgemm_f64_f64_f64t_8x3vl_rvv_Alias,
                         dgemm_alias_test_values(), SgemmTestParamNameGenerator());

class Test_dgemm_f64_f64t_f64_8x3vl_rvv_Alias : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_dgemm_f64_f64t_f64_8x3vl_rvv_Alias, CDAlias)
{
    auto funcs = make_dgemm_f64_f64t_f64_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_alias_test<double, double, ALayout::kTransposed, BLayout::kRowMajor>(
        GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(DgemmAliasTests, Test_dgemm_f64_f64t_f64_8x3vl_rvv_Alias,
                         dgemm_alias_test_values(), SgemmTestParamNameGenerator());

class Test_dgemm_f64_f64t_f64t_8x3vl_rvv_Alias : public testing::TestWithParam<SgemmTestParams>
{
};

TEST_P(Test_dgemm_f64_f64t_f64t_8x3vl_rvv_Alias, CDAlias)
{
    auto funcs = make_dgemm_f64_f64t_f64t_8x3vl_rvv_funcs();
    run_nonpacked_sgemm_alias_test<double, double, ALayout::kTransposed, BLayout::kTransposed>(
        GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(DgemmAliasTests, Test_dgemm_f64_f64t_f64t_8x3vl_rvv_Alias,
                         dgemm_alias_test_values(), SgemmTestParamNameGenerator());
