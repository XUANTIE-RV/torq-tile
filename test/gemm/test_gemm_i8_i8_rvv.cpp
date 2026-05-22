//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv.h"
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv.h"
#endif
#include "src/gemm/gemm_i8_i8/gemm_clamp_i32_i8_i8/tqt_gemm_clamp_i32_i8_i8_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_clamp_i32_i8p_i8p/tqt_gemm_clamp_i32_i8p_i8p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p/tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv.h"
#include "test/common/gemm_test_runner.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

// ============================================================================
// Common test parameters for int8 GEMM
// ============================================================================

static auto int8_common_test_values()
{
    return testing::Values(GemmTestParams(32, 32, 32),   //
                           GemmTestParams(3, 8, 11),     //
                           GemmTestParams(5, 11, 10),    //
                           GemmTestParams(9, 19, 23),    //
                           GemmTestParams(15, 32, 25),   //
                           GemmTestParams(60, 63, 127),  //
                           GemmTestParams(32, 32, 32, "WithC").set_c(true),
                           GemmTestParams(60, 63, 127, "WithC").set_c(true));
}

// ============================================================================
// gemm_clamp_i32_i8_i8_4x1vl_rvv (int8 inputs -> int32 output)
// ============================================================================

static NonPackedGemmFuncs<int32_t> make_nonpacked_clamp_i32_i8_i8_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        tqt_get_n_step_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        tqt_get_a_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        tqt_get_b_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        tqt_get_c_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        nullptr,  // no bias offset
        tqt_get_d_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        tqt_get_d_size_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        // Wrap run_gemm to ignore bias parameter
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, int32_t clamp_min, int32_t clamp_max) {
            (void)bias;  // unused
            tqt_run_gemm_clamp_i32_i8_i8_4x1vl_rvv(m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C,
                                                   ldc, D, ldd, clamp_min, clamp_max);
        },
    };
}

class Test_gemm_clamp_i32_i8_i8_4x1vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_clamp_i32_i8_i8_4x1vl_rvv, Correctness)
{
    auto funcs = make_nonpacked_clamp_i32_i8_i8_4x1vl_rvv();
    run_clamp_i32_i8_gemm_test(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_clamp_i32_i8_i8_4x1vl_rvv, int8_common_test_values(),
                         GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv
// int8 (A asymmetric) x int8 (B per-channel symmetric, transposed [N,K]) -> int8 with 1xN bias
// ============================================================================

static NonPackedRequantizedGemmFuncs<tqt_qai8_qai8_qsi8cx_params>
make_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
    };
}

class Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv_funcs();
    run_1xnbias_reqi32toi8_gemm_test(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv
// int8 (A per-channel symmetric) x int8 (B global asymmetric, [K,N]) -> int8 with Mx1 bias
// ============================================================================

static NonPackedRequantizedGemmFuncs<tqt_qai8_qsi8dx_qai8_params>
make_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_a_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_b_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_bias_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_d_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_run_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    };
}

class Test_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv, Correctness)
{
    auto funcs = make_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv_funcs();
    run_mx1bias_reqi32toi8_gemm_test(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_clamp_i32_i8p_i8p_4x1vl_rvv (packed int8 inputs -> int32 output)
// ============================================================================

static PackedGemmFuncs<int32_t> make_packed_clamp_i32_i8p_i8p_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_n_step_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_c_offset_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        nullptr,  // no bias offset
        tqt_get_d_offset_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_d_size_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_run_a_pack_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_run_b_pack_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        // Wrap run_gemm to ignore bias parameter
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, int32_t clamp_min, int32_t clamp_max) {
            (void)bias;  // unused
            tqt_run_gemm_clamp_i32_i8p_i8p_4x1vl_rvv(m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C,
                                                     ldc, D, ldd, clamp_min, clamp_max);
        },
    };
}

class Test_gemm_clamp_i32_i8p_i8p_4x1vl_rvv : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_clamp_i32_i8p_i8p_4x1vl_rvv, Correctness)
{
    auto funcs = make_packed_clamp_i32_i8p_i8p_4x1vl_rvv();
    run_packed_clamp_i32_i8_gemm_test(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv
// packed: int8 (A asymmetric packed) x int8 (B per-channel symmetric packed) -> int8 with 1xN bias
// ============================================================================

static PackedRequantizedGemmFuncs<tqt_qai8_qai8_qsi8cx_params>
make_packed_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    };
}

class Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_packed_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv_funcs();
    run_packed_1xnbias_reqi32toi8_gemm_test(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv
// packed: int8 (A per-channel symmetric packed) x int8 (B global asymmetric packed) -> int8
// with Mx1 bias
// ============================================================================

static PackedRequantizedGemmFuncs<tqt_qai8_qsi8dx_qai8_params>
make_packed_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_c_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_bias_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_d_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_a_pack_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_b_pack_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
    };
}

class Test_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv, Correctness)
{
    auto funcs = make_packed_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv_funcs();
    run_packed_mx1bias_reqi32toi8_gemm_test(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv
// B-only-packed: int8 (A asymmetric original) x int8 (B per-channel symmetric packed) -> int8
// with 1xN bias
// ============================================================================

static BOnlyPackedRequantizedGemmFuncs<tqt_qai8_qai8_qsi8cx_params>
make_b_only_packed_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    };
}

class Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_b_only_packed_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv_funcs();
    run_b_only_packed_1xnbias_reqi32toi8_gemm_test(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv
// NonPacked a_bt: int8 (A per-row asymmetric) x int8 (B per-channel symmetric, transposed)
// -> f32 with 1xN bias
// ============================================================================

static NonPackedClampFpGemmFuncs<float> make_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
    };
}

class Test_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv_funcs();
    run_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv
// Packed ap_bp: int8 (A per-row asymmetric packed) x int8 (B per-channel symmetric packed)
// -> f32 with 1xN bias
// ============================================================================

static PackedClampFpGemmFuncs<float> make_packed_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
    };
}

class Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_packed_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv_funcs();
    run_packed_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
// ============================================================================
// gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv
// NonPacked a_bt: int8 inputs -> f16 output with 1xN bias
// ============================================================================

static NonPackedClampFpGemmFuncs<float16_t> make_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    };
}

class Test_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv_funcs();
    run_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv
// Packed ap_bp: int8 inputs -> f16 output with 1xN bias
// ============================================================================

static PackedClampFpGemmFuncs<float16_t>
make_packed_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
    };
}

class Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_packed_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv_funcs();
    run_packed_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
                         int8_common_test_values(), GemmTestParamNameGenerator());
#endif  // __riscv_zfh && __riscv_zvfh
