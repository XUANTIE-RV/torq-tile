//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "torq_tile/tqt_compile_config.h"
#if TORQ_TILE_HAS_RVV_F16
#include "src/gemm/gemm_i8_i8/gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt/tqt_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp/tqt_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv.h"
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
                           GemmTestParams(2, 8, 11),     //
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
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv, float, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, false>(),
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

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv_funcs();
    run_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
    with_clamp_dimensions(int8_common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

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
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv, float, 13, 11, 12,
                       14, ClampSource::kColumnBias, false, 5, false>(),
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

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_packed_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv_funcs();
    run_packed_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
    with_clamp_dimensions(int8_common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());

#if TORQ_TILE_HAS_RVV_F16
// ============================================================================
// gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv
// NonPacked a_bt: int8 inputs -> f16 output with 1xN bias
// ============================================================================

static NonPackedClampFpGemmFuncs<float16_t>
make_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_d_size_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv, float16_t, 13,
                       11, 12, 14, ClampSource::kColumnBias, false, 5, false, float>(),
    };
}

class Test_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv_funcs();
    run_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv_funcs();
    run_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    with_clamp_dimensions(int8_common_test_values(),
                          tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv
// Packed ap_bp: int8 inputs -> f16 output with 1xN bias
// ============================================================================

static PackedClampFpGemmFuncs<float16_t>
make_packed_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_d_size_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        clamp_test_run<tqt_run_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv, float16_t, 13,
                       11, 12, 14, ClampSource::kColumnBias, false, 5, false, float>(),
    };
}

class Test_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_packed_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv_funcs();
    run_packed_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_packed_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv_funcs();
    run_packed_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
    with_clamp_dimensions(int8_common_test_values(),
                          tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());
#endif  // TORQ_TILE_HAS_RVV_F16

TEST(GemmI8I8Rvv, PackedNonzeroKOrigin)
{
    const auto i32_funcs = make_packed_clamp_i32_i8p_i8p_4x1vl_rvv();
    const size_t m = i32_funcs.get_m_step() + 1;
    const size_t n = i32_funcs.get_n_step() + 3;
    constexpr size_t k = 67;
    constexpr size_t split = 32;
    run_packed_clamp_i32_i8_split_k_test(m, n, k, split, i32_funcs);

    run_packed_requantized_i8_nonzero_origin_test(
        m, n, k, split, make_packed_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv_funcs(),
        make_b_only_packed_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv_funcs(),
        make_packed_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv_funcs());
    run_packed_clamp_fp_i8_nonzero_origin_test(
        m, n, k, split, make_packed_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv_funcs());
#if TORQ_TILE_HAS_RVV_F16
    run_packed_clamp_fp_i8_nonzero_origin_test(
        m, n, k, split, make_packed_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv_funcs());
#endif
}

static void run_qsi8d256_qsi8c16_test(size_t m, size_t n)
{
    auto data = make_qsi8_block_quantized_gemm_test_data(n, 512, 256, 16, false, 31, m);
    std::vector<int8_t> a_packed(data.m * data.k), b_packed(data.n * data.k);
    std::vector<float> output(data.m * data.n);
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv(
        data.m, data.k, data.k, data.k, 0, data.a.data(), a_packed.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv(
        data.n, data.k, data.k, data.k, 0, data.b.data(), b_packed.data());
    const tqt_qsi8d256_qsi8c16_params params = {data.scale_a.data(), data.k / 256,
                                                data.scale_b.data(), data.k / 16};
    clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv, float, 13, 11, 12,
                   14, ClampSource::kColumnBias, false, 5, true>()(
        data.m, data.n, data.k, a_packed.data(), data.k, 0, b_packed.data(), data.k, 0,
        data.c.data(), data.n, output.data(), data.n, data.bias.data(),
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), &params);
    check_qsi8_block_quantized_gemm_result(output, data.reference);

    std::vector<float> partial(data.m * data.n);
    const size_t split = 256;
    clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv, float, 13, 11, 12,
                   14, ClampSource::kColumnBias, false, 5, true>()(
        data.m, data.n, split, a_packed.data(), data.k, 0, b_packed.data(), data.k, 0,
        data.c.data(), data.n, partial.data(), data.n, nullptr,
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), &params);
    const size_t a_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv(0, split,
                                                                                    data.k, data.m);
    const size_t b_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv(0, split,
                                                                                    data.k, data.n);
    clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv, float, 13, 11, 12,
                   14, ClampSource::kColumnBias, false, 5, true>()(
        data.m, data.n, data.k - split, (uint8_t *)a_packed.data() + a_offset, data.k, split,
        (uint8_t *)b_packed.data() + b_offset, data.k, split, partial.data(), data.n, output.data(),
        data.n, data.bias.data(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), &params);
    check_qsi8_block_quantized_gemm_result(output, data.reference);
}

TEST(GemmI8I8Rvv, Qsi8d256Qsi8c16)
{
    const size_t n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv();
    for (size_t rows = 1; rows <= 5; ++rows) {
        for (size_t columns : {size_t{1}, n_step, n_step + 3}) {
            run_qsi8d256_qsi8c16_test(rows, columns);
        }
    }
}

TEST(GemmI8I8Rvv, ClampSpecialValues_Qsi8d256Qsi8c16)
{
    ScopedGemmClampProbe probe;
    const size_t n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv();
    for (size_t rows = 1; rows <= 5; ++rows) {
        for (size_t columns : {size_t{1}, n_step, n_step + 3}) {
            run_qsi8d256_qsi8c16_test(rows, columns);
        }
    }
}

TEST(GemmI8I8Rvv, Qsi8d32Qsi8c32AndSplitK)
{
    const size_t n = tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv() + 3;
    auto data = make_qsi8_block_quantized_gemm_test_data(n, 64, 32, 32, false, 63);
    std::vector<int8_t> a_packed(data.m * data.k), b_packed(data.n * data.k);
    std::vector<float> output(data.m * data.n), partial(data.m * data.n);
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv(
        data.m, data.k, data.k, data.k, 0, data.a.data(), a_packed.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv(
        data.n, data.k, data.k, data.k, 0, data.b.data(), b_packed.data());
    const tqt_qsi8d32_qsi8c32_params params = {data.scale_a.data(), data.k / 32,
                                               data.scale_b.data(), data.k / 32};
    clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv, float, 13, 11, 12,
                   14, ClampSource::kColumnBias, false, 5, true>()(
        data.m, data.n, data.k, a_packed.data(), data.k, 0, b_packed.data(), data.k, 0,
        data.c.data(), data.n, output.data(), data.n, data.bias.data(),
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), &params);
    check_qsi8_block_quantized_gemm_result(output, data.reference);

    const size_t split = 32;
    clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv, float, 13, 11, 12,
                   14, ClampSource::kColumnBias, false, 5, true>()(
        data.m, data.n, split, a_packed.data(), data.k, 0, b_packed.data(), data.k, 0,
        data.c.data(), data.n, partial.data(), data.n, nullptr,
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), &params);
    const size_t a_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv(0, split, data.k,
                                                                                   data.m);
    const size_t b_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv(0, split, data.k,
                                                                                   data.n);
    clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv, float, 13, 11, 12,
                   14, ClampSource::kColumnBias, false, 5, true>()(
        data.m, data.n, data.k - split, (uint8_t *)a_packed.data() + a_offset, data.k, split,
        (uint8_t *)b_packed.data() + b_offset, data.k, split, partial.data(), data.n, output.data(),
        data.n, data.bias.data(), -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), &params);
    check_qsi8_block_quantized_gemm_result(output, data.reference);
}

static tqt_qsi8d32_qsi8c32_params make_qsi8d32_qsi8c32_qx0_rvv_params(const float *scale_a,
                                                                      size_t ldsa,
                                                                      const float *scale_b,
                                                                      size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

TEST(GemmI8I8Rvv, Qx0I8ExpandedSourceAware)
{
    for (const auto encoding : {Qx0SourceEncoding::Bipolar1, Qx0SourceEncoding::U2Z1,
                                Qx0SourceEncoding::I4, Qx0SourceEncoding::I5}) {
        const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qsi8c32_params> funcs = {
            encoding,
            8,
            32,
            tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            make_qsi8d32_qsi8c32_qx0_rvv_params,
            clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv, float, 13,
                           11, 12, 14, ClampSource::kColumnBias, false, 5, true>()};
        run_packed_qsi8_qx0_source_test(5, funcs.get_n_step() + 3,
                                        2 * qx0_source_block_size(encoding), funcs);
    }
}

TEST(GemmI8I8Rvv, ClampSpecialValues_Qx0I8ExpandedSourceAware)
{
    ScopedGemmClampProbe probe;
    for (const auto encoding : {Qx0SourceEncoding::Bipolar1, Qx0SourceEncoding::U2Z1,
                                Qx0SourceEncoding::I4, Qx0SourceEncoding::I5}) {
        const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qsi8c32_params> funcs = {
            encoding,
            8,
            32,
            tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv,
            make_qsi8d32_qsi8c32_qx0_rvv_params,
            clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv, float, 13,
                           11, 12, 14, ClampSource::kColumnBias, false, 5, true>()};
        for (size_t rows = 1;
             rows <= tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv() + 1;
             ++rows) {
            for (size_t columns :
                 {size_t{1}, tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv(),
                  tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi8c32p_4x1vl_rvv() + 3}) {
                run_packed_qsi8_qx0_source_test(rows, columns, 2 * qx0_source_block_size(encoding),
                                                funcs);
            }
        }
    }
}

TEST(RequantizedI8Contract, gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv)
{
    check_requantized_i8_contract<false>(
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(),
        [](size_t m, size_t n, size_t k, const int8_t *a, size_t lda, const int8_t *b, size_t ldb,
           const int32_t *c, size_t ldc, const int32_t *bias, int8_t *d, size_t ldd, int8_t lo,
           int8_t hi, const auto *params) {
            std::vector<int8_t> bp(n * k + 16);
            tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
                n, k, ldb, k, 0, b, bp.data());
            tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
                m, n, k, a, lda, 0, bp.data(), k, 0, c, ldc, d, ldd, bias, lo, hi, params);
        });
}

TEST(RequantizedI8Contract, gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv)
{
    check_requantized_i8_contract<false>(
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(),
        [](size_t m, size_t n, size_t k, const int8_t *a, size_t lda, const int8_t *b, size_t ldb,
           const int32_t *c, size_t ldc, const int32_t *bias, int8_t *d, size_t ldd, int8_t lo,
           int8_t hi, const auto *params) {
            tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(
                m, n, k, a, lda, 0, b, ldb, 0, c, ldc, d, ldd, bias, lo, hi, params);
        });
}

TEST(RequantizedI8Contract, gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv)
{
    check_requantized_i8_contract<false>(
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(),
        [](size_t m, size_t n, size_t k, const int8_t *a, size_t lda, const int8_t *b, size_t ldb,
           const int32_t *c, size_t ldc, const int32_t *bias, int8_t *d, size_t ldd, int8_t lo,
           int8_t hi, const auto *params) {
            std::vector<int8_t> ap(m * k + 16);
            tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                m, k, lda, k, 0, a, ap.data());
            std::vector<int8_t> bp(n * k + 16);
            tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                n, k, ldb, k, 0, b, bp.data());
            tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                m, n, k, ap.data(), k, 0, bp.data(), k, 0, c, ldc, d, ldd, bias, lo, hi, params);
        });
}

TEST(RequantizedI8Contract, gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv)
{
    check_requantized_i8_contract<true>(
        tqt_get_m_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(),
        tqt_get_n_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(),
        [](size_t m, size_t n, size_t k, const int8_t *a, size_t lda, const int8_t *b, size_t ldb,
           const int32_t *c, size_t ldc, const int32_t *bias, int8_t *d, size_t ldd, int8_t lo,
           int8_t hi, const auto *params) {
            tqt_run_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(
                m, n, k, a, lda, 0, b, ldb, 0, c, ldc, d, ldd, bias, lo, hi, params);
        });
}

TEST(RequantizedI8Contract, gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv)
{
    check_requantized_i8_contract<true>(
        tqt_get_m_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(),
        tqt_get_n_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(),
        [](size_t m, size_t n, size_t k, const int8_t *a, size_t lda, const int8_t *b, size_t ldb,
           const int32_t *c, size_t ldc, const int32_t *bias, int8_t *d, size_t ldd, int8_t lo,
           int8_t hi, const auto *params) {
            std::vector<int8_t> ap(m * k + 16);
            tqt_run_a_pack_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                m, k, lda, k, 0, a, ap.data());
            std::vector<int8_t> bp(n * k + 16);
            tqt_run_b_pack_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                n, k, ldb, k, 0, b, bp.data());
            tqt_run_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                m, n, k, ap.data(), k, 0, bp.data(), k, 0, c, ldc, d, ldd, bias, lo, hi, params);
        });
}

#if TORQ_TILE_HAS_RVV_F16

TEST(F32BiasContract, gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv)
{
    const size_t ms = tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv();
    const size_t m = ms + 1, n = ns + 1, full_m = ms + m, full_n = ns + n;
    const size_t full_k = 96, ki = 32, k = 32, ld = 128, bl = 16, lds = 8;
    std::vector<int8_t> a(full_m * ld, static_cast<int8_t>(0));
    std::vector<int8_t> b(full_n * ld, static_cast<int8_t>(0));
    for (size_t i = ms; i < full_m; ++i) {
        for (size_t p = ki; p < ki + k; ++p)
            a[i * ld + p] = 81;
    }
    for (size_t j = ns; j < full_n; ++j) {
        for (size_t p = ki; p < ki + k; ++p)
            b[j * ld + p] = 32;
    }
    const void *aw =
        reinterpret_cast<const uint8_t *>(a.data()) +
        tqt_get_a_offset_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(ms, ki, ld);
    const void *bw =
        reinterpret_cast<const uint8_t *>(b.data()) +
        tqt_get_b_offset_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(ns, ki, ld);
    std::vector<float> sa(full_m * lds, 1.0f), sb(full_n * lds, 1.0f);
    std::vector<int32_t> za(full_m, 17);
    tqt_qai8dxp_qsi8cxp_params params{sa.data() + ms, za.data() + ms, sb.data() + ns};
    EXPECT_EQ(tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(ns),
              ns * sizeof(float));
    check_f32_bias_epilogue<float16_t>(
        m, n, 65536.0f, false,
        [&](const float16_t *c, size_t stride, const float *bias, float16_t *d, float16_t lo,
            float16_t hi) {
            std::vector<float> full_bias(full_n, 999.0f);
            if (bias)
                std::copy(bias, bias + n, full_bias.begin() + ns);
            const float *bw_bias =
                bias ? full_bias.data() +
                           tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(
                               ns) /
                               sizeof(float)
                     : nullptr;
            tqt_run_gemm_1xnbiasf32_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(
                m, n, k, aw, ld, ki, bw, ld, ki, c, stride, d, stride, bw_bias, lo, hi, &params);
        });
}
#endif

#if TORQ_TILE_HAS_RVV_F16

TEST(F32BiasContract, gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv)
{
    const size_t ms = tqt_get_m_step_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv();
    const size_t m = ms + 1, n = ns + 1, full_m = ms + m, full_n = ns + n;
    const size_t full_k = 96, ki = 32, k = 32, ld = 128, bl = 16, lds = 8;
    std::vector<int8_t> a(full_m * ld, static_cast<int8_t>(0));
    std::vector<int8_t> b(full_n * ld, static_cast<int8_t>(0));
    for (size_t i = ms; i < full_m; ++i) {
        for (size_t p = ki; p < ki + k; ++p)
            a[i * ld + p] = 81;
    }
    for (size_t j = ns; j < full_n; ++j) {
        for (size_t p = ki; p < ki + k; ++p)
            b[j * ld + p] = 32;
    }
    std::vector<uint8_t> ap(
        tqt_get_a_packed_size_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(full_m, full_k));
    std::vector<uint8_t> bp(
        tqt_get_b_packed_size_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(full_n, full_k));
    tqt_run_a_pack_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(full_m, full_k, ld, full_k,
                                                                       0, a.data(), ap.data());
    tqt_run_bt_pack_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(full_n, full_k, ld, full_k,
                                                                        0, b.data(), bp.data());
    const void *aw =
        ap.data() + tqt_get_a_packed_offset_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(
                        ms, ki, full_k, m);
    const void *bw =
        bp.data() + tqt_get_b_packed_offset_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(
                        ns, ki, full_k, n);
    std::vector<float> sa(full_m * lds, 1.0f), sb(full_n * lds, 1.0f);
    std::vector<int32_t> za(full_m, 17);
    tqt_qai8dxp_qsi8cxp_params params{sa.data() + ms, za.data() + ms, sb.data() + ns};
    EXPECT_EQ(tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(ns),
              ns * sizeof(float));
    check_f32_bias_epilogue<float16_t>(
        m, n, 65536.0f, false,
        [&](const float16_t *c, size_t stride, const float *bias, float16_t *d, float16_t lo,
            float16_t hi) {
            std::vector<float> full_bias(full_n, 999.0f);
            if (bias)
                std::copy(bias, bias + n, full_bias.begin() + ns);
            const float *bw_bias =
                bias ? full_bias.data() +
                           tqt_get_bias_offset_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(
                               ns) /
                               sizeof(float)
                     : nullptr;
            tqt_run_gemm_1xnbiasf32_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(
                m, n, k, aw, full_k, ki, bw, full_k, ki, c, stride, d, stride, bw_bias, lo, hi,
                &params);
        });
}
#endif

TEST(F32BiasContract, gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv)
{
    const size_t ms = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv();
    const size_t m = ms + 1, n = ns + 1, full_m = ms + m, full_n = ns + n;
    const size_t full_k = 96, ki = 32, k = 32, ld = 128, bl = 16, lds = 8;
    std::vector<int8_t> a(full_m * ld, static_cast<int8_t>(0));
    std::vector<int8_t> b(full_n * ld, static_cast<int8_t>(0));
    for (size_t i = ms; i < full_m; ++i) {
        for (size_t p = ki; p < ki + k; ++p)
            a[i * ld + p] = 81;
    }
    for (size_t j = ns; j < full_n; ++j) {
        for (size_t p = ki; p < ki + k; ++p)
            b[j * ld + p] = 32;
    }
    const void *aw = reinterpret_cast<const uint8_t *>(a.data()) +
                     tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(ms, ki, ld);
    const void *bw = reinterpret_cast<const uint8_t *>(b.data()) +
                     tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(ns, ki, ld);
    std::vector<float> sa(full_m * lds, 1.0f), sb(full_n * lds, 1.0f);
    std::vector<int32_t> za(full_m, 17);
    tqt_qai8dxp_qsi8cxp_params params{sa.data() + ms, za.data() + ms, sb.data() + ns};
    EXPECT_EQ(tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(ns),
              ns * sizeof(float));
    check_f32_bias_epilogue<float>(
        m, n, 65536.0f, false,
        [&](const float *c, size_t stride, const float *bias, float *d, float lo, float hi) {
            std::vector<float> full_bias(full_n, 999.0f);
            if (bias)
                std::copy(bias, bias + n, full_bias.begin() + ns);
            const float *bw_bias =
                bias ? full_bias.data() +
                           tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(ns) /
                               sizeof(float)
                     : nullptr;
            tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(
                m, n, k, aw, ld, ki, bw, ld, ki, c, stride, d, stride, bw_bias, lo, hi, &params);
        });
}

TEST(F32BiasContract, gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv)
{
    const size_t ms = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv();
    const size_t m = ms + 1, n = ns + 1, full_m = ms + m, full_n = ns + n;
    const size_t full_k = 96, ki = 32, k = 32, ld = 128, bl = 16, lds = 8;
    std::vector<int8_t> a(full_m * ld, static_cast<int8_t>(0));
    std::vector<int8_t> b(full_n * ld, static_cast<int8_t>(0));
    for (size_t i = ms; i < full_m; ++i) {
        for (size_t p = ki; p < ki + k; ++p)
            a[i * ld + p] = 81;
    }
    for (size_t j = ns; j < full_n; ++j) {
        for (size_t p = ki; p < ki + k; ++p)
            b[j * ld + p] = 32;
    }
    std::vector<uint8_t> ap(
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(full_m, full_k));
    std::vector<uint8_t> bp(
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(full_n, full_k));
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(full_m, full_k, ld, full_k, 0,
                                                                    a.data(), ap.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(full_n, full_k, ld, full_k, 0,
                                                                     b.data(), bp.data());
    const void *aw =
        ap.data() +
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(ms, ki, full_k, m);
    const void *bw =
        bp.data() +
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(ns, ki, full_k, n);
    std::vector<float> sa(full_m * lds, 1.0f), sb(full_n * lds, 1.0f);
    std::vector<int32_t> za(full_m, 17);
    tqt_qai8dxp_qsi8cxp_params params{sa.data() + ms, za.data() + ms, sb.data() + ns};
    EXPECT_EQ(tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(ns),
              ns * sizeof(float));
    check_f32_bias_epilogue<float>(
        m, n, 65536.0f, false,
        [&](const float *c, size_t stride, const float *bias, float *d, float lo, float hi) {
            std::vector<float> full_bias(full_n, 999.0f);
            if (bias)
                std::copy(bias, bias + n, full_bias.begin() + ns);
            const float *bw_bias =
                bias
                    ? full_bias.data() +
                          tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(ns) /
                              sizeof(float)
                    : nullptr;
            tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(
                m, n, k, aw, full_k, ki, bw, full_k, ki, c, stride, d, stride, bw_bias, lo, hi,
                &params);
        });
}

TEST(RequantizedI8Window, gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv)
{
    check_requantized_i8_contract<false>(
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(),
        [](size_t m, size_t n, size_t k, const int8_t *a, size_t lda, const int8_t *b, size_t ldb,
           const int32_t *c, size_t ldc, const int32_t *bias, int8_t *d, size_t ldd, int8_t lo,
           int8_t hi, const auto *params) {
            const size_t origin = 4;
            const size_t full_k = origin + k;
            std::vector<int8_t> af(m * full_k, -117), bf(n * full_k, 119);
            for (size_t i = 0; i < m; ++i)
                for (size_t q = 0; q < k; ++q)
                    af[i * full_k + origin + q] = a[i * lda + q];
            for (size_t j = 0; j < n; ++j)
                for (size_t q = 0; q < k; ++q)
                    bf[j * full_k + origin + q] = b[j * ldb + q];
            std::vector<int8_t> ap(
                tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                    m, full_k));
            std::vector<int8_t> bp(
                tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                    n, full_k));
            tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                m, full_k, full_k, full_k, 0, af.data(), ap.data());
            tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                n, full_k, full_k, full_k, 0, bf.data(), bp.data());
            const int8_t *aw =
                ap.data() +
                tqt_get_a_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                    0, origin, full_k, m);
            const int8_t *bw =
                bp.data() +
                tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                    0, origin, full_k, n);
            tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                m, n, k, aw, full_k, origin, bw, full_k, origin, c, ldc, d, ldd, bias, lo, hi,
                params);
        });
}

TEST(RequantizedI8Window, gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv)
{
    check_requantized_i8_contract<true>(
        tqt_get_m_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(),
        tqt_get_n_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(),
        [](size_t m, size_t n, size_t k, const int8_t *a, size_t lda, const int8_t *b, size_t ldb,
           const int32_t *c, size_t ldc, const int32_t *bias, int8_t *d, size_t ldd, int8_t lo,
           int8_t hi, const auto *params) {
            const size_t origin = 4;
            const size_t full_k = origin + k;
            std::vector<int8_t> af(m * full_k, -117), bf(n * full_k, 119);
            for (size_t i = 0; i < m; ++i)
                for (size_t q = 0; q < k; ++q)
                    af[i * full_k + origin + q] = a[i * lda + q];
            for (size_t j = 0; j < n; ++j)
                for (size_t q = 0; q < k; ++q)
                    bf[(origin + q) * n + j] = b[q * ldb + j];
            std::vector<int8_t> ap(
                tqt_get_a_packed_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                    m, full_k));
            std::vector<int8_t> bp(
                tqt_get_b_packed_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                    n, full_k));
            tqt_run_a_pack_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                m, full_k, full_k, full_k, 0, af.data(), ap.data());
            tqt_run_b_pack_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                n, full_k, n, full_k, 0, bf.data(), bp.data());
            const int8_t *aw =
                ap.data() +
                tqt_get_a_packed_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                    0, origin, full_k, m);
            const int8_t *bw =
                bp.data() +
                tqt_get_b_packed_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                    0, origin, full_k, n);
            tqt_run_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                m, n, k, aw, full_k, origin, bw, full_k, origin, c, ldc, d, ldd, bias, lo, hi,
                params);
        });
}

#if TORQ_TILE_HAS_RVV_F16
// Original narrow-bias entry points remain covered after adding biasf32 variants.
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv.h"

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
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, false>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv_funcs();
    run_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t, float16_t>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv_funcs();
    run_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t, float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    with_clamp_dimensions(int8_common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

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
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, false>(),
    };
}

class Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_packed_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv_funcs();
    run_packed_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t, float16_t>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_packed_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv_funcs();
    run_packed_1xnbias_clamp_fp_qai8dxp_qsi8cxp_gemm_test<float16_t, float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
    with_clamp_dimensions(int8_common_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());
#endif  // TORQ_TILE_HAS_RVV_F16
