//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"

// --- W4A8 per-channel (qai8dx × qsi4cxt): F32 always; F16/BF16 gated below ---
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv.h"

// --- W4A8 per-channel requantized int8 output (qai8dx × qsi4cxt -> qai8) ---
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv.h"

// --- W4A8 block-quantized: Class 1 qai8dx × qsi4c2 (F32 always) ---
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv.h"
// --- W4A8 block-quantized: Class 2 qsi8d2 × qsi4c2 (F32 always) ---
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv.h"
// --- W4A8 block-quantized: Class 3 qsi8d2 × qai4c2 (F32 always) ---
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t/tqt_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p/tqt_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv.h"

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
// W4A8 per-channel F16
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv.h"
// W4A8 block-quantized F16
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dxp_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2_qai4c2t/tqt_gemm_1xnbias_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_qsi8d2_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2p_qai4c2p/tqt_gemm_1xnbias_clamp_f16_qsi8d2p_qai4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2p_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv.h"
#endif  // __riscv_zfh && __riscv_zvfh

#if defined(__riscv_zvfbfwma)
// W4A8 per-channel BF16
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv.h"
// W4A8 block-quantized BF16
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dx_qsi4c2t/tqt_gemm_1xnbias_clamp_bf16_qai8dx_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dxp_qsi4c2p/tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2_qai4c2t/tqt_gemm_1xnbias_clamp_bf16_qsi8d2_qai4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t/tqt_gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2p_qai4c2p/tqt_gemm_1xnbias_clamp_bf16_qsi8d2p_qai4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p/tqt_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv.h"
#endif  // __riscv_zvfbfwma

#include "test/common/gemm_test_runner.h"
#include "test/common/quantized_gemm_test_runner.h"
#include "test/common/test_common.h"

using namespace torq_tile::test;

// ============================================================================
// Common test values
// ============================================================================

static auto common_w4a8_test_values()
{
    return testing::Values(
        GemmTestParams(8, 8, 8, "Basic"), GemmTestParams(16, 16, 16, "Small"),
        GemmTestParams(32, 32, 32, "Medium"), GemmTestParams(5, 11, 16, "NonAligned"),
        GemmTestParams(7, 3, 8, "SmallNonAligned"), GemmTestParams(17, 19, 24, "Odd"),
        GemmTestParams(31, 35, 64, "Large"), GemmTestParams(60, 61, 128, "LargerK"),
        GemmTestParams(32, 32, 32, "WithBias").set_bias(true),
        GemmTestParams(17, 19, 24, "WithBias_Odd").set_bias(true),
        GemmTestParams(32, 32, 32, "WithClamp").set_clamp(-0.5f, 0.5f),
        GemmTestParams(32, 32, 32, "WithC").set_c(true),
        GemmTestParams(32, 32, 32, "WithC_Bias_Clamp")
            .set_c(true)
            .set_bias(true)
            .set_clamp(-0.5f, 0.5f));
}

static auto common_w4a8_bq_test_values()
{
    return testing::Values(
        // BL=2 (minimum block size)
        GemmTestParams(8, 8, 8, "Basic_BL2").set_bl(2),
        GemmTestParams(16, 16, 16, "Small_BL2").set_bl(2),
        GemmTestParams(5, 11, 16, "NonAligned_BL2").set_bl(2),
        // BL=4
        GemmTestParams(32, 32, 32, "Medium_BL4").set_bl(4),
        GemmTestParams(17, 19, 24, "Odd_BL4").set_bl(4),
        GemmTestParams(7, 3, 8, "SmallNonAligned_BL4").set_bl(4),
        // BL=8
        GemmTestParams(32, 32, 32, "Medium_BL8").set_bl(8),
        // BL=16
        GemmTestParams(31, 35, 64, "Large_BL16").set_bl(16),
        // BL=32 (typical LLM group_size)
        GemmTestParams(60, 61, 128, "LargerK_BL32").set_bl(32),
        // With bias
        GemmTestParams(32, 32, 32, "WithBias_BL4").set_bl(4).set_bias(true),
        // With clamp
        GemmTestParams(32, 32, 32, "WithClamp_BL4").set_bl(4).set_clamp(-0.5f, 0.5f),
        // With C accumulation
        GemmTestParams(32, 32, 32, "WithC_BL4").set_bl(4).set_c(true),
        // Combined
        GemmTestParams(32, 32, 32, "WithC_Bias_Clamp_BL4")
            .set_bl(4)
            .set_c(true)
            .set_bias(true)
            .set_clamp(-0.5f, 0.5f));
}

// ============================================================================
// W4A8 per-channel (qai8dx × qsi4cxt)
// ============================================================================

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
// ---------- gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv (a_bt) ----------
static W4A8GemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8GemmFuncs<float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

// ---------- gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv (ap_bp) ----------
static W4A8PackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8PackedGemmFuncs<float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());
#endif  // __riscv_zfh && __riscv_zvfh

// ---------- gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv (a_bt) ----------
static W4A8GemmFuncs<float> make_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8GemmFuncs<float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

// ---------- gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv (ap_bp) ----------
static W4A8PackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8PackedGemmFuncs<float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// W4A8 per-channel requantized int8 output (qai8dx × qsi4cxt -> qai8)
// ============================================================================

// ---------- gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv (a_bt) ----------
static W4A8ReqI8GemmFuncs make_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8ReqI8GemmFuncs funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_reqi8_gemm_test(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests,
                         Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

// ---------- gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv (ap_bp) ----------
static W4A8ReqI8PackedGemmFuncs
make_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8ReqI8PackedGemmFuncs funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack =
        tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_reqi8_packed_gemm_test(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests,
                         Test_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

#if defined(__riscv_zvfbfwma)
// ---------- gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv (a_bt) ----------
static W4A8GemmFuncs<bfloat16_t> make_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8GemmFuncs<bfloat16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<bfloat16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());

// ---------- gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv (ap_bp) ----------
static W4A8PackedGemmFuncs<bfloat16_t>
make_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8PackedGemmFuncs<bfloat16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<bfloat16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
                         common_w4a8_test_values(), GemmTestParamNameGenerator());
#endif  // __riscv_zvfbfwma

static tqt_qsi8d32_qsi4c32_params make_qsi8d32_qsi4c32_rvv_params(const float *scale_a, size_t ldsa,
                                                                  const float *scale_b, size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

TEST(GemmI8I4Rvv, Q4_0SourceAwareTailsAndSplitK)
{
    const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qsi4c32_params> funcs = {
        Qx0SourceEncoding::I4,
        4,
        32,
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        make_qsi8d32_qsi4c32_rvv_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv};
    run_packed_qsi8_qx0_source_test(5, funcs.get_n_step() + 3, 64, funcs);
}

// ============================================================================
// W4A8 block-quantized — Class 1: qai8dx × qsi4c2 (F32 always; F16/BF16 gated)
// ============================================================================

W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f32_qai8dx_qsi4c2t, float,
                             run_w4a8_bq_qai8dx_qsi4c2t_test, W4A8BQ_Qai8dx_Qsi4c2_Funcs);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f32_qai8dxp_qsi4c2p, float,
                           run_w4a8_bq_qai8dxp_qsi4c2p_test, W4A8BQ_Qai8dx_Qsi4c2_PackedFuncs);

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f16_qai8dx_qsi4c2t, float16_t,
                             run_w4a8_bq_qai8dx_qsi4c2t_test, W4A8BQ_Qai8dx_Qsi4c2_Funcs);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f16_qai8dxp_qsi4c2p, float16_t,
                           run_w4a8_bq_qai8dxp_qsi4c2p_test, W4A8BQ_Qai8dx_Qsi4c2_PackedFuncs);
#endif  // __riscv_zfh && __riscv_zvfh

#if defined(__riscv_zvfbfwma)
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_bf16_qai8dx_qsi4c2t, bfloat16_t,
                             run_w4a8_bq_qai8dx_qsi4c2t_test, W4A8BQ_Qai8dx_Qsi4c2_Funcs);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_bf16_qai8dxp_qsi4c2p, bfloat16_t,
                           run_w4a8_bq_qai8dxp_qsi4c2p_test, W4A8BQ_Qai8dx_Qsi4c2_PackedFuncs);
#endif  // __riscv_zvfbfwma

// ============================================================================
// W4A8 block-quantized — Class 2: qsi8d2 × qsi4c2 (F32 always; F16/BF16 gated)
// ============================================================================

W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f32_qsi8d2_qsi4c2t, float,
                             run_w4a8_bq_qsi8d2_qsi4c2t_test, W4A8BQ_Qsi8d2_Qsi4c2_Funcs);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f32_qsi8d2p_qsi4c2p, float,
                           run_w4a8_bq_qsi8d2p_qsi4c2p_test, W4A8BQ_Qsi8d2_Qsi4c2_PackedFuncs);

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f16_qsi8d2_qsi4c2t, float16_t,
                             run_w4a8_bq_qsi8d2_qsi4c2t_test, W4A8BQ_Qsi8d2_Qsi4c2_Funcs);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f16_qsi8d2p_qsi4c2p, float16_t,
                           run_w4a8_bq_qsi8d2p_qsi4c2p_test, W4A8BQ_Qsi8d2_Qsi4c2_PackedFuncs);
#endif  // __riscv_zfh && __riscv_zvfh

#if defined(__riscv_zvfbfwma)
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_bf16_qsi8d2_qsi4c2t, bfloat16_t,
                             run_w4a8_bq_qsi8d2_qsi4c2t_test, W4A8BQ_Qsi8d2_Qsi4c2_Funcs);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_bf16_qsi8d2p_qsi4c2p, bfloat16_t,
                           run_w4a8_bq_qsi8d2p_qsi4c2p_test, W4A8BQ_Qsi8d2_Qsi4c2_PackedFuncs);
#endif  // __riscv_zvfbfwma

// ============================================================================
// W4A8 block-quantized — Class 3: qsi8d2 × qai4c2 (F32 always; F16/BF16 gated)
// ============================================================================

W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f32_qsi8d2_qai4c2t, float,
                             run_w4a8_bq_qsi8d2_qai4c2t_test, W4A8BQ_Qsi8d2_Qai4c2_Funcs);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f32_qsi8d2p_qai4c2p, float,
                           run_w4a8_bq_qsi8d2p_qai4c2p_test, W4A8BQ_Qsi8d2_Qai4c2_PackedFuncs);

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f16_qsi8d2_qai4c2t, float16_t,
                             run_w4a8_bq_qsi8d2_qai4c2t_test, W4A8BQ_Qsi8d2_Qai4c2_Funcs);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f16_qsi8d2p_qai4c2p, float16_t,
                           run_w4a8_bq_qsi8d2p_qai4c2p_test, W4A8BQ_Qsi8d2_Qai4c2_PackedFuncs);
#endif  // __riscv_zfh && __riscv_zvfh

#if defined(__riscv_zvfbfwma)
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_bf16_qsi8d2_qai4c2t, bfloat16_t,
                             run_w4a8_bq_qsi8d2_qai4c2t_test, W4A8BQ_Qsi8d2_Qai4c2_Funcs);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_bf16_qsi8d2p_qai4c2p, bfloat16_t,
                           run_w4a8_bq_qsi8d2p_qai4c2p_test, W4A8BQ_Qsi8d2_Qai4c2_PackedFuncs);
#endif  // __riscv_zvfbfwma
