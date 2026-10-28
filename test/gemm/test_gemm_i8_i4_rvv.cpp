//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "torq_tile/tqt_compile_config.h"

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

#if TORQ_TILE_HAS_RVV_F16
// W4A8 per-channel F16
// W4A8 block-quantized F16
#endif  // TORQ_TILE_HAS_RVV_F16

#if TORQ_TILE_HAS_RVV_BF16
// W4A8 per-channel BF16
// W4A8 block-quantized BF16
#endif  // TORQ_TILE_HAS_RVV_BF16

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

#if TORQ_TILE_HAS_RVV_F16

#endif  // TORQ_TILE_HAS_RVV_F16

// ---------- gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv (a_bt) ----------
static W4A8GemmFuncs<float> make_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8GemmFuncs<float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm = clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv, float,
                                    13, 11, 12, 14, ClampSource::kColumnBias, false, 5, true>();
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

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
    with_clamp_dimensions(common_w4a8_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());

// ---------- gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv (ap_bp) ----------
static W4A8PackedGemmFuncs<float> make_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8PackedGemmFuncs<float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm = clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv, float,
                                    13, 11, 12, 14, ClampSource::kColumnBias, false, 5, true>();
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

TEST_P(Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<float>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv,
    with_clamp_dimensions(common_w4a8_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());

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
    funcs.get_a_packed_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
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

#if TORQ_TILE_HAS_RVV_BF16

#endif  // TORQ_TILE_HAS_RVV_BF16

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
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>()};
    run_packed_qsi8_qx0_source_test(5, funcs.get_n_step() + 3, 64, funcs);
}

TEST(GemmI8I4Rvv, ClampSpecialValues_Q4_0SourceAwareTailsAndSplitK)
{
    ScopedGemmClampProbe probe;
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
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>()};
    for (size_t rows = 1;
         rows <= tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv() + 1; ++rows) {
        for (size_t columns :
             {size_t{1}, tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv(),
              tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv() + 3}) {
            run_packed_qsi8_qx0_source_test(rows, columns, 64, funcs);
        }
    }
}

// ============================================================================
// W4A8 block-quantized — Class 1: qai8dx × qsi4c2 (F32 always; F16/BF16 gated)
// ============================================================================

W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f32_qai8dx_qsi4c2t, float,
                             run_w4a8_bq_qai8dx_qsi4c2t_test, W4A8BQ_Qai8dx_Qsi4c2_Funcs, float);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f32_qai8dxp_qsi4c2p, float,
                           run_w4a8_bq_qai8dxp_qsi4c2p_test, W4A8BQ_Qai8dx_Qsi4c2_PackedFuncs,
                           float);

#if TORQ_TILE_HAS_RVV_F16

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv.h"
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f16_qai8dx_qsi4c2t, float16_t,
                             run_w4a8_bq_qai8dx_qsi4c2t_test, W4A8BQ_Qai8dx_Qsi4c2_Funcs,
                             float16_t);

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dxp_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi4c2p_4x1vl_rvv.h"
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f16_qai8dxp_qsi4c2p, float16_t,
                           run_w4a8_bq_qai8dxp_qsi4c2p_test, W4A8BQ_Qai8dx_Qsi4c2_PackedFuncs,
                           float16_t);
#endif  // TORQ_TILE_HAS_RVV_F16

#if TORQ_TILE_HAS_RVV_BF16

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dx_qsi4c2t/tqt_gemm_1xnbias_clamp_bf16_qai8dx_qsi4c2t_4x1vl_rvv.h"
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_bf16_qai8dx_qsi4c2t, bfloat16_t,
                             run_w4a8_bq_qai8dx_qsi4c2t_test, W4A8BQ_Qai8dx_Qsi4c2_Funcs,
                             bfloat16_t);

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dxp_qsi4c2p/tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4c2p_4x1vl_rvv.h"
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_bf16_qai8dxp_qsi4c2p, bfloat16_t,
                           run_w4a8_bq_qai8dxp_qsi4c2p_test, W4A8BQ_Qai8dx_Qsi4c2_PackedFuncs,
                           bfloat16_t);
#endif  // TORQ_TILE_HAS_RVV_BF16

// ============================================================================
// W4A8 block-quantized — Class 2: qsi8d2 × qsi4c2 (F32 always; F16/BF16 gated)
// ============================================================================

W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f32_qsi8d2_qsi4c2t, float,
                             run_w4a8_bq_qsi8d2_qsi4c2t_test, W4A8BQ_Qsi8d2_Qsi4c2_Funcs, float);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f32_qsi8d2p_qsi4c2p, float,
                           run_w4a8_bq_qsi8d2p_qsi4c2p_test, W4A8BQ_Qsi8d2_Qsi4c2_PackedFuncs,
                           float);

#if TORQ_TILE_HAS_RVV_F16

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_qsi8d2_qsi4c2t_4x1vl_rvv.h"
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f16_qsi8d2_qsi4c2t, float16_t,
                             run_w4a8_bq_qsi8d2_qsi4c2t_test, W4A8BQ_Qsi8d2_Qsi4c2_Funcs,
                             float16_t);

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2p_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv.h"
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f16_qsi8d2p_qsi4c2p, float16_t,
                           run_w4a8_bq_qsi8d2p_qsi4c2p_test, W4A8BQ_Qsi8d2_Qsi4c2_PackedFuncs,
                           float16_t);
#endif  // TORQ_TILE_HAS_RVV_F16

#if TORQ_TILE_HAS_RVV_BF16

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t/tqt_gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t_4x1vl_rvv.h"
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_bf16_qsi8d2_qsi4c2t, bfloat16_t,
                             run_w4a8_bq_qsi8d2_qsi4c2t_test, W4A8BQ_Qsi8d2_Qsi4c2_Funcs,
                             bfloat16_t);

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p/tqt_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv.h"
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_bf16_qsi8d2p_qsi4c2p, bfloat16_t,
                           run_w4a8_bq_qsi8d2p_qsi4c2p_test, W4A8BQ_Qsi8d2_Qsi4c2_PackedFuncs,
                           bfloat16_t);
#endif  // TORQ_TILE_HAS_RVV_BF16

// ============================================================================
// W4A8 block-quantized — Class 3: qsi8d2 × qai4c2 (F32 always; F16/BF16 gated)
// ============================================================================

W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f32_qsi8d2_qai4c2t, float,
                             run_w4a8_bq_qsi8d2_qai4c2t_test, W4A8BQ_Qsi8d2_Qai4c2_Funcs, float);
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f32_qsi8d2p_qai4c2p, float,
                           run_w4a8_bq_qsi8d2p_qai4c2p_test, W4A8BQ_Qsi8d2_Qai4c2_PackedFuncs,
                           float);

#if TORQ_TILE_HAS_RVV_F16

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2_qai4c2t/tqt_gemm_1xnbias_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv.h"
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_f16_qsi8d2_qai4c2t, float16_t,
                             run_w4a8_bq_qsi8d2_qai4c2t_test, W4A8BQ_Qsi8d2_Qai4c2_Funcs,
                             float16_t);

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2p_qai4c2p/tqt_gemm_1xnbias_clamp_f16_qsi8d2p_qai4c2p_4x1vl_rvv.h"
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_f16_qsi8d2p_qai4c2p, float16_t,
                           run_w4a8_bq_qsi8d2p_qai4c2p_test, W4A8BQ_Qsi8d2_Qai4c2_PackedFuncs,
                           float16_t);
#endif  // TORQ_TILE_HAS_RVV_F16

#if TORQ_TILE_HAS_RVV_BF16

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2_qai4c2t/tqt_gemm_1xnbias_clamp_bf16_qsi8d2_qai4c2t_4x1vl_rvv.h"
W4A8_BQ_DEFINE_UNPACKED_TEST(1xnbias_clamp_bf16_qsi8d2_qai4c2t, bfloat16_t,
                             run_w4a8_bq_qsi8d2_qai4c2t_test, W4A8BQ_Qsi8d2_Qai4c2_Funcs,
                             bfloat16_t);

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2p_qai4c2p/tqt_gemm_1xnbias_clamp_bf16_qsi8d2p_qai4c2p_4x1vl_rvv.h"
W4A8_BQ_DEFINE_PACKED_TEST(1xnbias_clamp_bf16_qsi8d2p_qai4c2p, bfloat16_t,
                           run_w4a8_bq_qsi8d2p_qai4c2p_test, W4A8BQ_Qsi8d2_Qai4c2_PackedFuncs,
                           bfloat16_t);
#endif  // TORQ_TILE_HAS_RVV_BF16

TEST(RequantizedI4Contract, gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv)
{
    const size_t ms = tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv();
    const size_t m = 2 * ms + 1, n = 2 * ns + 1, k = 40;
    const size_t lda = k + 4, ldb = k + 8, ldd = n + 7;
    std::vector<int8_t> a(m * lda);
    std::vector<uint8_t> b(n * (ldb / 2));
    std::vector<int32_t> bias_i32(n), zp(m, 37);
    std::vector<float> scale_a(m), scale_b(n), bias(n), c(m * (n + 3));
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = (int8_t)((int)(i * 29 % 256) - 128);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = (uint8_t)(i * 83 + 29);
    for (size_t i = 0; i < m; ++i)
        scale_a[i] = i % 2 ? 2.0f : 0.25f;
    for (size_t i = 0; i < n; ++i) {
        bias_i32[i] = (int32_t)(i % 11) - 5;
        bias[i] = (float)((int)(i % 7) - 3) * 0.25f;
        scale_b[i] = i % 3 == 0 ? 0.5f : i % 3 == 1 ? 1.0f : 2.0f;
    }
    for (bool window : {false, true}) {
        const size_t mi = window ? ms : 0, ni = window ? ns : 0;
        const size_t ki = window ? 2 : 0, count = window ? 36 : k;
        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < n; ++col) {
                int64_t dot = 0;
                for (size_t j = ki; j < ki + count; ++j) {
                    const int value = ((b[col * (ldb / 2) + j / 2] >> (4 * (j % 2))) & 15) - 8;
                    dot += (int64_t)(a[row * lda + j] - 37) * value;
                }
                c[row * (n + 3) + col] = -(float)dot * scale_a[row] * scale_b[col] +
                                         (float)((int)((row + col) % 5) - 2) * 0.25f;
            }
        }
        for (int options = 0; options < 8; ++options) {
            for (unsigned long mode : {0UL, 1UL, 2UL, 3UL}) {
                std::vector<int8_t> d(m * ldd + 16, 93);
                tqt_qai8_qai8dx_qsi4cx_params params{};
                params.scale_a = scale_a.data() + mi;
                params.zero_point_a = zp.data() + mi;
                params.scale_b = scale_b.data() + ni;
                params.scale_d = 0.5f;
                params.zero_point_d = -7;
                params.bias_i32 = options & 1 ? bias_i32.data() + ni : nullptr;
                const auto *a_ptr =
                    a.data() +
                    tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv(
                        mi, ki, lda);
                const auto *b_ptr =
                    b.data() +
                    tqt_get_b_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv(
                        ni, ki, ldb);
                unsigned long saved;
                asm volatile("csrrw %0, frm, %1" : "=r"(saved) : "r"(mode) : "memory");
                tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv(
                    m - mi, n - ni, count, a_ptr, lda, ki, b_ptr, ldb, ki,
                    options & 4 ? c.data() + mi * (n + 3) + ni : nullptr, n + 3,
                    d.data() + mi * ldd + ni, ldd, options & 2 ? bias.data() + ni : nullptr, -53,
                    79, &params);
                unsigned long after;
                asm volatile("csrr %0, frm\n\tcsrw frm, %1" : "=&r"(after) : "r"(saved) : "memory");
                ASSERT_EQ(after, mode);
                for (size_t row = 0; row < m; ++row) {
                    for (size_t col = 0; col < ldd; ++col) {
                        if (row < mi || col < ni || col >= n) {
                            ASSERT_EQ(d[row * ldd + col], 93);
                            continue;
                        }
                        int64_t dot = options & 1 ? bias_i32[col] : 0;
                        for (size_t j = ki; j < ki + count; ++j) {
                            const int value =
                                ((b[col * (ldb / 2) + j / 2] >> (4 * (j % 2))) & 15) - 8;
                            dot += (int64_t)(a[row * lda + j] - 37) * value;
                        }
                        float real = (float)dot * scale_a[row] * scale_b[col];
                        if (options & 4)
                            real += c[row * (n + 3) + col];
                        if (options & 2)
                            real += bias[col];
                        // All operands above are exactly represented binary fractions.
                        ASSERT_EQ(d[row * ldd + col], torq_tile::test::reference_requantized_i8(
                                                          (int64_t)(real * 8), 0.25f, -7, -53, 79))
                            << "row=" << row << " col=" << col << " options=" << options
                            << " window=" << window << " mode=" << mode;
                    }
                }
                for (size_t i = m * ldd; i < d.size(); ++i)
                    ASSERT_EQ(d[i], 93);
            }
        }
    }
}

TEST(RequantizedI4Contract, gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv)
{
    const size_t ms = tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv();
    const size_t m = 2 * ms + 1, n = 2 * ns + 1, k = 40;
    const size_t lda = k + 4, ldb = k + 8, ldd = n + 7;
    std::vector<int8_t> a(m * lda);
    std::vector<uint8_t> b(n * (ldb / 2));
    std::vector<int32_t> bias_i32(n), zp(m, 37);
    std::vector<float> scale_a(m), scale_b(n), bias(n), c(m * (n + 3));
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = (int8_t)((int)(i * 29 % 256) - 128);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = (uint8_t)(i * 83 + 29);
    for (size_t i = 0; i < m; ++i)
        scale_a[i] = i % 2 ? 2.0f : 0.25f;
    for (size_t i = 0; i < n; ++i) {
        bias_i32[i] = (int32_t)(i % 11) - 5;
        bias[i] = (float)((int)(i % 7) - 3) * 0.25f;
        scale_b[i] = i % 3 == 0 ? 0.5f : i % 3 == 1 ? 1.0f : 2.0f;
    }
    for (bool window : {false, true}) {
        const size_t mi = window ? ms : 0, ni = window ? ns : 0;
        const size_t ki = window ? 2 : 0, count = window ? 36 : k;
        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < n; ++col) {
                int64_t dot = 0;
                for (size_t j = ki; j < ki + count; ++j) {
                    const int value = ((b[col * (ldb / 2) + j / 2] >> (4 * (j % 2))) & 15) - 8;
                    dot += (int64_t)(a[row * lda + j] - 37) * value;
                }
                c[row * (n + 3) + col] = -(float)dot * scale_a[row] * scale_b[col] +
                                         (float)((int)((row + col) % 5) - 2) * 0.25f;
            }
        }
        for (int options = 0; options < 8; ++options) {
            for (unsigned long mode : {0UL, 1UL, 2UL, 3UL}) {
                std::vector<int8_t> d(m * ldd + 16, 93);
                tqt_qai8_qai8dx_qsi4cx_params params{};
                params.scale_a = scale_a.data() + mi;
                params.zero_point_a = zp.data() + mi;
                params.scale_b = scale_b.data() + ni;
                params.scale_d = 0.5f;
                params.zero_point_d = -7;
                params.bias_i32 = options & 1 ? bias_i32.data() + ni : nullptr;
                std::vector<int8_t> ap(
                    tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                        m, k));
                std::vector<uint8_t> bp(
                    tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                        n, k));
                tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                    m, k, lda, k, 0, a.data(), ap.data());
                tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                    n, k, ldb, k, 0, b.data(), bp.data());
                const auto *a_ptr =
                    ap.data() +
                    tqt_get_a_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                        mi, ki, k, m - mi);
                const auto *b_ptr =
                    bp.data() +
                    tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                        ni, ki, k, n - ni);
                unsigned long saved;
                asm volatile("csrrw %0, frm, %1" : "=r"(saved) : "r"(mode) : "memory");
                tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                    m - mi, n - ni, count, a_ptr, k, ki, b_ptr, k, ki,
                    options & 4 ? c.data() + mi * (n + 3) + ni : nullptr, n + 3,
                    d.data() + mi * ldd + ni, ldd, options & 2 ? bias.data() + ni : nullptr, -53,
                    79, &params);
                unsigned long after;
                asm volatile("csrr %0, frm\n\tcsrw frm, %1" : "=&r"(after) : "r"(saved) : "memory");
                ASSERT_EQ(after, mode);
                for (size_t row = 0; row < m; ++row) {
                    for (size_t col = 0; col < ldd; ++col) {
                        if (row < mi || col < ni || col >= n) {
                            ASSERT_EQ(d[row * ldd + col], 93);
                            continue;
                        }
                        int64_t dot = options & 1 ? bias_i32[col] : 0;
                        for (size_t j = ki; j < ki + count; ++j) {
                            const int value =
                                ((b[col * (ldb / 2) + j / 2] >> (4 * (j % 2))) & 15) - 8;
                            dot += (int64_t)(a[row * lda + j] - 37) * value;
                        }
                        float real = (float)dot * scale_a[row] * scale_b[col];
                        if (options & 4)
                            real += c[row * (n + 3) + col];
                        if (options & 2)
                            real += bias[col];
                        // All operands above are exactly represented binary fractions.
                        ASSERT_EQ(d[row * ldd + col], torq_tile::test::reference_requantized_i8(
                                                          (int64_t)(real * 8), 0.25f, -7, -53, 79))
                            << "row=" << row << " col=" << col << " options=" << options
                            << " window=" << window << " mode=" << mode;
                    }
                }
                for (size_t i = m * ldd; i < d.size(); ++i)
                    ASSERT_EQ(d[i], 93);
            }
        }
    }
}

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_F16

#endif

#if TORQ_TILE_HAS_RVV_F16

#endif

#if TORQ_TILE_HAS_RVV_F16

#endif

#if TORQ_TILE_HAS_RVV_F16

#endif

#if TORQ_TILE_HAS_RVV_F16

#endif

#if TORQ_TILE_HAS_RVV_F16

#endif

#if TORQ_TILE_HAS_RVV_F16

#endif

#if TORQ_TILE_HAS_RVV_F16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_BF16

#endif

#if TORQ_TILE_HAS_RVV_F16

#endif  // TORQ_TILE_HAS_RVV_F16

TEST(W4Contract, gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv)
{
    check_w4_contract<int8_t, float, false, true, true>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            const void *aw =
                (const uint8_t *)a.data() +
                tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(mi, ki, ld, bl);
            const void *bw =
                b.data() +
                tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(ni, ki, ld, bl);
            const float *biasw =
                bias ? bias +
                           tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(ni) /
                               sizeof(float)
                     : nullptr;
            tqt_qai8dx_qsi4c2_params params{};
            std::vector<float> row_scales(fm);
            for (size_t i = 0; i < fm; ++i)
                row_scales[i] = sa[i * lds];
            params.scale_a = row_scales.data() + mi;
            params.zero_point_a = za.data() + mi;
            params.scale_b = sb.data() + ni * lds;
            params.ldsb = lds;
            params.lut = lut8;
            tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv(m, n, k, bl, aw, ld, ki, bw, ld,
                                                                    ki, c, ldd, d, ldd, biasw,
                                                                    (float)lo, (float)hi, &params);
        });
}

TEST(W4Contract, gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv)
{
    check_w4_contract<int8_t, float, false, true, true>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            std::vector<uint8_t> ap(
                tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(fm, fk)),
                bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(fn, fk));
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(fm, fk, ld, fk, 0,
                                                                            a.data(), ap.data());
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(fn, fk, ld, fk, 0,
                                                                             b.data(), bp.data());
            const void *aw =
                ap.data() +
                tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(mi, ki, fk,
                                                                                         m);
            const void *bw =
                bp.data() +
                tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(ni, ki, fk,
                                                                                         n);
            const float *biasw =
                bias
                    ? bias +
                          tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(ni) /
                              sizeof(float)
                    : nullptr;
            tqt_qai8dx_qsi4c2_params params{};
            std::vector<float> row_scales(fm);
            for (size_t i = 0; i < fm; ++i)
                row_scales[i] = sa[i * lds];
            params.scale_a = row_scales.data() + mi;
            params.zero_point_a = za.data() + mi;
            params.scale_b = sb.data() + ni * lds;
            params.ldsb = lds;
            params.lut = lut8;
            tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv(m, n, k, bl, aw, fk, ki, bw,
                                                                     fk, ki, c, ldd, d, ldd, biasw,
                                                                     (float)lo, (float)hi, &params);
        });
}

TEST(W4Contract, gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv)
{
    check_w4_contract<int8_t, float, true, true, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            const void *aw =
                (const uint8_t *)a.data() +
                tqt_get_a_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(mi, ki, ld, bl);
            const void *bw =
                b.data() +
                tqt_get_b_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(ni, ki, ld, bl);
            const float *biasw =
                bias ? bias +
                           tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(ni) /
                               sizeof(float)
                     : nullptr;
            tqt_qsi8d2_qai4c2_params params{};
            params.scale_a = sa.data() + mi * lds;
            params.ldsa = lds;
            params.scale_b = sb.data() + ni * lds;
            params.ldsb = lds;
            params.offset_b = ob.data() + ni * lds;
            params.ldob = lds;
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(m, n, k, bl, aw, ld, ki, bw, ld,
                                                                    ki, c, ldd, d, ldd, biasw,
                                                                    (float)lo, (float)hi, &params);
        });
}

TEST(W4Contract, gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv)
{
    check_w4_contract<int8_t, float, false, true, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            const void *aw =
                (const uint8_t *)a.data() +
                tqt_get_a_offset_gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv(mi, ki, ld, bl);
            const void *bw =
                b.data() +
                tqt_get_b_offset_gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv(ni, ki, ld, bl);
            const float *biasw =
                bias ? bias +
                           tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv(ni) /
                               sizeof(float)
                     : nullptr;
            tqt_qsi8d2_qsi4c2_params params{};
            params.scale_a = sa.data() + mi * lds;
            params.ldsa = lds;
            params.scale_b = sb.data() + ni * lds;
            params.ldsb = lds;
            params.lut = lut8;
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv(m, n, k, bl, aw, ld, ki, bw, ld,
                                                                    ki, c, ldd, d, ldd, biasw,
                                                                    (float)lo, (float)hi, &params);
        });
}

TEST(W4Contract, gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv)
{
    check_w4_contract<int8_t, float, true, true, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            std::vector<uint8_t> ap(
                tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(fm, fk)),
                bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(fn, fk));
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(fm, fk, ld, fk, 0,
                                                                            a.data(), ap.data());
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(fn, fk, ld, fk, 0,
                                                                             b.data(), bp.data());
            const void *aw =
                ap.data() +
                tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(mi, ki, fk,
                                                                                         m);
            const void *bw =
                bp.data() +
                tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(ni, ki, fk,
                                                                                         n);
            const float *biasw =
                bias
                    ? bias +
                          tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(ni) /
                              sizeof(float)
                    : nullptr;
            tqt_qsi8d2_qai4c2_params params{};
            params.scale_a = sa.data() + mi * lds;
            params.ldsa = lds;
            params.scale_b = sb.data() + ni * lds;
            params.ldsb = lds;
            params.offset_b = ob.data() + ni * lds;
            params.ldob = lds;
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(m, n, k, bl, aw, fk, ki, bw,
                                                                     fk, ki, c, ldd, d, ldd, biasw,
                                                                     (float)lo, (float)hi, &params);
        });
}

TEST(W4Contract, gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv)
{
    check_w4_contract<int8_t, float, false, true, false>(
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(),
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(), 0.00001f,
        [&](size_t m, size_t n, size_t k, size_t bl, size_t fm, size_t fn, size_t fk, size_t mi,
            size_t ni, size_t ki, size_t ld, size_t lds, const auto &a, const auto &b,
            const auto &sa, const auto &za, const auto &sb, const auto &ob, const float *bias,
            const float *c, float *d, size_t ldd, float lo, float hi, const float16_t *lut16,
            const int8_t *lut8) {
            std::vector<uint8_t> ap(
                tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(fm, fk)),
                bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(fn, fk));
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(fm, fk, ld, fk, 0,
                                                                            a.data(), ap.data());
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(fn, fk, ld, fk, 0,
                                                                             b.data(), bp.data());
            const void *aw =
                ap.data() +
                tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(mi, ki, fk,
                                                                                         m);
            const void *bw =
                bp.data() +
                tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(ni, ki, fk,
                                                                                         n);
            const float *biasw =
                bias
                    ? bias +
                          tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(ni) /
                              sizeof(float)
                    : nullptr;
            tqt_qsi8d2_qsi4c2_params params{};
            params.scale_a = sa.data() + mi * lds;
            params.ldsa = lds;
            params.scale_b = sb.data() + ni * lds;
            params.ldsb = lds;
            params.lut = lut8;
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv(m, n, k, bl, aw, fk, ki, bw,
                                                                     fk, ki, c, ldd, d, ldd, biasw,
                                                                     (float)lo, (float)hi, &params);
        });
}

TEST(LowbitOutputScale, gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv)
{
    torq_tile::test::check_lowbit_output_scales(
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv(),
        [&](size_t m, size_t n, float sa, float sd, const int32_t *bias, int8_t *d, size_t ldd) {
            const size_t k = 4;
            std::vector<int8_t> a(m * k, 0);
            std::vector<uint8_t> b(n * k / 2, 0x88);
            std::vector<float> sb(n, 1.0f);
            std::vector<float> scales(m, sa);
            std::vector<int32_t> za(m, 0);
            tqt_qai8_qai8dx_qsi4cx_params params{scales.data(), za.data(), sb.data(), sd, 7, bias};
            tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv(
                m, n, k, a.data(), k, 0, b.data(), k, 0, nullptr, n, d, ldd, nullptr, -53, 79,
                &params);
        });
}

TEST(LowbitOutputScale, gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv)
{
    torq_tile::test::check_lowbit_output_scales(
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(),
        [&](size_t m, size_t n, float sa, float sd, const int32_t *bias, int8_t *d, size_t ldd) {
            const size_t k = 4;
            std::vector<int8_t> a(m * k, 0);
            std::vector<uint8_t> b(n * k / 2, 0x88);
            std::vector<float> sb(n, 1.0f);
            std::vector<uint8_t> ap(
                tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                    m, k)),
                bp(tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                    n, k));
            tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                m, k, k, k, 0, a.data(), ap.data());
            tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                n, k, k, k, 0, b.data(), bp.data());
            std::vector<float> scales(m, sa);
            std::vector<int32_t> za(m, 0);
            tqt_qai8_qai8dx_qsi4cx_params params{scales.data(), za.data(), sb.data(), sd, 7, bias};
            tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
                m, n, k, ap.data(), k, 0, bp.data(), k, 0, nullptr, n, d, ldd, nullptr, -53, 79,
                &params);
        });
}

#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv.h"

#if TORQ_TILE_HAS_RVV_BF16
static W4A8GemmFuncs<bfloat16_t> make_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8GemmFuncs<bfloat16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm =
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv, bfloat16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>();
    return funcs;
}

class Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<bfloat16_t, bfloat16_t>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<bfloat16_t, bfloat16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv,
    with_clamp_dimensions(common_w4a8_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());
#endif

#if TORQ_TILE_HAS_RVV_BF16
static W4A8PackedGemmFuncs<bfloat16_t>
make_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8PackedGemmFuncs<bfloat16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm =
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv, bfloat16_t, 13,
                       11, 12, 14, ClampSource::kColumnBias, false, 5, true>();
    return funcs;
}

class Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<bfloat16_t, bfloat16_t>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<bfloat16_t, bfloat16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    with_clamp_dimensions(common_w4a8_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());
#endif

#if TORQ_TILE_HAS_RVV_F16
static W4A8GemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv_funcs()
{
    W4A8GemmFuncs<float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    funcs.run_gemm =
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>();
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<float16_t, float16_t>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv_funcs();
    run_w4a8_gemm_test<float16_t, float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv,
    with_clamp_dimensions(common_w4a8_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(), 1, true,
                          false),
    GemmTestParamNameGenerator());
#endif

#if TORQ_TILE_HAS_RVV_F16
static W4A8PackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs()
{
    W4A8PackedGemmFuncs<float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    funcs.run_gemm =
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv, float16_t, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>();
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<float16_t, float16_t>(GetParam(), funcs);
}

TEST_P(Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    auto funcs = make_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv_funcs();
    run_w4a8_packed_gemm_test<float16_t, float16_t>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(
    GemmTests, Test_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv,
    with_clamp_dimensions(common_w4a8_test_values(),
                          tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv(),
                          tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv(), 1,
                          true, false),
    GemmTestParamNameGenerator());
#endif
