//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

// ============================================================================
// Common test values shared by all W4A16 GEMM variants in this file.
// ============================================================================

static auto common_quantized_weight_test_values()
{
    return testing::Values(
        // Basic shapes with bl=2 (minimum)
        GemmTestParams(8, 8, 4, "Basic").set_bl(2), GemmTestParams(16, 16, 8, "Small").set_bl(2),
        GemmTestParams(32, 32, 32, "Medium_BL2").set_bl(2),
        // bl=4
        GemmTestParams(32, 32, 32, "Medium_BL4").set_bl(4),
        // bl=32 (typical LLM group_size)
        GemmTestParams(32, 32, 32, "Medium_BL32").set_bl(32),
        // Non-aligned M/N
        GemmTestParams(5, 11, 10, "NonAligned_BL2").set_bl(2),
        GemmTestParams(7, 3, 10, "SmallNonAligned").set_bl(2),
        GemmTestParams(17, 19, 24, "Odd_BL4").set_bl(4),
        // Larger shapes
        GemmTestParams(31, 35, 64, "Large_BL32").set_bl(32),
        GemmTestParams(60, 61, 128, "LargerK_BL32").set_bl(32),
        // With bias
        GemmTestParams(32, 32, 32, "WithBias").set_bl(2).set_bias(true),
        GemmTestParams(17, 19, 24, "WithBias_BL4").set_bl(4).set_bias(true),
        // With clamp
        GemmTestParams(32, 32, 32, "WithClamp").set_bl(4).set_clamp(-0.5f, 0.5f),
        // With C
        GemmTestParams(32, 32, 32, "WithC").set_bl(4).set_c(true),
        // Combined
        GemmTestParams(32, 32, 32, "WithC_WithBias_WithClamp")
            .set_bl(4)
            .set_c(true)
            .set_bias(true)
            .set_clamp(-0.5f, 0.5f));
}

// ============================================================================
// gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv (D=fp16)
// ============================================================================

static QuantizedWeightGemmFuncs<float16_t, float16_t>
make_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv_funcs()
{
    QuantizedWeightGemmFuncs<float16_t, float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_a_offset = tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_b_offset = tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_scale_b_offset = tqt_get_scale_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_c_offset = tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_bias_offset = tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_size = tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv_funcs();
    run_quantized_weight_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv,
                         common_quantized_weight_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv (D=fp16, packed A and B)
// ============================================================================

static PackedQuantizedWeightGemmFuncs<float16_t, float16_t>
make_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv_funcs()
{
    PackedQuantizedWeightGemmFuncs<float16_t, float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_scale_b_offset = tqt_get_scale_b_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_c_offset = tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_bias_offset = tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_size = tqt_get_d_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv_funcs();
    run_packed_quantized_weight_gemm_test<float16_t, float16_t, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv,
                         common_quantized_weight_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv (D=fp32, A=fp16)
// ============================================================================

static QuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv_funcs()
{
    QuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_a_offset = tqt_get_a_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_b_offset = tqt_get_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_scale_b_offset = tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_c_offset = tqt_get_c_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_bias_offset = tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_size = tqt_get_d_size_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv_funcs();
    run_quantized_weight_gemm_test<float16_t, float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv,
                         common_quantized_weight_test_values(), GemmTestParamNameGenerator());

// ============================================================================
// gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv (D=fp32, A=fp16, packed A and B)
// ============================================================================

static PackedQuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv_funcs()
{
    PackedQuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_offset =
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_offset =
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_scale_b_offset = tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_c_offset = tqt_get_c_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_bias_offset = tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_size = tqt_get_d_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    return funcs;
}

class Test_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv_funcs();
    run_packed_quantized_weight_gemm_test<float16_t, float, BiasMode::k1xN>(GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
                         common_quantized_weight_test_values(), GemmTestParamNameGenerator());
