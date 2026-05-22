//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "src/conv/igemm_i8_i8/igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv.h"
#include "src/conv/igemm_i8_i8/igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8/tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv.h"
#include "src/conv/igemm_i8_i8/igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp/tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv.h"
#include "src/conv/igemm_i8_i8/igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.h"
#include "test/common/quantized_igemm_test_runner.h"

using namespace torq_tile::test;

// ============================================================================
// Common test cases for int8 igemm
// ============================================================================

static auto i8_common_igemm_test_values()
{
    return testing::Values(
        // Basic 2D convolutions
        ConvTestParams(3, 16, 1, 8, 8, 3, 3), ConvTestParams(16, 32, 1, 8, 8, 3, 3),
        ConvTestParams(16, 32, 1, 8, 8, 1, 1),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Pad1").set_pad(1, 1),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Stride2").set_stride(2, 2),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "Dilation2").set_dilation(2, 2).set_pad(2, 2),
        ConvTestParams(3, 16, 1, 8, 8, 3, 3, "PadStride").set_pad(1, 1).set_stride(2, 2),
        // Non-square
        ConvTestParams(8, 16, 1, 7, 11, 3, 5),
        // 1x1 conv
        ConvTestParams(64, 128, 1, 4, 4, 1, 1),
        // Larger
        ConvTestParams(3, 32, 1, 16, 16, 3, 3, "Large").set_pad(1, 1),
        // Small IC/OC
        ConvTestParams(3, 8, 1, 6, 6, 3, 3),
        // Larger IC
        ConvTestParams(64, 32, 1, 4, 4, 3, 3, "LargeIC").set_pad(1, 1));
}

// ============================================================================
// Igemmcf NonPacked (mx1bias, reqi32toi8, qai8_qsi8dx_qai8) - RVV 1vlx4
// ============================================================================

static NonPackedQuantizedIgemmcfFuncs<tqt_qai8_qsi8dx_qai8_params>
make_igemmcf_nonpacked_reqi32toi8_funcs()
{
    return {
        tqt_get_m_step_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv,
        tqt_get_n_step_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv,
        tqt_run_igemm_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv,
    };
}

class Test_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv, Correctness)
{
    run_nonpacked_igemmcf_reqi32toi8_test(GetParam(), make_igemmcf_nonpacked_reqi32toi8_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests,
                         Test_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv,
                         i8_common_igemm_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// Igemmcf Packed (mx1bias, reqi32toi8, qai8_qsi8dxp_qai8) - RVV 1vlx4
// ============================================================================

static PackedQuantizedIgemmcfFuncs<tqt_qai8_qsi8dx_qai8_params>
make_igemmcf_packed_reqi32toi8_funcs()
{
    return {
        tqt_get_m_step_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv,
        tqt_get_n_step_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv,
        tqt_get_a_packed_size_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv,
        tqt_run_a_pack_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv,
        tqt_run_igemm_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv,
    };
}

class Test_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv, Correctness)
{
    run_packed_igemmcf_reqi32toi8_test(GetParam(), make_igemmcf_packed_reqi32toi8_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests,
                         Test_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv,
                         i8_common_igemm_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// Igemmcl NonPacked (1xnbias, reqi32toi8, qai8_qai8_qsi8cxt) - RVV 4x1vl
// ============================================================================

static NonPackedQuantizedIgemmclFuncs<tqt_qai8_qai8_qsi8cx_params>
make_igemmcl_nonpacked_reqi32toi8_funcs()
{
    return {
        tqt_get_m_step_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_n_step_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_run_igemm_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
    };
}

class Test_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv, Correctness)
{
    run_nonpacked_igemmcl_reqi32toi8_test(GetParam(), make_igemmcl_nonpacked_reqi32toi8_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests,
                         Test_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
                         i8_common_igemm_test_values(), ConvTestParamNameGenerator());

// ============================================================================
// Igemmcl Packed (1xnbias, reqi32toi8, qai8_qai8_qsi8cxp) - RVV 4x1vl
// ============================================================================

static PackedQuantizedIgemmclFuncs<tqt_qai8_qai8_qsi8cx_params>
make_igemmcl_packed_reqi32toi8_funcs()
{
    return {
        tqt_get_m_step_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_run_igemm_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    };
}

class Test_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv
    : public testing::TestWithParam<ConvTestParams>
{
};

TEST_P(Test_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv, Correctness)
{
    run_packed_igemmcl_reqi32toi8_test(GetParam(), make_igemmcl_packed_reqi32toi8_funcs());
}

INSTANTIATE_TEST_SUITE_P(ConvTests,
                         Test_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
                         i8_common_igemm_test_values(), ConvTestParamNameGenerator());
