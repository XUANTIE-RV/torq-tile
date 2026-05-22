//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/quantized_conv_benchmark_runner.h"
#include "src/conv/conv_i8_i8/convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.h"
#include "src/conv/conv_i8_i8/convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p/tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv.h"
#include "src/conv/conv_i8_i8/convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.h"
#include "src/conv/conv_i8_i8/convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp/tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Non-packed Conv-CF: convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv
// ============================================================================

static NonPackedRequantizedConvCfBenchFuncs make_nonpacked_convcf_i8_4x1vl_rvv()
{
    return {
        tqt_get_b_im2col_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_run_im2col_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_run_conv_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    };
}

// ============================================================================
// Non-packed Conv-CL: convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv
// ============================================================================

static NonPackedRequantizedConvClBenchFuncs make_nonpacked_convcl_i8_4x1vl_rvv()
{
    return {
        tqt_get_a_im2col_size_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_run_im2col_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_run_conv_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
    };
}

// ============================================================================
// Packed Conv-CF: convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv
// ============================================================================

static PackedRequantizedConvCfBenchFuncs make_packed_convcf_i8_4x1vl_rvv()
{
    return {
        tqt_get_a_packed_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_b_im2colpack_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_a_pack_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_b_im2colpack_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_conv_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
    };
}

// ============================================================================
// Packed Conv-CL: convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv
// ============================================================================

static PackedRequantizedConvClBenchFuncs make_packed_convcl_i8_4x1vl_rvv()
{
    return {
        tqt_get_a_im2colpack_size_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_a_im2colpack_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_conv_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_conv_i8_i8_rvv_benchmarks()
{
    register_nonpacked_requantized_convcf_benchmark(
        "tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv",
        make_nonpacked_convcf_i8_4x1vl_rvv());

    register_nonpacked_requantized_convcl_benchmark(
        "tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv",
        make_nonpacked_convcl_i8_4x1vl_rvv());

    register_packed_requantized_convcf_benchmark(
        "tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv",
        make_packed_convcf_i8_4x1vl_rvv());

    register_packed_requantized_convcl_benchmark(
        "tqt_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv",
        make_packed_convcl_i8_4x1vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
