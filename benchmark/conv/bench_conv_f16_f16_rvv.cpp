//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/conv_benchmark_runner.h"
#include "src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16_f16/tqt_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv.h"
#include "src/conv/conv_f16_f16/convcf_mx1bias_clamp_f16_f16p_f16p/tqt_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv.h"
#include "src/conv/conv_f16_f16/convcl_1xnbias_clamp_f16_f16_f16t/tqt_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h"
#include "src/conv/conv_f16_f16/convcl_1xnbias_clamp_f16_f16p_f16p/tqt_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Non-packed Conv-CF: convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv
// ============================================================================

static NonPackedConvCfBenchFuncs<float16_t> make_nonpacked_convcf_f16_8x3vl_rvv()
{
    return {
        tqt_get_b_im2col_size_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_run_im2col_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_run_conv_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
    };
}

// ============================================================================
// Non-packed Conv-CL: convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv
// ============================================================================

static NonPackedConvClBenchFuncs<float16_t> make_nonpacked_convcl_f16_8x3vl_rvv()
{
    return {
        tqt_get_a_im2col_size_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_run_im2col_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_run_conv_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
    };
}

// ============================================================================
// Packed Conv-CF: convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv
// ============================================================================

static PackedConvCfBenchFuncs<float16_t> make_packed_convcf_f16_8x3vl_rvv()
{
    return {
        tqt_get_a_packed_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_im2colpack_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_a_pack_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_b_im2colpack_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_conv_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
    };
}

// ============================================================================
// Packed Conv-CL: convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv
// ============================================================================

static PackedConvClBenchFuncs<float16_t> make_packed_convcl_f16_8x3vl_rvv()
{
    return {
        tqt_get_a_im2colpack_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_size_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_bt_pack_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_a_im2colpack_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_conv_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_conv_f16_f16_rvv_benchmarks()
{
    register_nonpacked_convcf_benchmark<float16_t, float16_t>(
        "tqt_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv", make_nonpacked_convcf_f16_8x3vl_rvv());

    register_nonpacked_convcl_benchmark<float16_t, float16_t>(
        "tqt_convcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv", make_nonpacked_convcl_f16_8x3vl_rvv());

    register_packed_convcf_benchmark<float16_t, float16_t>(
        "tqt_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv", make_packed_convcf_f16_8x3vl_rvv());

    register_packed_convcl_benchmark<float16_t, float16_t>(
        "tqt_convcl_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv", make_packed_convcl_f16_8x3vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
