//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/conv_benchmark_runner.h"
#include "src/conv/conv_f32_f32/convcf_mx1bias_clamp_f32_f32_f32/tqt_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv.h"
#include "src/conv/conv_f32_f32/convcf_mx1bias_clamp_f32_f32p_f32p/tqt_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv.h"
#include "src/conv/conv_f32_f32/convcl_1xnbias_clamp_f32_f32_f32t/tqt_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.h"
#include "src/conv/conv_f32_f32/convcl_1xnbias_clamp_f32_f32p_f32p/tqt_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Non-packed Conv-CF: convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv
// ============================================================================

static NonPackedConvCfBenchFuncs<float> make_nonpacked_convcf_f32_8x3vl_rvv()
{
    return {
        tqt_get_b_im2col_size_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_run_im2col_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
        tqt_run_conv_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv,
    };
}

// ============================================================================
// Non-packed Conv-CL: convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv
// ============================================================================

static NonPackedConvClBenchFuncs<float> make_nonpacked_convcl_f32_8x3vl_rvv()
{
    return {
        tqt_get_a_im2col_size_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_run_im2col_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
        tqt_run_conv_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv,
    };
}

// ============================================================================
// Packed Conv-CF: convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv
// ============================================================================

static PackedConvCfBenchFuncs<float> make_packed_convcf_f32_8x3vl_rvv()
{
    return {
        tqt_get_a_packed_size_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_b_im2colpack_size_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_a_pack_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_b_im2colpack_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_conv_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv,
    };
}

// ============================================================================
// Packed Conv-CL: convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv
// ============================================================================

static PackedConvClBenchFuncs<float> make_packed_convcl_f32_8x3vl_rvv()
{
    return {
        tqt_get_a_im2colpack_size_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_get_b_packed_size_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_bt_pack_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_a_im2colpack_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
        tqt_run_conv_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_conv_f32_f32_rvv_benchmarks()
{
    register_nonpacked_convcf_benchmark<float, float>(
        "tqt_convcf_mx1bias_clamp_f32_f32_f32_8x3vl_rvv", make_nonpacked_convcf_f32_8x3vl_rvv());

    register_nonpacked_convcl_benchmark<float, float>(
        "tqt_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv", make_nonpacked_convcl_f32_8x3vl_rvv());

    register_packed_convcf_benchmark<float, float>(
        "tqt_convcf_mx1bias_clamp_f32_f32p_f32p_8x3vl_rvv", make_packed_convcf_f32_8x3vl_rvv());

    register_packed_convcl_benchmark<float, float>(
        "tqt_convcl_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv", make_packed_convcl_f32_8x3vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
