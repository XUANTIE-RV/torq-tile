//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/conv_benchmark_runner.h"
#include "src/conv/conv_bf16_bf16/convcf_mx1bias_clamp_bf16_bf16_bf16/tqt_convcf_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv.h"
#include "src/conv/conv_bf16_bf16/convcf_mx1bias_clamp_bf16_bf16p_bf16p/tqt_convcf_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv.h"
#include "src/conv/conv_bf16_bf16/convcl_1xnbias_clamp_bf16_bf16_bf16t/tqt_convcl_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv.h"
#include "src/conv/conv_bf16_bf16/convcl_1xnbias_clamp_bf16_bf16p_bf16p/tqt_convcl_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Non-packed Conv-CF: convcf_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv
// ============================================================================

static NonPackedConvCfBenchFuncs<bfloat16_t> make_nonpacked_convcf_bf16_8x1vl_rvv()
{
    return {
        tqt_get_b_im2col_size_convcf_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_run_im2col_convcf_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_run_conv_convcf_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    };
}

// ============================================================================
// Non-packed Conv-CL: convcl_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv
// ============================================================================

static NonPackedConvClBenchFuncs<bfloat16_t> make_nonpacked_convcl_bf16_8x1vl_rvv()
{
    return {
        tqt_get_a_im2col_size_convcl_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_run_im2col_convcl_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_run_conv_convcl_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
    };
}

// ============================================================================
// Packed Conv-CF: convcf_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv
// ============================================================================

static PackedConvCfBenchFuncs<bfloat16_t> make_packed_convcf_bf16_8x1vl_rvv()
{
    return {
        tqt_get_a_packed_size_convcf_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_im2colpack_size_convcf_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_pack_convcf_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_b_im2colpack_convcf_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_conv_convcf_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
    };
}

// ============================================================================
// Packed Conv-CL: convcl_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv
// ============================================================================

static PackedConvClBenchFuncs<bfloat16_t> make_packed_convcl_bf16_8x1vl_rvv()
{
    return {
        tqt_get_a_im2colpack_size_convcl_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_convcl_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_convcl_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_im2colpack_convcl_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_conv_convcl_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_conv_bf16_bf16_rvv_benchmarks()
{
    register_nonpacked_convcf_benchmark<bfloat16_t, bfloat16_t>(
        "tqt_convcf_mx1bias_clamp_bf16_bf16_bf16_8x1vl_rvv",
        make_nonpacked_convcf_bf16_8x1vl_rvv());

    register_nonpacked_convcl_benchmark<bfloat16_t, bfloat16_t>(
        "tqt_convcl_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv",
        make_nonpacked_convcl_bf16_8x1vl_rvv());

    register_packed_convcf_benchmark<bfloat16_t, bfloat16_t>(
        "tqt_convcf_mx1bias_clamp_bf16_bf16p_bf16p_8x1vl_rvv", make_packed_convcf_bf16_8x1vl_rvv());

    register_packed_convcl_benchmark<bfloat16_t, bfloat16_t>(
        "tqt_convcl_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv", make_packed_convcl_bf16_8x1vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
