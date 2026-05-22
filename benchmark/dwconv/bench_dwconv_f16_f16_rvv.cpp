//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/dwconv_benchmark_runner.h"
// Generic kernels (4)
#include "src/dwconv/dwconv_f16_f16/dwconvcf_mx1bias_clamp_f16_f16_f16/tqt_dwconvcf_mx1bias_clamp_f16_f16_f16_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcf_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl_1xnbias_clamp_f16_f16_f16t/tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv.h"
// Specialized kernels (6)
#include "src/dwconv/dwconv_f16_f16/dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p/tqt_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f16_f16/dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p/tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Non-packed dwconv-CF: dwconvcf_mx1bias_clamp_f16_f16_f16_1x1vl_rvv
// ============================================================================

static NonPackedDwconvBenchFuncs<float16_t> make_nonpacked_dwconvcf_f16_1x1vl_rvv()
{
    return {
        tqt_run_dwconv_dwconvcf_mx1bias_clamp_f16_f16_f16_1x1vl_rvv,
    };
}

// ============================================================================
// Non-packed dwconv-CL: dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv
// ============================================================================

static NonPackedDwconvBenchFuncs<float16_t> make_nonpacked_dwconvcl_f16_1x1vl_rvv()
{
    return {
        tqt_run_dwconv_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv,
    };
}

// ============================================================================
// Generic packed dwconv-CF: dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv
// ============================================================================

static GenericPackedDwconvBenchFuncs<float16_t> make_generic_packed_dwconvcf_f16p_1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_get_output_size_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_pack_input_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_dwconv_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv,
    };
}

// ============================================================================
// Generic packed dwconv-CL: dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv
// ============================================================================

static GenericPackedDwconvBenchFuncs<float16_t> make_generic_packed_dwconvcl_f16p_1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_get_output_size_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_pack_input_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
        tqt_run_dwconv_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CF 3x3s1
// ============================================================================

static SpecPackedDwconvBenchFuncs<float16_t> make_spec_packed_dwconvcf3x3s1_f16p_4x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_output_size_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CF 3x3s2
// ============================================================================

static SpecPackedDwconvBenchFuncs<float16_t> make_spec_packed_dwconvcf3x3s2_f16p_2x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_output_size_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CF 5x5s1
// ============================================================================

static SpecPackedDwconvBenchFuncs<float16_t> make_spec_packed_dwconvcf5x5s1_f16p_4x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_output_size_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CL 3x3s1
// ============================================================================

static SpecPackedDwconvBenchFuncs<float16_t> make_spec_packed_dwconvcl3x3s1_f16p_4x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_output_size_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CL 3x3s2
// ============================================================================

static SpecPackedDwconvBenchFuncs<float16_t> make_spec_packed_dwconvcl3x3s2_f16p_2x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_output_size_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CL 5x5s1
// ============================================================================

static SpecPackedDwconvBenchFuncs<float16_t> make_spec_packed_dwconvcl5x5s1_f16p_4x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_output_size_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_dwconv_f16_f16_rvv_benchmarks()
{
    register_nonpacked_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcf_mx1bias_clamp_f16_f16_f16_1x1vl_rvv",
        make_nonpacked_dwconvcf_f16_1x1vl_rvv());

    register_nonpacked_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcl_1xnbias_clamp_f16_f16_f16t_1x1vl_rvv",
        make_nonpacked_dwconvcl_f16_1x1vl_rvv());

    register_generic_packed_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcf_mx1bias_clamp_f16p_f16p_f16p_1x1vl_rvv",
        make_generic_packed_dwconvcf_f16p_1x1vl_rvv());

    register_generic_packed_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcl_1xnbias_clamp_f16p_f16p_f16p_1x1vl_rvv",
        make_generic_packed_dwconvcl_f16p_1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcf3x3s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv",
        make_spec_packed_dwconvcf3x3s1_f16p_4x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcf3x3s2_mx1bias_clamp_f16p_f16p_f16p_2x1x1vl_rvv",
        make_spec_packed_dwconvcf3x3s2_f16p_2x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcf5x5s1_mx1bias_clamp_f16p_f16p_f16p_4x1x1vl_rvv",
        make_spec_packed_dwconvcf5x5s1_f16p_4x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcl3x3s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv",
        make_spec_packed_dwconvcl3x3s1_f16p_4x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcl3x3s2_1xnbias_clamp_f16p_f16p_f16p_2x1x1vl_rvv",
        make_spec_packed_dwconvcl3x3s2_f16p_2x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float16_t, float16_t>(
        "tqt_dwconvcl5x5s1_1xnbias_clamp_f16p_f16p_f16p_4x1x1vl_rvv",
        make_spec_packed_dwconvcl5x5s1_f16p_4x1x1vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
