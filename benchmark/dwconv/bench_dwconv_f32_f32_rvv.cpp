//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/dwconv_benchmark_runner.h"
// Generic kernels (4)
#include "src/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32_f32_f32/tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcf_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl_1xnbias_clamp_f32_f32_f32t/tqt_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv.h"
// Specialized kernels (6)
#include "src/dwconv/dwconv_f32_f32/dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p/tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv.h"
#include "src/dwconv/dwconv_f32_f32/dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p/tqt_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Non-packed dwconv-CF: dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv
// ============================================================================

static NonPackedDwconvBenchFuncs<float> make_nonpacked_dwconvcf_f32_1x1vl_rvv()
{
    return {
        tqt_run_dwconv_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv,
    };
}

// ============================================================================
// Non-packed dwconv-CL: dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv
// ============================================================================

static NonPackedDwconvBenchFuncs<float> make_nonpacked_dwconvcl_f32_1x1vl_rvv()
{
    return {
        tqt_run_dwconv_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv,
    };
}

// ============================================================================
// Generic packed dwconv-CF: dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv
// ============================================================================

static GenericPackedDwconvBenchFuncs<float> make_generic_packed_dwconvcf_f32p_1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_get_output_size_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_pack_input_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_unpack_output_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_dwconv_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv,
    };
}

// ============================================================================
// Generic packed dwconv-CL: dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv
// ============================================================================

static GenericPackedDwconvBenchFuncs<float> make_generic_packed_dwconvcl_f32p_1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_get_output_size_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_pack_input_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_unpack_output_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
        tqt_run_dwconv_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CF 3x3s1
// ============================================================================

static SpecPackedDwconvBenchFuncs<float> make_spec_packed_dwconvcf3x3s1_f32p_4x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_output_size_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CF 3x3s2
// ============================================================================

static SpecPackedDwconvBenchFuncs<float> make_spec_packed_dwconvcf3x3s2_f32p_2x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_output_size_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CF 5x5s1
// ============================================================================

static SpecPackedDwconvBenchFuncs<float> make_spec_packed_dwconvcf5x5s1_f32p_4x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_output_size_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CL 3x3s1
// ============================================================================

static SpecPackedDwconvBenchFuncs<float> make_spec_packed_dwconvcl3x3s1_f32p_4x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_output_size_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CL 3x3s2
// ============================================================================

static SpecPackedDwconvBenchFuncs<float> make_spec_packed_dwconvcl3x3s2_f32p_2x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_output_size_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv,
    };
}

// ============================================================================
// Specialized packed dwconv-CL 5x5s1
// ============================================================================

static SpecPackedDwconvBenchFuncs<float> make_spec_packed_dwconvcl5x5s1_f32p_4x1x1vl_rvv()
{
    return {
        tqt_get_packed_input_size_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_packed_weight_size_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_output_size_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_get_oh_step_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_input_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_pack_weight_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
        tqt_run_dwconv_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_dwconv_f32_f32_rvv_benchmarks()
{
    register_nonpacked_dwconv_benchmark<float, float>(
        "tqt_dwconvcf_mx1bias_clamp_f32_f32_f32_1x1vl_rvv",
        make_nonpacked_dwconvcf_f32_1x1vl_rvv());

    register_nonpacked_dwconv_benchmark<float, float>(
        "tqt_dwconvcl_1xnbias_clamp_f32_f32_f32t_1x1vl_rvv",
        make_nonpacked_dwconvcl_f32_1x1vl_rvv());

    register_generic_packed_dwconv_benchmark<float, float>(
        "tqt_dwconvcf_mx1bias_clamp_f32p_f32p_f32p_1x1vl_rvv",
        make_generic_packed_dwconvcf_f32p_1x1vl_rvv());

    register_generic_packed_dwconv_benchmark<float, float>(
        "tqt_dwconvcl_1xnbias_clamp_f32p_f32p_f32p_1x1vl_rvv",
        make_generic_packed_dwconvcl_f32p_1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float, float>(
        "tqt_dwconvcf3x3s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv",
        make_spec_packed_dwconvcf3x3s1_f32p_4x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float, float>(
        "tqt_dwconvcf3x3s2_mx1bias_clamp_f32p_f32p_f32p_2x1x1vl_rvv",
        make_spec_packed_dwconvcf3x3s2_f32p_2x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float, float>(
        "tqt_dwconvcf5x5s1_mx1bias_clamp_f32p_f32p_f32p_4x1x1vl_rvv",
        make_spec_packed_dwconvcf5x5s1_f32p_4x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float, float>(
        "tqt_dwconvcl3x3s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv",
        make_spec_packed_dwconvcl3x3s1_f32p_4x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float, float>(
        "tqt_dwconvcl3x3s2_1xnbias_clamp_f32p_f32p_f32p_2x1x1vl_rvv",
        make_spec_packed_dwconvcl3x3s2_f32p_2x1x1vl_rvv());

    register_spec_packed_dwconv_benchmark<float, float>(
        "tqt_dwconvcl5x5s1_1xnbias_clamp_f32p_f32p_f32p_4x1x1vl_rvv",
        make_spec_packed_dwconvcl5x5s1_f32p_4x1x1vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
