//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16p/tqt_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16p_f16p/tqt_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16_f16/tqt_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv.h"
#include "src/gemm/gemm_f16_f16/gemm_mx1bias_clamp_f16_f16p_f16p/tqt_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// NonPacked, B RowMajor: gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_f16_f16_f16_8x3vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv,
    };
}

// ============================================================================
// NonPacked, B Transposed: gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_f16_f16_f16t_8x3vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv,
    };
}

// ============================================================================
// NonPacked, B RowMajor, Mx1 bias: gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_mx1bias_f16_f16_f16_8x3vl_rvv()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_a_offset_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_b_offset_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
        tqt_run_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv,
    };
}

// ============================================================================
// Packed, 1xN bias: gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv
// ============================================================================

static PackedGemmFuncs<float16_t> make_packed_1xnbias_f16_f16p_f16p_8x3vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv,
    };
}

// ============================================================================
// Packed, Mx1 bias: gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv
// ============================================================================

static PackedGemmFuncs<float16_t> make_packed_mx1bias_f16_f16p_f16p_8x3vl_rvv()
{
    return {
        tqt_get_m_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_n_step_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_packed_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_a_packed_size_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_get_b_packed_size_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_a_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_bt_pack_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
        tqt_run_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv,
    };
}

// ============================================================================
// B-only-packed, 1xN bias: gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv
// ============================================================================

static BOnlyPackedGemmFuncs<float16_t> make_b_only_packed_1xnbias_f16_f16_f16p_8x3vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    };
}

// ============================================================================
// NonPacked, B RowMajor: gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_f16_f16_f16_4x4vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
    };
}

// ============================================================================
// NonPacked, B Transposed: gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_f16_f16_f16t_4x4vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv,
    };
}

// ============================================================================
// NonPacked, B RowMajor: gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_f16_f16_f16_2x8vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv,
    };
}

// ============================================================================
// NonPacked, B Transposed: gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_f16_f16_f16t_2x8vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv,
    };
}

// ============================================================================
// Packed, 1xN bias: gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv
// ============================================================================

static PackedGemmFuncs<float16_t> make_packed_1xnbias_f16_f16p_f16p_4x4vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv,
    };
}

// ============================================================================
// Packed, 1xN bias: gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv
// ============================================================================

static PackedGemmFuncs<float16_t> make_packed_1xnbias_f16_f16p_f16p_2x8vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_gemm_f16_f16_rvv_benchmarks()
{
    register_nonpacked_gemm_benchmark<float16_t, float16_t, BLayout::kRowMajor>(
        "tqt_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv", make_nonpacked_f16_f16_f16_8x3vl_rvv());

    register_nonpacked_gemm_benchmark<float16_t, float16_t, BLayout::kRowMajor>(
        "tqt_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv", make_nonpacked_f16_f16_f16_4x4vl_rvv());

    register_nonpacked_gemm_benchmark<float16_t, float16_t, BLayout::kRowMajor>(
        "tqt_gemm_1xnbias_clamp_f16_f16_f16_2x8vl_rvv", make_nonpacked_f16_f16_f16_2x8vl_rvv());

    register_nonpacked_gemm_benchmark<float16_t, float16_t, BLayout::kTransposed>(
        "tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv", make_nonpacked_f16_f16_f16t_8x3vl_rvv());

    register_nonpacked_gemm_benchmark<float16_t, float16_t, BLayout::kTransposed>(
        "tqt_gemm_1xnbias_clamp_f16_f16_f16t_4x4vl_rvv", make_nonpacked_f16_f16_f16t_4x4vl_rvv());

    register_nonpacked_gemm_benchmark<float16_t, float16_t, BLayout::kTransposed>(
        "tqt_gemm_1xnbias_clamp_f16_f16_f16t_2x8vl_rvv", make_nonpacked_f16_f16_f16t_2x8vl_rvv());

    register_nonpacked_gemm_benchmark<float16_t, float16_t, BLayout::kRowMajor>(
        "tqt_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv",
        make_nonpacked_mx1bias_f16_f16_f16_8x3vl_rvv());

    register_packed_gemm_benchmark<float16_t, float16_t>(
        "tqt_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv",
        make_packed_1xnbias_f16_f16p_f16p_8x3vl_rvv());

    register_packed_gemm_benchmark<float16_t, float16_t>(
        "tqt_gemm_1xnbias_clamp_f16_f16p_f16p_4x4vl_rvv",
        make_packed_1xnbias_f16_f16p_f16p_4x4vl_rvv());

    register_packed_gemm_benchmark<float16_t, float16_t>(
        "tqt_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv",
        make_packed_1xnbias_f16_f16p_f16p_2x8vl_rvv());

    register_packed_gemm_benchmark<float16_t, float16_t>(
        "tqt_gemm_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv",
        make_packed_mx1bias_f16_f16p_f16p_8x3vl_rvv());

    register_b_only_packed_gemm_benchmark<float16_t, float16_t>(
        "tqt_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv",
        make_b_only_packed_1xnbias_f16_f16_f16p_8x3vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
