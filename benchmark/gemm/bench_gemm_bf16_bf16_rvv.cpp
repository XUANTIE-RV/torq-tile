//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16p/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16t/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16p_bf16p/tqt_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f16_bf16_bf16/tqt_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f16_bf16_bf16t/tqt_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f16_bf16p_bf16p/tqt_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16_bf16/tqt_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16_bf16t/tqt_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_f32_bf16p_bf16p/tqt_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// NonPacked, B RowMajor: gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<bfloat16_t> make_nonpacked_bf16_bf16_bf16_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    };
}

// ============================================================================
// NonPacked, B Transposed: gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<bfloat16_t> make_nonpacked_bf16_bf16_bf16t_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv,
    };
}

// ============================================================================
// Packed, 1xN bias: gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv
// ============================================================================

static PackedGemmFuncs<bfloat16_t> make_packed_1xnbias_bf16_bf16p_bf16p_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv,
    };
}

// ============================================================================
// B-only-packed, 1xN bias: gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv
// ============================================================================

static BOnlyPackedGemmFuncs<bfloat16_t> make_b_only_packed_1xnbias_bf16_bf16_bf16p_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv,
    };
}

// ============================================================================
// D=f16, NonPacked, B RowMajor: gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_f16_bf16_bf16_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv,
    };
}

// ============================================================================
// D=f16, NonPacked, B Transposed: gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_f16_bf16_bf16t_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv,
    };
}

// ============================================================================
// D=f16, Packed, 1xN bias: gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv
// ============================================================================

static PackedGemmFuncs<float16_t> make_packed_1xnbias_f16_bf16p_bf16p_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv,
    };
}

// ============================================================================
// D=f32, NonPacked, B RowMajor: gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float> make_nonpacked_f32_bf16_bf16_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    };
}

// ============================================================================
// D=f32, NonPacked, B Transposed: gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float> make_nonpacked_f32_bf16_bf16t_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv,
    };
}

// ============================================================================
// D=f32, Packed, 1xN bias: gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv
// ============================================================================

static PackedGemmFuncs<float> make_packed_1xnbias_f32_bf16p_bf16p_8x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_gemm_bf16_bf16_rvv_benchmarks()
{
    register_nonpacked_gemm_benchmark<bfloat16_t, bfloat16_t, BLayout::kRowMajor>(
        "tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv",
        make_nonpacked_bf16_bf16_bf16_8x1vl_rvv());

    register_nonpacked_gemm_benchmark<bfloat16_t, bfloat16_t, BLayout::kTransposed>(
        "tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv",
        make_nonpacked_bf16_bf16_bf16t_8x1vl_rvv());

    register_packed_gemm_benchmark<bfloat16_t, bfloat16_t>(
        "tqt_gemm_1xnbias_clamp_bf16_bf16p_bf16p_8x1vl_rvv",
        make_packed_1xnbias_bf16_bf16p_bf16p_8x1vl_rvv());

    register_b_only_packed_gemm_benchmark<bfloat16_t, bfloat16_t>(
        "tqt_gemm_1xnbias_clamp_bf16_bf16_bf16p_8x1vl_rvv",
        make_b_only_packed_1xnbias_bf16_bf16_bf16p_8x1vl_rvv());

    register_nonpacked_gemm_benchmark<bfloat16_t, float16_t, BLayout::kRowMajor>(
        "tqt_gemm_1xnbias_clamp_f16_bf16_bf16_8x1vl_rvv", make_nonpacked_f16_bf16_bf16_8x1vl_rvv());

    register_nonpacked_gemm_benchmark<bfloat16_t, float16_t, BLayout::kTransposed>(
        "tqt_gemm_1xnbias_clamp_f16_bf16_bf16t_8x1vl_rvv",
        make_nonpacked_f16_bf16_bf16t_8x1vl_rvv());

    register_packed_gemm_benchmark<bfloat16_t, float16_t>(
        "tqt_gemm_1xnbias_clamp_f16_bf16p_bf16p_8x1vl_rvv",
        make_packed_1xnbias_f16_bf16p_bf16p_8x1vl_rvv());

    register_nonpacked_gemm_benchmark<bfloat16_t, float, BLayout::kRowMajor>(
        "tqt_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv", make_nonpacked_f32_bf16_bf16_8x1vl_rvv());

    register_nonpacked_gemm_benchmark<bfloat16_t, float, BLayout::kTransposed>(
        "tqt_gemm_1xnbias_clamp_f32_bf16_bf16t_8x1vl_rvv",
        make_nonpacked_f32_bf16_bf16t_8x1vl_rvv());

    register_packed_gemm_benchmark<bfloat16_t, float>(
        "tqt_gemm_1xnbias_clamp_f32_bf16p_bf16p_8x1vl_rvv",
        make_packed_1xnbias_f32_bf16p_bf16p_8x1vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
