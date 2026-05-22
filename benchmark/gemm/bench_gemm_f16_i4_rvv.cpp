//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv.h"

using namespace torq_tile::benchmark;

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// QuantizedWeight: gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv
// ============================================================================

static QuantizedWeightGemmFuncs<float16_t, float16_t>
make_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv()
{
    QuantizedWeightGemmFuncs<float16_t, float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_a_offset = tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_b_offset = tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    return funcs;
}

// ============================================================================
// PackedQuantizedWeight: gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv
// ============================================================================

static PackedQuantizedWeightGemmFuncs<float16_t, float16_t>
make_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv()
{
    PackedQuantizedWeightGemmFuncs<float16_t, float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    return funcs;
}

// ============================================================================
// QuantizedWeight: gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv
// ============================================================================

static QuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv()
{
    QuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_a_offset = tqt_get_a_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_b_offset = tqt_get_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    return funcs;
}

// ============================================================================
// PackedQuantizedWeight: gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv
// ============================================================================

static PackedQuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv()
{
    PackedQuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    return funcs;
}

// ============================================================================
// Registration
// ============================================================================

void register_gemm_f16_i4_rvv_benchmarks()
{
    // D = fp16 variants
    auto funcs_f16 = make_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv();
    register_quantized_weight_gemm_benchmark<float16_t, float16_t>(
        "gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv", funcs_f16);

    auto packed_funcs_f16 = make_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv();
    register_packed_quantized_weight_gemm_benchmark<float16_t, float16_t>(
        "gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv", packed_funcs_f16);

    // D = fp32 variants (A still fp16, scale still fp16)
    auto funcs_f32 = make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv();
    register_quantized_weight_gemm_benchmark<float16_t, float>(
        "gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv", funcs_f32);

    auto packed_funcs_f32 = make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv();
    register_packed_quantized_weight_gemm_benchmark<float16_t, float>(
        "gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv", packed_funcs_f32);
}

}  // namespace benchmark
}  // namespace torq_tile
