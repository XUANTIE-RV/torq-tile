//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_f32_f16/gemm_1xnbias_clamp_f32_f32_f16t/tqt_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv.h"
#include "src/gemm/gemm_f32_f16/gemm_1xnbias_clamp_f32_f32p_f16p/tqt_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

static NonPackedGemmFuncs<float> make_direct_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv,
    };
}

static PackedGemmFuncs<float> make_packed_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
    };
}

void register_gemm_f32_f16_rvv_benchmarks()
{
    register_nonpacked_gemm_benchmark<float, float, BLayout::kTransposed, float, float16_t>(
        "tqt_gemm_1xnbias_clamp_f32_f32_f16t_8x1vl_rvv", make_direct_funcs());
    register_packed_gemm_benchmark<float, float, float, float16_t>(
        "tqt_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv/kernel_only", make_packed_funcs());
    register_gemm_a_pack_benchmark<float>(
        "tqt_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv/a_pack_only", make_packed_funcs());
    register_gemm_a_pack_and_gemm_benchmark<float, float, float, float16_t>(
        "tqt_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv/a_pack_and_gemm", make_packed_funcs());
    register_gemm_b_pack_benchmark<float16_t>(
        "tqt_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv/b_pack_only", make_packed_funcs());
}

}  // namespace benchmark
}  // namespace torq_tile
