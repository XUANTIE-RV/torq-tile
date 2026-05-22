//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/igemm_benchmark_runner.h"
#include "src/conv/igemm_f16_f16/igemmcf_mx1bias_clamp_f16_f16_f16/tqt_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv.h"
#include "src/conv/igemm_f16_f16/igemmcf_mx1bias_clamp_f16_f16p_f16/tqt_igemmcf_mx1bias_clamp_f16_f16p_f16_3vlx8_rvv.h"
#include "src/conv/igemm_f16_f16/igemmcl_1xnbias_clamp_f16_f16_f16p/tqt_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.h"
#include "src/conv/igemm_f16_f16/igemmcl_1xnbias_clamp_f16_f16_f16t/tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// NonPacked igemmcf: igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv
// ============================================================================

static NonPackedIgemmcfBenchFuncs<float16_t> make_nonpacked_igemmcf_f16_3vlx8_rvv()
{
    return {tqt_run_igemm_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv};
}

// ============================================================================
// Packed igemmcf: igemmcf_mx1bias_clamp_f16_f16p_f16_3vlx8_rvv
// ============================================================================

static PackedIgemmcfBenchFuncs<float16_t> make_packed_igemmcf_f16_3vlx8_rvv()
{
    return {
        tqt_get_a_packed_size_igemmcf_mx1bias_clamp_f16_f16p_f16_3vlx8_rvv,
        tqt_run_a_pack_igemmcf_mx1bias_clamp_f16_f16p_f16_3vlx8_rvv,
        tqt_run_igemm_igemmcf_mx1bias_clamp_f16_f16p_f16_3vlx8_rvv,
    };
}

// ============================================================================
// NonPacked igemmcl: igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv
// ============================================================================

static NonPackedIgemmclBenchFuncs<float16_t> make_nonpacked_igemmcl_f16_8x3vl_rvv()
{
    return {tqt_run_igemm_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv};
}

// ============================================================================
// Packed igemmcl: igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv
// ============================================================================

static PackedIgemmclBenchFuncs<float16_t> make_packed_igemmcl_f16_8x3vl_rvv()
{
    return {
        tqt_get_b_packed_size_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_run_bt_pack_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
        tqt_run_igemm_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_igemmcf_f16_f16_rvv_benchmarks()
{
    register_nonpacked_igemmcf_benchmark<float16_t, float16_t>(
        "tqt_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv", make_nonpacked_igemmcf_f16_3vlx8_rvv());
    register_packed_igemmcf_benchmark<float16_t, float16_t>(
        "tqt_igemmcf_mx1bias_clamp_f16_f16p_f16_3vlx8_rvv", make_packed_igemmcf_f16_3vlx8_rvv());
}

void register_igemmcl_f16_f16_rvv_benchmarks()
{
    register_nonpacked_igemmcl_benchmark<float16_t, float16_t>(
        "tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv", make_nonpacked_igemmcl_f16_8x3vl_rvv());
    register_packed_igemmcl_benchmark<float16_t, float16_t>(
        "tqt_igemmcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv", make_packed_igemmcl_f16_8x3vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
