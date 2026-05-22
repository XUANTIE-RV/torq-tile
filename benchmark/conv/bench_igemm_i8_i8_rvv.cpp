//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/quantized_igemm_benchmark_runner.h"
#include "src/conv/igemm_i8_i8/igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv.h"
#include "src/conv/igemm_i8_i8/igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8/tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv.h"
#include "src/conv/igemm_i8_i8/igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp/tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv.h"
#include "src/conv/igemm_i8_i8/igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Non-packed igemmcf: igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv
// ============================================================================

static NonPackedRequantizedIgemmcfBenchFuncs make_nonpacked_igemmcf_i8_1vlx4_rvv()
{
    return {
        tqt_run_igemm_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv,
    };
}

// ============================================================================
// Packed igemmcf: igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv
// ============================================================================

static PackedRequantizedIgemmcfBenchFuncs make_packed_igemmcf_i8_1vlx4_rvv()
{
    return {
        tqt_get_a_packed_size_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv,
        tqt_run_a_pack_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv,
        tqt_run_igemm_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv,
    };
}

// ============================================================================
// Non-packed igemmcl: igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv
// ============================================================================

static NonPackedRequantizedIgemmclBenchFuncs make_nonpacked_igemmcl_i8_4x1vl_rvv()
{
    return {
        tqt_run_igemm_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
    };
}

// ============================================================================
// Packed igemmcl: igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv
// ============================================================================

static PackedRequantizedIgemmclBenchFuncs make_packed_igemmcl_i8_4x1vl_rvv()
{
    return {
        tqt_get_b_packed_size_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_run_igemm_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_igemm_i8_i8_rvv_benchmarks()
{
    register_nonpacked_requantized_igemmcf_benchmark(
        "tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv",
        make_nonpacked_igemmcf_i8_1vlx4_rvv());

    register_packed_requantized_igemmcf_benchmark(
        "tqt_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8_1vlx4_rvv",
        make_packed_igemmcf_i8_1vlx4_rvv());

    register_nonpacked_requantized_igemmcl_benchmark(
        "tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv",
        make_nonpacked_igemmcl_i8_4x1vl_rvv());

    register_packed_requantized_igemmcl_benchmark(
        "tqt_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv",
        make_packed_igemmcl_i8_4x1vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
