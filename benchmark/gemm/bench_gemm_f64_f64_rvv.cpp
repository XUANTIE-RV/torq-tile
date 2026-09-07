//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/sgemm_benchmark_runner.h"
#include "src/gemm/gemm_f64_f64/dgemm_f64_f64_f64/tqt_dgemm_f64_f64_f64_8x3vl_rvv.h"
#include "src/gemm/gemm_f64_f64/dgemm_f64_f64_f64t/tqt_dgemm_f64_f64_f64t_8x3vl_rvv.h"
#include "src/gemm/gemm_f64_f64/dgemm_f64_f64t_f64/tqt_dgemm_f64_f64t_f64_8x3vl_rvv.h"
#include "src/gemm/gemm_f64_f64/dgemm_f64_f64t_f64t/tqt_dgemm_f64_f64t_f64t_8x3vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// DGEMM A=RowMajor, B=RowMajor: dgemm_f64_f64_f64_8x3vl_rvv
// ============================================================================

static NonPackedSgemmFuncs<double> make_dgemm_f64_f64_f64_8x3vl_rvv()
{
    return {
        tqt_get_m_step_dgemm_f64_f64_f64_8x3vl_rvv,   tqt_get_n_step_dgemm_f64_f64_f64_8x3vl_rvv,
        tqt_get_a_offset_dgemm_f64_f64_f64_8x3vl_rvv, tqt_get_b_offset_dgemm_f64_f64_f64_8x3vl_rvv,
        tqt_get_d_offset_dgemm_f64_f64_f64_8x3vl_rvv, tqt_run_dgemm_f64_f64_f64_8x3vl_rvv,
    };
}

// ============================================================================
// DGEMM A=RowMajor, B=Transposed: dgemm_f64_f64_f64t_8x3vl_rvv
// ============================================================================

static NonPackedSgemmFuncs<double> make_dgemm_f64_f64_f64t_8x3vl_rvv()
{
    return {
        tqt_get_m_step_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_n_step_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_a_offset_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_b_offset_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_get_d_offset_dgemm_f64_f64_f64t_8x3vl_rvv,
        tqt_run_dgemm_f64_f64_f64t_8x3vl_rvv,
    };
}

// ============================================================================
// DGEMM A=Transposed, B=RowMajor: dgemm_f64_f64t_f64_8x3vl_rvv
// ============================================================================

static NonPackedSgemmFuncs<double> make_dgemm_f64_f64t_f64_8x3vl_rvv()
{
    return {
        tqt_get_m_step_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_n_step_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_a_offset_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_b_offset_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_get_d_offset_dgemm_f64_f64t_f64_8x3vl_rvv,
        tqt_run_dgemm_f64_f64t_f64_8x3vl_rvv,
    };
}

// ============================================================================
// DGEMM A=Transposed, B=Transposed: dgemm_f64_f64t_f64t_8x3vl_rvv
// ============================================================================

static NonPackedSgemmFuncs<double> make_dgemm_f64_f64t_f64t_8x3vl_rvv()
{
    return {
        tqt_get_m_step_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_n_step_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_a_offset_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_b_offset_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_get_d_offset_dgemm_f64_f64t_f64t_8x3vl_rvv,
        tqt_run_dgemm_f64_f64t_f64t_8x3vl_rvv,
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_gemm_f64_f64_rvv_benchmarks()
{
    register_nonpacked_sgemm_benchmark<double, ALayout::kRowMajor, BLayout::kRowMajor>(
        "tqt_dgemm_f64_f64_f64_8x3vl_rvv", make_dgemm_f64_f64_f64_8x3vl_rvv());

    register_nonpacked_sgemm_benchmark<double, ALayout::kRowMajor, BLayout::kTransposed>(
        "tqt_dgemm_f64_f64_f64t_8x3vl_rvv", make_dgemm_f64_f64_f64t_8x3vl_rvv());

    register_nonpacked_sgemm_benchmark<double, ALayout::kTransposed, BLayout::kRowMajor>(
        "tqt_dgemm_f64_f64t_f64_8x3vl_rvv", make_dgemm_f64_f64t_f64_8x3vl_rvv());

    register_nonpacked_sgemm_benchmark<double, ALayout::kTransposed, BLayout::kTransposed>(
        "tqt_dgemm_f64_f64t_f64t_8x3vl_rvv", make_dgemm_f64_f64t_f64t_8x3vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
