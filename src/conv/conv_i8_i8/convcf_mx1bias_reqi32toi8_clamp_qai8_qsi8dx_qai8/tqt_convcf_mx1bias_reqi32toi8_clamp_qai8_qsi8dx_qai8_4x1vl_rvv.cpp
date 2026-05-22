//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.h"

#include "src/gemm/gemm_i8_i8/gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.h"

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(void)
{
    return tqt_get_m_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv();
}

size_t tqt_get_n_step_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(void)
{
    return tqt_get_n_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv();
}

size_t tqt_get_a_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(size_t m_idx,
                                                                                   size_t k_idx,
                                                                                   size_t lda)
{
    return tqt_get_a_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(m_idx, k_idx,
                                                                                     lda);
}

size_t tqt_get_c_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(size_t m_idx,
                                                                                   size_t n_idx,
                                                                                   size_t ldc)
{
    return tqt_get_c_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(m_idx, n_idx,
                                                                                     ldc);
}

size_t tqt_get_bias_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(size_t m_idx)
{
    return tqt_get_bias_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(m_idx);
}

size_t tqt_get_d_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(size_t m_idx,
                                                                                   size_t n_idx,
                                                                                   size_t ldd)
{
    return tqt_get_d_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(m_idx, n_idx,
                                                                                     ldd);
}

size_t tqt_get_d_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(size_t m, size_t n)
{
    return tqt_get_d_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(m, n);
}

size_t tqt_get_b_im2col_size_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(size_t k,
                                                                                        size_t n)
{
    return k * n * sizeof(int8_t);
}

size_t tqt_get_b_im2col_offset_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(
    size_t n_idx, size_t k_idx, size_t ldb_im2col)
{
    return (k_idx * ldb_im2col + n_idx) * sizeof(int8_t);
}

void tqt_run_im2col_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(
    size_t group_idx, const void *activation, void *im2col_buf, const tqt_conv_params *params)
{
    tqt_im2col_cf<int8_t>(group_idx, activation, im2col_buf, params);
}

// ============================================================================
// Conv Main Function
// ============================================================================

void tqt_run_conv_convcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    int8_t clamp_min, int8_t clamp_max, const struct tqt_qai8_qsi8dx_qai8_params *params)
{
    tqt_run_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(
        m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C, ldc, D, ldd, bias, clamp_min, clamp_max,
        params);
}
