//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.h"

#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16p/tqt_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.h"

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(void)
{
    return tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv();
}

size_t tqt_get_n_step_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(void)
{
    return tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv();
}

size_t tqt_get_a_im2col_offset_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t m_idx,
                                                                           size_t k_idx,
                                                                           size_t lda_im2col)
{
    return (m_idx * lda_im2col + k_idx) * sizeof(float16_t);
}

size_t tqt_get_b_packed_offset_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t n_idx,
                                                                           size_t k_idx, size_t ldb,
                                                                           size_t actual_n)
{
    return tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(n_idx, k_idx, ldb,
                                                                             actual_n);
}

size_t tqt_get_c_offset_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldc)
{
    return tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(m_idx, n_idx, ldc);
}

size_t tqt_get_bias_offset_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t n_idx)
{
    return tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(n_idx);
}

size_t tqt_get_d_offset_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldd)
{
    return tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(m_idx, n_idx, ldd);
}

size_t tqt_get_d_size_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t m, size_t n)
{
    return tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(m, n);
}

size_t tqt_get_a_im2col_size_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t m, size_t k)
{
    return m * k * sizeof(float16_t);
}

size_t tqt_get_b_packed_size_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t n, size_t k)
{
    return tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(n, k);
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_bt_pack_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t n, size_t k, size_t ldb,
                                                                 size_t ldb_packed, size_t k_idx,
                                                                 const void *B, void *B_packed)
{
    tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(n, k, ldb, ldb_packed, k_idx, B,
                                                              B_packed);
}

// ============================================================================
// im2col for A (activation), no packing
// ============================================================================

void tqt_run_im2col_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(size_t group_idx,
                                                                const void *activation,
                                                                void *im2col_buf,
                                                                const tqt_conv_params *params)
{
    tqt_im2col_cl<float16_t>(group_idx, activation, im2col_buf, params);
}

// ============================================================================
// Conv Main Function
// ============================================================================

void tqt_run_conv_convcl_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B_packed,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float16_t clamp_min, float16_t clamp_max)
{
    tqt_run_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv(m, n, k, A, lda, k_idx_a, B_packed, ldb,
                                                      k_idx_b, C, ldc, D, ldd, bias, clamp_min,
                                                      clamp_max);
}
