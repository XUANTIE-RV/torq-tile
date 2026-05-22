//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.h"

#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32t/tqt_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv.h"

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(void)
{
    return tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv();
}

size_t tqt_get_n_step_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(void)
{
    return tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv();
}

size_t tqt_get_b_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t n_idx, size_t k_idx,
                                                                    size_t ldb)
{
    return tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(n_idx, k_idx, ldb);
}

size_t tqt_get_c_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldc)
{
    return tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(m_idx, n_idx, ldc);
}

size_t tqt_get_bias_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t n_idx)
{
    return tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(n_idx);
}

size_t tqt_get_d_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldd)
{
    return tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(m_idx, n_idx, ldd);
}

size_t tqt_get_d_size_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m, size_t n)
{
    return tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(m, n);
}

size_t tqt_get_a_im2col_size_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m, size_t k)
{
    return m * k * sizeof(float);
}

size_t tqt_get_a_im2col_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m_idx,
                                                                           size_t k_idx,
                                                                           size_t lda_im2col)
{
    return (m_idx * lda_im2col + k_idx) * sizeof(float);
}

void tqt_run_im2col_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t group_idx,
                                                                const void *activation,
                                                                void *im2col_buf,
                                                                const tqt_conv_params *params)
{
    tqt_im2col_cl<float>(group_idx, activation, im2col_buf, params);
}

// ============================================================================
// Conv Main Function
// ============================================================================

void tqt_run_conv_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float clamp_min, float clamp_max)
{
    tqt_run_gemm_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C,
                                                      ldc, D, ldd, bias, clamp_min, clamp_max);
}
