//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_f16_i4/tqt_kernel_gemm_f16_i4_8x2vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float16_t);
static const size_t m_step = 8;
static const size_t n_step = 2 * vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t m_idx, size_t k_idx,
                                                                     size_t lda, size_t bl)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(bl >= 2 && bl % 2 == 0);
    TQT_ASSUME(k_idx % bl == 0);
    return (m_idx * lda + k_idx) * sizeof(float16_t);
}

size_t tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t n_idx, size_t k_idx,
                                                                     size_t ldb, size_t bl)
{
    TQT_ASSUME(n_idx % n_step == 0);
    TQT_ASSUME(bl >= 2 && bl % 2 == 0);
    TQT_ASSUME(k_idx % bl == 0);
    return n_idx * (ldb / 2) + k_idx / 2;
}

size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t n_idx,
                                                                           size_t b_idx,
                                                                           size_t ldsb)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldsb + b_idx) * sizeof(float16_t);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float16_t);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float16_t);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float16_t);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float16_t);
}

// ============================================================================
// Template GEMM kernel
// ============================================================================

template <size_t MR, size_t NR>
static void gemm_kernel(size_t K, size_t bl, float16_t *D, const float16_t *A, const uint8_t *B,
                        const float16_t *scale_b, const float16_t *C, const float16_t *bias,
                        size_t ldd, size_t lda, size_t ldb, size_t ldsb, size_t ldc,
                        float16_t clamp_min, float16_t clamp_max, size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 2, "NR must be in range [1, 2]");

    size_t a_stride_row = lda;
    size_t b_byte_stride = ldb / 2;
    size_t scale_byte_stride = ldsb * sizeof(float16_t);
    size_t c_stride_row = ldc;
    size_t d_stride_row = ldd;
    size_t d_stride_col = vlmax;

    if (C) {
        tqt_init_c_kernel_gemm_f16_i4_8x2vl_rvv<MR, NR>(vl, C, c_stride_row, vlmax);
    } else {
        tqt_init_zero_kernel_gemm_f16_i4_8x2vl_rvv<MR, NR>(vl);
    }

    size_t num_blocks = K / bl;
    for (size_t blk = 0; blk < num_blocks; blk++) {
        const float16_t *scale_col0 = scale_b + 0 * vl * ldsb + blk;
        const float16_t *scale_col1 = (NR >= 2) ? scale_b + 1 * vl * ldsb + blk : nullptr;
        tqt_load_scale_kernel_gemm_f16_i4_8x2vl_rvv<NR>(vl, scale_col0, scale_col1,
                                                        scale_byte_stride);

        size_t k_start = blk * bl;
        size_t k_end = k_start + bl;
        for (size_t ki = k_start; ki < k_end; ki += 2) {
            const uint8_t *b_col0 = B + 0 * vl * b_byte_stride + ki / 2;
            const uint8_t *b_col1 = (NR >= 2) ? B + 1 * vl * b_byte_stride + ki / 2 : nullptr;
            tqt_dequant_bt_kernel_gemm_f16_i4_8x2vl_rvv<NR>(vl, b_col0, b_col1, b_byte_stride);

            // FMA for ki+0 (lo nibble)
            tqt_load_a_kernel_gemm_f16_i4_8x2vl_rvv<MR>(A + ki, a_stride_row);
            tqt_vfmacc_lo_kernel_gemm_f16_i4_8x2vl_rvv<MR, NR>();

            // FMA for ki+1 (hi nibble)
            tqt_load_a_kernel_gemm_f16_i4_8x2vl_rvv<MR>(A + ki + 1, a_stride_row);
            tqt_vfmacc_hi_kernel_gemm_f16_i4_8x2vl_rvv<MR, NR>();
        }
    }

    if (bias) {
        tqt_add_1xnbias_kernel_gemm_f16_i4_8x2vl_rvv<MR, NR>(vl, bias);
    }
    tqt_clamp_kernel_gemm_f16_i4_8x2vl_rvv<MR, NR>(vl, clamp_min, clamp_max);
    tqt_store_kernel_gemm_f16_i4_8x2vl_rvv<MR, NR>(vl, D, d_stride_row, d_stride_col);
}

// ============================================================================
// Kernel dispatch
// ============================================================================

using gemm_kernel_fn_t = void (*)(size_t K, size_t bl, float16_t *D, const float16_t *A,
                                  const uint8_t *B, const float16_t *scale_b, const float16_t *C,
                                  const float16_t *bias, size_t ldd, size_t lda, size_t ldb,
                                  size_t ldsb, size_t ldc, float16_t clamp_min, float16_t clamp_max,
                                  size_t vl);

static constexpr gemm_kernel_fn_t kernel_table[8][2] = {
    {gemm_kernel<1, 1>, gemm_kernel<1, 2>}, {gemm_kernel<2, 1>, gemm_kernel<2, 2>},
    {gemm_kernel<3, 1>, gemm_kernel<3, 2>}, {gemm_kernel<4, 1>, gemm_kernel<4, 2>},
    {gemm_kernel<5, 1>, gemm_kernel<5, 2>}, {gemm_kernel<6, 1>, gemm_kernel<6, 2>},
    {gemm_kernel<7, 1>, gemm_kernel<7, 2>}, {gemm_kernel<8, 1>, gemm_kernel<8, 2>},
};

static void gemm_mrxnr_rvv(size_t M, size_t N, size_t K, size_t bl, float16_t *D,
                           const float16_t *A, const uint8_t *B, const float16_t *scale_b,
                           const float16_t *C, const float16_t *bias, size_t ldd, size_t lda,
                           size_t ldb, size_t ldsb, size_t ldc, float16_t clamp_min,
                           float16_t clamp_max, size_t vl)
{
    TQT_ASSUME(M > 0 && M <= 8);
    TQT_ASSUME(N == 2 * vlmax || (N > 0 && N <= vlmax));
    TQT_ASSUME(vl <= vlmax);

    size_t mr_idx = M - 1;
    size_t nr_idx = (N + vlmax - 1) / vlmax - 1;

    kernel_table[mr_idx][nr_idx](K, bl, D, A, B, scale_b, C, bias, ldd, lda, ldb, ldsb, ldc,
                                 clamp_min, clamp_max, vl);
}

// ============================================================================
// Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *scale_b, size_t ldsb, const void *C,
    size_t ldc, void *D, size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);

    const size_t b_byte_stride = ldb / 2;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const float16_t *a_ptr = (const float16_t *)A + m_idx * lda;
            const uint8_t *b_ptr = (const uint8_t *)B + n_idx * b_byte_stride;
            const float16_t *scale_ptr = (const float16_t *)scale_b + n_idx * ldsb;
            const float16_t *c_ptr = C ? (const float16_t *)C + m_idx * ldc + n_idx : NULL;
            const float16_t *bias_ptr = bias ? (const float16_t *)bias + n_idx : NULL;
            float16_t *d_ptr = (float16_t *)D + m_idx * ldd + n_idx;

            size_t size_n = 0;
            if (actual_n >= vlmax) {
                if (actual_n >= 2 * vlmax) {
                    size_n = 2 * vlmax;
                } else {
                    size_n = vlmax;
                }
                gemm_mrxnr_rvv(actual_m, size_n, k, bl, d_ptr, a_ptr, b_ptr, scale_ptr, c_ptr,
                               bias_ptr, ldd, lda, ldb, ldsb, ldc, clamp_min, clamp_max, vlmax);
            }
            size_t remain_n = actual_n - size_n;
            if (remain_n > 0) {
                b_ptr += size_n * b_byte_stride;
                scale_ptr += size_n * ldsb;
                if (C)
                    c_ptr += size_n;
                if (bias)
                    bias_ptr += size_n;
                d_ptr += size_n;
                gemm_mrxnr_rvv(actual_m, remain_n, k, bl, d_ptr, a_ptr, b_ptr, scale_ptr, c_ptr,
                               bias_ptr, ldd, lda, ldb, ldsb, ldc, clamp_min, clamp_max, remain_n);
            }
        }
    }
}
