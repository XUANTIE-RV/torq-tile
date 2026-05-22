//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zvfbfwma)
#error This file must be compiled for riscv, riscv_vector, zvfbfwma.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_bf16_bf16/tqt_kernel_gemm_bf16_bf16_8x1vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(bfloat16_t);
static const size_t m_step = 8;
static const size_t n_step = vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                     size_t lda)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx) * sizeof(bfloat16_t);
}

size_t tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                     size_t ldb)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx) * sizeof(bfloat16_t);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(bfloat16_t);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(bfloat16_t);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(bfloat16_t);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(bfloat16_t);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

template <size_t MR>
static void gemm_kernel(size_t K, bfloat16_t *D, const bfloat16_t *A, const bfloat16_t *B,
                        const bfloat16_t *C, const bfloat16_t *bias, size_t ldd, size_t lda,
                        size_t ldb, size_t ldc, bfloat16_t clamp_min, bfloat16_t clamp_max,
                        size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");

    size_t a_stride_row = lda;
    size_t b_stride_row = ldb;
    size_t d_stride_row = ldd;

    const bfloat16_t *a_ptr = A;
    const bfloat16_t *b_ptr = B;

    if (C) {
        tqt_init_c_kernel_gemm_bf16_bf16_8x1vl_rvv<MR>(vl, C, ldc, 0);
    } else {
        tqt_init_zero_kernel_gemm_bf16_bf16_8x1vl_rvv<MR>(vl);
    }

    for (size_t k_idx = 0; k_idx < K; k_idx++) {
        tqt_load_a_kernel_gemm_bf16_bf16_8x1vl_rvv<MR>(a_ptr, a_stride_row);
        tqt_load_bt_kernel_gemm_bf16_bf16_8x1vl_rvv(vl, b_ptr, b_stride_row);
        tqt_vfwmacc_kernel_gemm_bf16_bf16_8x1vl_rvv<MR>(vl);
        a_ptr += 1;
        b_ptr += 1;
    }

    if (bias) {
        tqt_add_1xnbias_kernel_gemm_bf16_bf16_8x1vl_rvv<MR>(vl, bias);
    }
    tqt_clamp_kernel_gemm_bf16_bf16_8x1vl_rvv<MR>(vl, clamp_min, clamp_max);

    tqt_cvt_fp32_to_bf16_kernel_gemm_bf16_bf16_8x1vl_rvv<MR>(vl);
    tqt_store_kernel_gemm_bf16_bf16_8x1vl_rvv<MR>(vl, D, d_stride_row);
}

using gemm_kernel_fn_t = void (*)(size_t K, bfloat16_t *D, const bfloat16_t *A, const bfloat16_t *B,
                                  const bfloat16_t *C, const bfloat16_t *bias, size_t ldd,
                                  size_t lda, size_t ldb, size_t ldc, bfloat16_t clamp_min,
                                  bfloat16_t clamp_max, size_t vl);

static constexpr gemm_kernel_fn_t kernel_table[8] = {
    gemm_kernel<1>, gemm_kernel<2>, gemm_kernel<3>, gemm_kernel<4>,
    gemm_kernel<5>, gemm_kernel<6>, gemm_kernel<7>, gemm_kernel<8>,
};

// ============================================================================
// Gemm Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16t_8x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    bfloat16_t clamp_min, bfloat16_t clamp_max)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const bfloat16_t *a_ptr = (const bfloat16_t *)A + m_idx * lda;
            const bfloat16_t *b_ptr = (const bfloat16_t *)B + n_idx * ldb;
            const bfloat16_t *c_ptr = C ? (const bfloat16_t *)C + m_idx * ldc + n_idx : NULL;
            const bfloat16_t *bias_ptr = bias ? (const bfloat16_t *)bias + n_idx : NULL;
            bfloat16_t *d_ptr = (bfloat16_t *)D + m_idx * ldd + n_idx;

            size_t mr_idx = actual_m - 1;
            kernel_table[mr_idx](k, d_ptr, a_ptr, b_ptr, c_ptr, bias_ptr, ldd, lda, ldb, ldc,
                                 clamp_min, clamp_max, actual_n);
        }
    }
}
