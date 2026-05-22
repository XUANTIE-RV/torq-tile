//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.h"

#include "../tqt_kernel_gemm_f32_f32_f32_rvv.h"
#include "src/common/tqt_common.h"
#include "src/common/tqt_pack_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t m_step = 8;
static const size_t n_step = 3 * vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m_idx,
                                                                          size_t k_idx, size_t lda,
                                                                          size_t actual_m)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(float);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t n_idx,
                                                                          size_t k_idx, size_t ldb,
                                                                          size_t actual_n)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx * TQT_MIN(actual_n, n_step)) * sizeof(float);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m, size_t k)
{
    return m * k * sizeof(float);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t n, size_t k)
{
    return n * k * sizeof(float);
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t m, size_t k, size_t lda,
                                                               size_t lda_packed, size_t k_idx,
                                                               const void *A, void *A_packed)
{
    tqt_run_pack_mxk_8x1_e32_rvv(m, k, lda, lda_packed, k_idx, A, A_packed);
}

void tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t n, size_t k, size_t ldb,
                                                               size_t ldb_packed, size_t k_idx,
                                                               const void *B, void *B_packed)
{
    tqt_run_pack_kxn_1x3vl_rvv(n, k, ldb, ldb_packed, k_idx, B, B_packed, 32);
}

void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(size_t n, size_t k, size_t ldb,
                                                                size_t ldb_packed, size_t k_idx,
                                                                const void *B, void *B_packed)
{
    tqt_run_pack_nxk_1x3vl_e32_rvv(n, k, ldb, ldb_packed, k_idx, B, B_packed);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

// Template-based GEMM kernel dispatcher
template <size_t MR, size_t NR>
static void gemm_kernel(size_t K, float *D, const float *A, const float *B, const float *C,
                        const float *bias, size_t ldd, size_t lda, size_t ldb, size_t ldc,
                        float clamp_min, float clamp_max, size_t vl)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    TQT_UNUSED(lda);
    TQT_UNUSED(ldb);

    size_t a_stride_row = 1;
    size_t b_stride_col = vlmax;
    size_t c_stride_row = ldc;
    size_t c_stride_col = vlmax;
    size_t d_stride_row = ldd;
    size_t d_stride_col = vlmax;

    const float *a_ptr = A;
    const float *b_ptr = B;

    if (C) {
        tqt_init_c_kernel_gemm_f32_f32_f32_rvv<MR, NR>(vl, C, c_stride_row, c_stride_col);
    } else {
        tqt_init_zero_kernel_gemm_f32_f32_f32_rvv<MR, NR>(vl);
    }

    for (size_t k_idx = 0; k_idx < K; k_idx++) {
        tqt_load_a_kernel_gemm_f32_f32_f32_rvv<MR>(a_ptr, a_stride_row);
        tqt_load_b_kernel_gemm_f32_f32_f32_rvv<NR>(vl, b_ptr, b_stride_col);
        tqt_vfmacc_kernel_gemm_f32_f32_f32_rvv<MR, NR>(vl);
        a_ptr += MR;
        b_ptr += NR * vl;
    }

    if (bias) {
        tqt_add_1xnbias_kernel_gemm_f32_f32_f32_rvv<MR, NR>(vl, bias);
    }
    tqt_clamp_kernel_gemm_f32_f32_f32_rvv<MR, NR>(vl, clamp_min, clamp_max);
    tqt_store_kernel_gemm_f32_f32_f32_rvv<MR, NR>(vl, D, d_stride_row, d_stride_col);
}

// Function pointer type for kernel dispatch
using gemm_kernel_fn_t = void (*)(size_t K, float *D, const float *A, const float *B,
                                  const float *C, const float *bias, size_t ldd, size_t lda,
                                  size_t ldb, size_t ldc, float clamp_min, float clamp_max,
                                  size_t vl);

// Kernel dispatch table (8 x 3 = 24 kernels)
static constexpr gemm_kernel_fn_t kernel_table[8][3] = {
    // MR = 1
    {gemm_kernel<1, 1>, gemm_kernel<1, 2>, gemm_kernel<1, 3>},
    // MR = 2
    {gemm_kernel<2, 1>, gemm_kernel<2, 2>, gemm_kernel<2, 3>},
    // MR = 3
    {gemm_kernel<3, 1>, gemm_kernel<3, 2>, gemm_kernel<3, 3>},
    // MR = 4
    {gemm_kernel<4, 1>, gemm_kernel<4, 2>, gemm_kernel<4, 3>},
    // MR = 5
    {gemm_kernel<5, 1>, gemm_kernel<5, 2>, gemm_kernel<5, 3>},
    // MR = 6
    {gemm_kernel<6, 1>, gemm_kernel<6, 2>, gemm_kernel<6, 3>},
    // MR = 7
    {gemm_kernel<7, 1>, gemm_kernel<7, 2>, gemm_kernel<7, 3>},
    // MR = 8
    {gemm_kernel<8, 1>, gemm_kernel<8, 2>, gemm_kernel<8, 3>},
};

static void gemm_mrxnr_rvv(size_t M, size_t N, size_t K, float *D, const float *A, const float *B,
                           const float *C, const float *bias, size_t ldd, size_t lda, size_t ldb,
                           size_t ldc, float clamp_min, float clamp_max, size_t vl)
{
    TQT_ASSUME(M > 0 && M <= 8);
    TQT_ASSUME(N == 3 * vlmax || N == 2 * vlmax || (N > 0 && N <= vlmax));
    TQT_ASSUME(vl <= vlmax);

    size_t mr_idx = M - 1;                        // 0-7
    size_t nr_idx = (N + vlmax - 1) / vlmax - 1;  // 0-2

    // Dispatch to specialized kernel
    kernel_table[mr_idx][nr_idx](K, D, A, B, C, bias, ldd, lda, ldb, ldc, clamp_min, clamp_max, vl);
}

// ============================================================================
// Gemm Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max)
{
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const float *a_ptr;
            const float *b_ptr;
            if (actual_m == m_step || k_idx_a == 0) {
                a_ptr = (const float *)A_packed + m_idx * lda;
            } else {
                a_ptr = (const float *)A_packed + m_idx * lda - k_idx_a * (m_step - actual_m);
            }
            if (actual_n == n_step || k_idx_b == 0) {
                b_ptr = (const float *)B_packed + n_idx * ldb;
            } else {
                b_ptr = (const float *)B_packed + n_idx * ldb - k_idx_b * (n_step - actual_n);
            }
            const float *c_ptr = C ? (const float *)C + m_idx * ldc + n_idx : NULL;
            const float *bias_ptr = bias ? (const float *)bias + n_idx : NULL;
            float *d_ptr = (float *)D + m_idx * ldd + n_idx;

            size_t size_n = 0;
            if (actual_n >= vlmax) {
                if (actual_n == 3 * vlmax) {
                    size_n = 3 * vlmax;
                } else if (actual_n >= 2 * vlmax) {
                    size_n = 2 * vlmax;
                } else {
                    size_n = vlmax;
                }
                gemm_mrxnr_rvv(actual_m, size_n, k, d_ptr, a_ptr, b_ptr, c_ptr, bias_ptr, ldd, lda,
                               ldb, ldc, clamp_min, clamp_max, vlmax);
            }
            size_t remain_n = actual_n - size_n;
            if (remain_n > 0) {
                b_ptr += size_n * ldb - k_idx_b * (remain_n);
                if (C)
                    c_ptr += size_n;
                if (bias)
                    bias_ptr += size_n;
                d_ptr += size_n;
                gemm_mrxnr_rvv(actual_m, remain_n, k, d_ptr, a_ptr, b_ptr, c_ptr, bias_ptr, ldd,
                               lda, ldb, ldc, clamp_min, clamp_max, remain_n);
            }
        }
    }
}
