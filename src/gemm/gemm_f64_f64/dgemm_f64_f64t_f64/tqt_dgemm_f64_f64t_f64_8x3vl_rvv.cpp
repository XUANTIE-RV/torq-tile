//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_dgemm_f64_f64t_f64_8x3vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_f64_f64/tqt_kernel_dgemm_f64_f64_8x3vl_rvv.h"

static inline size_t get_vlenb()
{
    static const size_t value = csrr_vlenb();
    return value;
}

static inline size_t get_vlmax(size_t vlenb)
{
    return vlenb / sizeof(double);
}

static inline size_t get_n_step(size_t vlenb)
{
    return 3 * get_vlmax(vlenb);
}

static const size_t m_step = 8;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_dgemm_f64_f64t_f64_8x3vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_dgemm_f64_f64t_f64_8x3vl_rvv(void)
{
    const size_t VLENB = get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    return n_step;
}

size_t tqt_get_a_offset_dgemm_f64_f64t_f64_8x3vl_rvv(size_t m_idx, size_t k_idx, size_t lda)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (k_idx * lda + m_idx) * sizeof(double);
}

size_t tqt_get_b_offset_dgemm_f64_f64t_f64_8x3vl_rvv(size_t n_idx, size_t k_idx, size_t ldb)
{
    const size_t VLENB = get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    TQT_ASSUME(n_idx % n_step == 0);
    return (k_idx * ldb + n_idx) * sizeof(double);
}

size_t tqt_get_c_offset_dgemm_f64_f64t_f64_8x3vl_rvv(size_t m_idx, size_t n_idx, size_t ldc)
{
    const size_t VLENB = get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(double);
}

size_t tqt_get_d_offset_dgemm_f64_f64t_f64_8x3vl_rvv(size_t m_idx, size_t n_idx, size_t ldd)
{
    const size_t VLENB = get_vlenb();
    const size_t n_step = get_n_step(VLENB);

    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(double);
}

size_t tqt_get_d_size_dgemm_f64_f64t_f64_8x3vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(double);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

// Template-based DGEMM kernel dispatcher
// Computes D = alpha * A^T * B + beta * C
// A is transposed [k, m], B is row-major [k, n]
template <size_t MR, size_t NR>
static void dgemm_kernel(size_t K, double *D, const double *A, const double *B, const double *C,
                         double alpha, double beta, size_t ldd, size_t lda, size_t ldb, size_t ldc,
                         size_t vl, size_t VLENB)
{
    const size_t vlmax = get_vlmax(VLENB);

    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    static_assert(NR >= 1 && NR <= 3, "NR must be in range [1, 3]");

    // A is stored as [k, m] row-major: consecutive m elements are contiguous
    size_t a_stride_row = 1;
    size_t b_stride_col = vlmax;
    size_t d_stride_row = ldd;
    size_t d_stride_col = vlmax;

    const double *a_ptr = A;
    const double *b_ptr = B;

    tqt_init_zero_kernel_dgemm_f64_f64_8x3vl_rvv<MR, NR>(vl);

    // Prologue: load first iteration's A and B data into ft0-ft7 / v0-v2
    tqt_load_a_kernel_dgemm_f64_f64_8x3vl_rvv<MR>(a_ptr, a_stride_row);
    tqt_load_b_kernel_dgemm_f64_f64_8x3vl_rvv<NR>(vl, b_ptr, b_stride_col);
    a_ptr += lda;
    b_ptr += ldb;

    // Steady-state K-loop: compute current iteration (column-first) while
    // prefetching next iteration's A and B data via software pipelining
    for (size_t k_idx = 1; k_idx < K; k_idx++) {
        tqt_fused_load_vfmacc_axb_kernel_dgemm_f64_f64_8x3vl_rvv<MR, NR>(a_ptr, a_stride_row, b_ptr,
                                                                         b_stride_col);
        a_ptr += lda;
        b_ptr += ldb;
    }

    // Epilogue: last iteration's pure vfmacc (no prefetch needed)
    tqt_epilogue_vfmacc_kernel_dgemm_f64_f64_8x3vl_rvv<MR, NR>();

    // Store: fuse alpha/beta/C into the store path so C is read before D is written
    const bool needs_postprocess = (alpha != 1.0 || beta != 0.0);
    if (needs_postprocess) {
        tqt_store_alpha_beta_kernel_dgemm_f64_f64_8x3vl_rvv<MR, NR>(
            vl, D, C, alpha, beta, d_stride_row, d_stride_col, ldc);
    } else {
        tqt_store_kernel_dgemm_f64_f64_8x3vl_rvv<MR, NR>(vl, D, d_stride_row, d_stride_col);
    }
}

// Function pointer type for kernel dispatch
using dgemm_kernel_fn_t = void (*)(size_t K, double *D, const double *A, const double *B,
                                   const double *C, double alpha, double beta, size_t ldd,
                                   size_t lda, size_t ldb, size_t ldc, size_t vl, size_t VLENB);

// Kernel dispatch table (8 x 3 = 24 kernels)
static constexpr dgemm_kernel_fn_t kernel_table[8][3] = {
    // MR = 1
    {dgemm_kernel<1, 1>, dgemm_kernel<1, 2>, dgemm_kernel<1, 3>},
    // MR = 2
    {dgemm_kernel<2, 1>, dgemm_kernel<2, 2>, dgemm_kernel<2, 3>},
    // MR = 3
    {dgemm_kernel<3, 1>, dgemm_kernel<3, 2>, dgemm_kernel<3, 3>},
    // MR = 4
    {dgemm_kernel<4, 1>, dgemm_kernel<4, 2>, dgemm_kernel<4, 3>},
    // MR = 5
    {dgemm_kernel<5, 1>, dgemm_kernel<5, 2>, dgemm_kernel<5, 3>},
    // MR = 6
    {dgemm_kernel<6, 1>, dgemm_kernel<6, 2>, dgemm_kernel<6, 3>},
    // MR = 7
    {dgemm_kernel<7, 1>, dgemm_kernel<7, 2>, dgemm_kernel<7, 3>},
    // MR = 8
    {dgemm_kernel<8, 1>, dgemm_kernel<8, 2>, dgemm_kernel<8, 3>},
};

static void dgemm_mrxnr_rvv(size_t M, size_t N, size_t K, double *D, const double *A,
                            const double *B, const double *C, double alpha, double beta, size_t ldd,
                            size_t lda, size_t ldb, size_t ldc, size_t vl, size_t VLENB)
{
    const size_t vlmax = get_vlmax(VLENB);

    TQT_ASSUME(M > 0 && M <= 8);
    TQT_ASSUME(N == 3 * vlmax || N == 2 * vlmax || (N > 0 && N <= vlmax));
    TQT_ASSUME(vl <= vlmax);

    size_t mr_idx = M - 1;                        // 0-7
    size_t nr_idx = (N + vlmax - 1) / vlmax - 1;  // 0-2

    // Dispatch to specialized kernel
    kernel_table[mr_idx][nr_idx](K, D, A, B, C, alpha, beta, ldd, lda, ldb, ldc, vl, VLENB);
}

// ============================================================================
// DGEMM Main Function
// ============================================================================

void tqt_run_dgemm_f64_f64t_f64_8x3vl_rvv(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                          size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                          const void *C, size_t ldc, void *D, size_t ldd,
                                          double alpha, double beta)
{
    const size_t VLENB = get_vlenb();
    const size_t vlmax = get_vlmax(VLENB);
    const size_t n_step = get_n_step(VLENB);

    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            // A is transposed [k, m]: entry at column m_idx
            const double *a_ptr = (const double *)A + m_idx;
            // B is row-major [k, n]: entry at column n_idx
            const double *b_ptr = (const double *)B + n_idx;
            const double *c_ptr = C ? (const double *)C + m_idx * ldc + n_idx : nullptr;
            double *d_ptr = (double *)D + m_idx * ldd + n_idx;

            size_t size_n = 0;
            if (actual_n >= vlmax) {
                if (actual_n == 3 * vlmax) {
                    size_n = 3 * vlmax;
                } else if (actual_n >= 2 * vlmax) {
                    size_n = 2 * vlmax;
                } else {
                    size_n = vlmax;
                }
                dgemm_mrxnr_rvv(actual_m, size_n, k, d_ptr, a_ptr, b_ptr, c_ptr, alpha, beta, ldd,
                                lda, ldb, ldc, vlmax, VLENB);
            }
            size_t remain_n = actual_n - size_n;
            if (remain_n > 0) {
                b_ptr += size_n;
                if (C)
                    c_ptr += size_n;
                d_ptr += size_n;
                dgemm_mrxnr_rvv(actual_m, remain_n, k, d_ptr, a_ptr, b_ptr, c_ptr, alpha, beta, ldd,
                                lda, ldb, ldc, remain_n, VLENB);
            }
        }
    }
}
