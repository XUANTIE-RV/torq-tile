//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_f32_f32/tqt_kernel_gemm_f32_f32_4x4vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t m_step = 4;
static const size_t n_step = 4 * vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(size_t m_idx, size_t k_idx,
                                                                  size_t lda)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx) * sizeof(float);
}

size_t tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(size_t n_idx, size_t k_idx,
                                                                  size_t ldb)
{
    TQT_ASSUME(n_idx % n_step == 0);
    // For transposed B [n, k]: element at (n_idx, k_idx) -> offset = (n_idx * ldb + k_idx)
    return (n_idx * ldb + k_idx) * sizeof(float);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(size_t m_idx, size_t n_idx,
                                                                  size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(size_t m_idx, size_t n_idx,
                                                                  size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

// Template-based GEMM kernel dispatcher for 4x4vl (LMUL=4) with transposed B.
// Handles arbitrary vl in [1, 4*vlmax] via RVV's variable-length vector semantics.
template <size_t MR>
static void gemm_kernel_4vl(size_t K, float *D, const float *A, const float *B, const float *C,
                            const float *bias, size_t ldd, size_t lda, size_t ldb, size_t ldc,
                            float clamp_min, float clamp_max, size_t vl)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    size_t a_stride_row = lda;
    const float *a_ptr = A;
    const float *b_ptr = B;

    if (C) {
        tqt_init_c_kernel_gemm_f32_f32_4x4vl_rvv<MR>(vl, C, ldc);
    } else {
        tqt_init_zero_kernel_gemm_f32_f32_4x4vl_rvv<MR>(vl);
    }

    // Prologue: load first iteration's A and B data
    // For transposed B, use strided load (vlse32)
    tqt_load_a_kernel_gemm_f32_f32_4x4vl_rvv<MR>(a_ptr, a_stride_row);
    tqt_load_bt_kernel_gemm_f32_f32_4x4vl_rvv(vl, b_ptr, ldb);
    a_ptr += 1;
    b_ptr += 1;  // For transposed B, advance by 1 element along k dimension

    // Steady-state K-loop with software pipelining (axbt variant)
    for (size_t k_idx = 1; k_idx < K; k_idx++) {
        tqt_fused_load_vfmacc_axbt_kernel_gemm_f32_f32_4x4vl_rvv<MR>(vl, a_ptr, a_stride_row, b_ptr,
                                                                     ldb);
        a_ptr += 1;
        b_ptr += 1;
    }

    // Epilogue: last iteration's pure vfmacc
    tqt_epilogue_vfmacc_kernel_gemm_f32_f32_4x4vl_rvv<MR>();

    if (bias) {
        tqt_add_1xnbias_kernel_gemm_f32_f32_4x4vl_rvv<MR>(vl, bias);
    }
    tqt_clamp_kernel_gemm_f32_f32_4x4vl_rvv<MR>(vl, clamp_min, clamp_max);
    tqt_store_kernel_gemm_f32_f32_4x4vl_rvv<MR>(vl, D, ldd);
}

// Function pointer type for kernel dispatch
using gemm_kernel_fn_t = void (*)(size_t K, float *D, const float *A, const float *B,
                                  const float *C, const float *bias, size_t ldd, size_t lda,
                                  size_t ldb, size_t ldc, float clamp_min, float clamp_max,
                                  size_t vl);

static constexpr gemm_kernel_fn_t kernel_table_4vl[4] = {
    gemm_kernel_4vl<1>,
    gemm_kernel_4vl<2>,
    gemm_kernel_4vl<3>,
    gemm_kernel_4vl<4>,
};

static void gemm_mrxnr_4vl_rvv(size_t M, size_t N, size_t K, float *D, const float *A,
                               const float *B, const float *C, const float *bias, size_t ldd,
                               size_t lda, size_t ldb, size_t ldc, float clamp_min, float clamp_max,
                               size_t vl)
{
    TQT_UNUSED(N);
    TQT_ASSUME(M > 0 && M <= 4);
    TQT_ASSUME(vl > 0 && vl <= n_step);
    size_t mr_idx = M - 1;
    kernel_table_4vl[mr_idx](K, D, A, B, C, bias, ldd, lda, ldb, ldc, clamp_min, clamp_max, vl);
}

// ============================================================================
// Gemm Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_f32_f32_f32t_4x4vl_rvv(size_t m, size_t n, size_t k, const void *A,
                                                       size_t lda, size_t k_idx_a, const void *B,
                                                       size_t ldb, size_t k_idx_b, const void *C,
                                                       size_t ldc, void *D, size_t ldd,
                                                       const void *bias, float clamp_min,
                                                       float clamp_max)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const float *a_ptr = (const float *)A + m_idx * lda;
            // For transposed B, start at row n_idx (B[n_idx, ...])
            const float *b_ptr = (const float *)B + n_idx * ldb;
            const float *c_ptr = C ? (const float *)C + m_idx * ldc + n_idx : NULL;
            const float *bias_ptr = bias ? (const float *)bias + n_idx : NULL;
            float *d_ptr = (float *)D + m_idx * ldd + n_idx;

            // Single LMUL=4 kernel call: vsetvli's runtime vl handles any actual_n
            // in [1, n_step] via RVV variable-length vector semantics. The strided
            // vlse32 load also respects vl, so transposed B works identically.
            gemm_mrxnr_4vl_rvv(actual_m, actual_n, k, d_ptr, a_ptr, b_ptr, c_ptr, bias_ptr, ldd,
                               lda, ldb, ldc, clamp_min, clamp_max, actual_n);
        }
    }
}
