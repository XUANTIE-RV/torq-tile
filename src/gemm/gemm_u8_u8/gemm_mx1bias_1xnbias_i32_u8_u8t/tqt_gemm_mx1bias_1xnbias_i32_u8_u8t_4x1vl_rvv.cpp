//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_u8_u8/tqt_kernel_gemm_u8_u8_4x1vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB;
static const size_t m_step = 4;
static const size_t n_step = vlmax;

size_t tqt_get_m_step_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                  size_t lda)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx) * sizeof(uint8_t);
}

size_t tqt_get_b_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                  size_t ldb)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx) * sizeof(uint8_t);
}

size_t tqt_get_c_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                  size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(int32_t);
}

size_t tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(size_t m_idx)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return m_idx * sizeof(int32_t);
}

size_t tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(int32_t);
}

size_t tqt_get_d_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                  size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(int32_t);
}

size_t tqt_get_d_size_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(int32_t);
}

template <size_t MR>
static void gemm_kernel(size_t K, int32_t *D, const uint8_t *A, const uint8_t *B, const int32_t *C,
                        const int32_t *bias_m, const int32_t *bias_n, size_t ldd, size_t lda,
                        size_t ldb, size_t ldc, size_t vl)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    size_t a_stride_row = lda;
    size_t b_stride_row = ldb;
    size_t d_stride_row = ldd;

    const uint8_t *a_ptr = A;
    const uint8_t *b_ptr = B;
    const uint8_t *a_ptrs[MR];

    if (C) {
        tqt_init_c_kernel_gemm_u8_u8_4x1vl_rvv<MR>(vl, C, ldc, 1);
    } else {
        tqt_init_zero_kernel_gemm_u8_u8_4x1vl_rvv<MR>(vl);
    }

    for (size_t k_idx = 0; k_idx < K; k_idx++) {
        tqt_get_a_ptr_gemm_u8_u8_4x1vl_rvv<MR>(a_ptr, a_stride_row, a_ptrs);
        tqt_load_bt_kernel_gemm_u8_u8_4x1vl_rvv(vl, b_ptr, b_stride_row);
        tqt_vwmaccu_kernel_gemm_u8_u8_4x1vl_rvv<MR>(vl, a_ptrs);
        a_ptr += 1;
        b_ptr += 1;
    }

    tqt_add_bias_mn_kernel_gemm_u8_u8_4x1vl_rvv<MR>(vl, bias_m, bias_n);
    tqt_store_i32_kernel_gemm_u8_u8_4x1vl_rvv<MR>(vl, D, d_stride_row);
}

using gemm_kernel_fn_t = void (*)(size_t K, int32_t *D, const uint8_t *A, const uint8_t *B,
                                  const int32_t *C, const int32_t *bias_m, const int32_t *bias_n,
                                  size_t ldd, size_t lda, size_t ldb, size_t ldc, size_t vl);

static constexpr gemm_kernel_fn_t kernel_table[4] = {
    gemm_kernel<1>,
    gemm_kernel<2>,
    gemm_kernel<3>,
    gemm_kernel<4>,
};

void tqt_run_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(size_t m, size_t n, size_t k, const void *A,
                                                       size_t lda, size_t k_idx_a, const void *B,
                                                       size_t ldb, size_t k_idx_b, const void *C,
                                                       size_t ldc, void *D, size_t ldd,
                                                       const void *bias_m, const void *bias_n)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);

    const uint8_t *A_ptr = (const uint8_t *)A;
    const uint8_t *B_ptr = (const uint8_t *)B;
    const int32_t *C_ptr = (const int32_t *)C;
    const int32_t *bias_m_ptr = (const int32_t *)bias_m;
    const int32_t *bias_n_ptr = (const int32_t *)bias_n;
    int32_t *D_ptr = (int32_t *)D;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const uint8_t *a_ptr = A_ptr + m_idx * lda;
            const uint8_t *b_ptr = B_ptr + n_idx * ldb;
            const int32_t *c_ptr = C_ptr ? C_ptr + m_idx * ldc + n_idx : nullptr;
            int32_t *d_ptr = D_ptr + m_idx * ldd + n_idx;
            const int32_t *bm_ptr = bias_m_ptr + m_idx;
            const int32_t *bn_ptr = bias_n_ptr + n_idx;

            size_t mr_idx = actual_m - 1;
            kernel_table[mr_idx](k, d_ptr, a_ptr, b_ptr, c_ptr, bm_ptr, bn_ptr, ldd, lda, ldb, ldc,
                                 actual_n);
        }
    }
}
