//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_f16_f16/tqt_kernel_gemm_f16_f16_2x8vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float16_t);
static const size_t m_step = 2;
static const size_t n_step = 8 * vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t m_idx,
                                                                          size_t k_idx, size_t lda,
                                                                          size_t actual_m)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(float16_t);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t n_idx,
                                                                          size_t k_idx, size_t ldb,
                                                                          size_t actual_n)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx * TQT_MIN(actual_n, n_step)) * sizeof(float16_t);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float16_t);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float16_t);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float16_t);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float16_t);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t m, size_t k)
{
    return m * k * sizeof(float16_t);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t n, size_t k)
{
    return n * k * sizeof(float16_t);
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t m, size_t k, size_t lda,
                                                               size_t lda_packed, size_t k_idx,
                                                               const void *A, void *A_packed)
{
    // A packed format: [m/MR, k, MR] (interleaved by actual_m within each m-block)
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t actual_m = TQT_MIN(m - m_idx, m_step);
        const float16_t *a_ptr = (const float16_t *)A + m_idx * lda;
        float16_t *a_packed_ptr;
        if (m_idx == 0 || k_idx == 0 || actual_m == m_step) {
            a_packed_ptr = (float16_t *)A_packed + m_idx * lda_packed;
        } else {
            a_packed_ptr = (float16_t *)A_packed + m_idx * lda_packed - k_idx * (m_step - actual_m);
        }

        switch (actual_m) {
            case 2:
                tqt_gemm_pack_mxk_e16_kernel<2>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 1:
                tqt_gemm_pack_mxk_e16_kernel_1row(k, a_ptr, a_packed_ptr);
                break;
            default:
                break;
        }
    }
}

void tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t n, size_t k, size_t ldb,
                                                               size_t ldb_packed, size_t k_idx,
                                                               const void *B, void *B_packed)
{
    const size_t bytes_per_elem = sizeof(float16_t);

    const uint8_t *B_u8 = (const uint8_t *)B;
    uint8_t *B_packed_u8 = (uint8_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint8_t *b_ptr = B_u8 + n_idx * bytes_per_elem;
        uint8_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u8 + (n_idx * ldb_packed) * bytes_per_elem;
        } else {
            b_packed_ptr =
                B_packed_u8 + (n_idx * ldb_packed - k_idx * (n_step - actual_n)) * bytes_per_elem;
        }

        size_t remain = actual_n;
        const uint8_t *src_local = b_ptr;
        uint8_t *dst_local = b_packed_ptr;
        while (remain > 0) {
            size_t vl_this = TQT_MIN(remain, vlmax);
            tqt_gemm_pack_kxn_kernel<1>(actual_n, k, src_local, ldb, dst_local, vl_this,
                                        bytes_per_elem);
            src_local += vl_this * bytes_per_elem;
            dst_local += vl_this * bytes_per_elem;
            remain -= vl_this;
        }
    }
}

void tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t n, size_t k, size_t ldb,
                                                                size_t ldb_packed, size_t k_idx,
                                                                const void *B, void *B_packed)
{
    const uint16_t *B_u16 = (const uint16_t *)B;
    uint16_t *B_packed_u16 = (uint16_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint16_t *b_ptr = B_u16 + n_idx * ldb;
        uint16_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u16 + n_idx * ldb_packed;
        } else {
            b_packed_ptr = B_packed_u16 + n_idx * ldb_packed - k_idx * (n_step - actual_n);
        }

        size_t remain = actual_n;
        const uint16_t *src_local = b_ptr;
        uint16_t *dst_local = b_packed_ptr;
        while (remain > 0) {
            size_t vl_this = TQT_MIN(remain, vlmax);
            tqt_gemm_pack_nxk_e16_kernel<1>(actual_n, k, src_local, ldb, dst_local, vl_this);
            src_local += vl_this * ldb;
            dst_local += vl_this;
            remain -= vl_this;
        }
    }
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

// Template-based GEMM kernel dispatcher for 2x8vl (LMUL=8) with packed A/B.
// Packed B layout per n-block: [k, vl] contiguous, where vl = block width.
// Packed A layout per m-block: [k, MR] interleaved.
template <size_t MR>
static void gemm_kernel_8vl(size_t K, float16_t *D, const float16_t *A, const float16_t *B,
                            const float16_t *C, const float16_t *bias, size_t ldd, size_t lda,
                            size_t ldb, size_t ldc, float16_t clamp_min, float16_t clamp_max,
                            size_t vl)
{
    static_assert(MR >= 1 && MR <= 2, "MR must be in range [1, 2]");

    TQT_UNUSED(lda);
    TQT_UNUSED(ldb);

    const size_t a_stride_row = 1;
    const float16_t *a_k_ptr = A;
    const float16_t *b_k_ptr = B;

    if (C) {
        tqt_init_c_kernel_gemm_f16_f16_2x8vl_rvv<MR>(vl, C, ldc);
    } else {
        tqt_init_zero_kernel_gemm_f16_f16_2x8vl_rvv<MR>(vl);
    }

    tqt_load_a_kernel_gemm_f16_f16_2x8vl_rvv<MR>(a_k_ptr, a_stride_row);
    tqt_load_b_kernel_gemm_f16_f16_2x8vl_rvv(vl, b_k_ptr);
    a_k_ptr += MR;
    b_k_ptr += vl;

    for (size_t k_idx = 1; k_idx < K; k_idx++) {
        tqt_fused_load_vfmacc_axb_kernel_gemm_f16_f16_2x8vl_rvv<MR>(vl, a_k_ptr, a_stride_row,
                                                                    b_k_ptr);
        a_k_ptr += MR;
        b_k_ptr += vl;
    }

    tqt_epilogue_vfmacc_kernel_gemm_f16_f16_2x8vl_rvv<MR>();

    if (bias) {
        tqt_add_1xnbias_kernel_gemm_f16_f16_2x8vl_rvv<MR>(vl, bias);
    }
    tqt_clamp_kernel_gemm_f16_f16_2x8vl_rvv<MR>(vl, clamp_min, clamp_max);
    tqt_store_kernel_gemm_f16_f16_2x8vl_rvv<MR>(vl, D, ldd);
}

using gemm_kernel_fn_t = void (*)(size_t K, float16_t *D, const float16_t *A, const float16_t *B,
                                  const float16_t *C, const float16_t *bias, size_t ldd, size_t lda,
                                  size_t ldb, size_t ldc, float16_t clamp_min, float16_t clamp_max,
                                  size_t vl);

static constexpr gemm_kernel_fn_t kernel_table_8vl[2] = {
    gemm_kernel_8vl<1>,
    gemm_kernel_8vl<2>,
};

static void gemm_mrxnr_8vl_rvv(size_t M, size_t N, size_t K, float16_t *D, const float16_t *A,
                               const float16_t *B, const float16_t *C, const float16_t *bias,
                               size_t ldd, size_t lda, size_t ldb, size_t ldc, float16_t clamp_min,
                               float16_t clamp_max, size_t vl)
{
    TQT_UNUSED(N);
    TQT_ASSUME(M > 0 && M <= 2);
    TQT_ASSUME(vl > 0 && vl <= n_step);
    size_t mr_idx = M - 1;
    kernel_table_8vl[mr_idx](K, D, A, B, C, bias, ldd, lda, ldb, ldc, clamp_min, clamp_max, vl);
}

// ============================================================================
// Gemm Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_f16_f16p_f16p_2x8vl_rvv(size_t m, size_t n, size_t k, const void *A,
                                                        size_t lda, size_t k_idx_a, const void *B,
                                                        size_t ldb, size_t k_idx_b, const void *C,
                                                        size_t ldc, void *D, size_t ldd,
                                                        const void *bias, float16_t clamp_min,
                                                        float16_t clamp_max)
{
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const float16_t *a_ptr;
            if (actual_m == m_step || k_idx_a == 0) {
                a_ptr = (const float16_t *)A + m_idx * lda;
            } else {
                a_ptr = (const float16_t *)A + m_idx * lda - k_idx_a * (m_step - actual_m);
            }
            const float16_t *b_ptr;
            if (actual_n == n_step || k_idx_b == 0) {
                b_ptr = (const float16_t *)B + n_idx * ldb;
            } else {
                b_ptr = (const float16_t *)B + n_idx * ldb - k_idx_b * (n_step - actual_n);
            }
            const float16_t *c_ptr = C ? (const float16_t *)C + m_idx * ldc + n_idx : NULL;
            const float16_t *bias_ptr = bias ? (const float16_t *)bias + n_idx : NULL;
            float16_t *d_ptr = (float16_t *)D + m_idx * ldd + n_idx;

            gemm_mrxnr_8vl_rvv(actual_m, actual_n, k, d_ptr, a_ptr, b_ptr, c_ptr, bias_ptr, ldd,
                               lda, ldb, ldc, clamp_min, clamp_max, actual_n);
        }
    }
}
