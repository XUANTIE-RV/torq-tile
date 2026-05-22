//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_f32_f32/tqt_kernel_gemm_f32_f32_4x4vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB / sizeof(float);
static const size_t m_step = 4;
static const size_t n_step = 4 * vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t m_idx,
                                                                          size_t k_idx, size_t lda,
                                                                          size_t actual_m)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(float);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t n_idx,
                                                                          size_t k_idx, size_t ldb,
                                                                          size_t actual_n)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx * TQT_MIN(actual_n, n_step)) * sizeof(float);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t m, size_t k)
{
    return m * k * sizeof(float);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t n, size_t k)
{
    return n * k * sizeof(float);
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t m, size_t k, size_t lda,
                                                               size_t lda_packed, size_t k_idx,
                                                               const void *A, void *A_packed)
{
    // A packed format: [m/MR, k, MR] (interleaved by actual_m within each m-block)
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t actual_m = TQT_MIN(m - m_idx, m_step);
        const float *a_ptr = (const float *)A + m_idx * lda;
        float *a_packed_ptr;
        if (m_idx == 0 || k_idx == 0 || actual_m == m_step) {
            a_packed_ptr = (float *)A_packed + m_idx * lda_packed;
        } else {
            a_packed_ptr = (float *)A_packed + m_idx * lda_packed - k_idx * (m_step - actual_m);
        }

        switch (actual_m) {
            case 4:
                tqt_gemm_pack_mxk_e32_kernel<4>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 3:
                tqt_gemm_pack_mxk_e32_kernel<3>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 2:
                tqt_gemm_pack_mxk_e32_kernel<2>(k, a_ptr, lda, a_packed_ptr);
                break;
            case 1:
                tqt_gemm_pack_mxk_e32_kernel_1row(k, a_ptr, a_packed_ptr);
                break;
            default:
                break;
        }
    }
}

void tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t n, size_t k, size_t ldb,
                                                               size_t ldb_packed, size_t k_idx,
                                                               const void *B, void *B_packed)
{
    // B packed format: each n-block laid out as a single contiguous [k, actual_n_in_block]
    // region (per-k row stride = actual_n_in_block). For trailing partial blocks,
    // the row stride shrinks accordingly so total packed size stays n*k*sizeof(float).
    const size_t bytes_per_elem = sizeof(float);

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

        // Pack actual_n elements per k row contiguously. The kxn helper kernel
        // operates at LMUL=1 (max vl = vlmax for float), so when actual_n exceeds
        // vlmax we issue multiple calls covering disjoint column ranges of the
        // same logical [k, actual_n] block. Passing n=actual_n makes the helper
        // use actual_n as its per-k destination row stride, so the column ranges
        // line up correctly.
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

void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t n, size_t k, size_t ldb,
                                                                size_t ldb_packed, size_t k_idx,
                                                                const void *B, void *B_packed)
{
    // Pack transposed B [n, k] into the same flat [k, actual_n_in_block] per-block
    // layout used by the row-major variant above. Inner loop slices by vlmax because
    // the nxk helper kernel is LMUL=1.

    const uint32_t *B_u32 = (const uint32_t *)B;
    uint32_t *B_packed_u32 = (uint32_t *)B_packed;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
        const size_t actual_n = TQT_MIN(n - n_idx, n_step);
        const uint32_t *b_ptr = B_u32 + n_idx * ldb;
        uint32_t *b_packed_ptr;
        if (n_idx == 0 || k_idx == 0 || actual_n == n_step) {
            b_packed_ptr = B_packed_u32 + n_idx * ldb_packed;
        } else {
            b_packed_ptr = B_packed_u32 + n_idx * ldb_packed - k_idx * (n_step - actual_n);
        }

        size_t remain = actual_n;
        const uint32_t *src_local = b_ptr;
        uint32_t *dst_local = b_packed_ptr;
        while (remain > 0) {
            size_t vl_this = TQT_MIN(remain, vlmax);
            tqt_gemm_pack_nxk_e32_kernel<1>(actual_n, k, src_local, ldb, dst_local, vl_this);
            src_local += vl_this * ldb;
            dst_local += vl_this;
            remain -= vl_this;
        }
    }
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

// Template-based GEMM kernel dispatcher for 4x4vl (LMUL=4) with packed A/B.
// Packed B layout per n-block: [k, vl] contiguous, where vl = block width.
// Packed A layout per m-block: [k, MR] interleaved.
template <size_t MR>
static void gemm_kernel_4vl(size_t K, float *D, const float *A, const float *B, const float *C,
                            const float *bias, size_t ldd, size_t lda, size_t ldb, size_t ldc,
                            float clamp_min, float clamp_max, size_t vl)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    TQT_UNUSED(lda);
    TQT_UNUSED(ldb);

    const size_t a_stride_row = 1;  // packed A: rows of one k are adjacent (interleave width MR)
    const float *a_k_ptr = A;
    const float *b_k_ptr = B;

    if (C) {
        tqt_init_c_kernel_gemm_f32_f32_4x4vl_rvv<MR>(vl, C, ldc);
    } else {
        tqt_init_zero_kernel_gemm_f32_f32_4x4vl_rvv<MR>(vl);
    }

    // Prologue
    tqt_load_a_kernel_gemm_f32_f32_4x4vl_rvv<MR>(a_k_ptr, a_stride_row);
    tqt_load_b_kernel_gemm_f32_f32_4x4vl_rvv(vl, b_k_ptr);
    a_k_ptr += MR;
    b_k_ptr += vl;

    // Steady-state K-loop
    for (size_t k_idx = 1; k_idx < K; k_idx++) {
        tqt_fused_load_vfmacc_axb_kernel_gemm_f32_f32_4x4vl_rvv<MR>(vl, a_k_ptr, a_stride_row,
                                                                    b_k_ptr);
        a_k_ptr += MR;
        b_k_ptr += vl;
    }

    // Epilogue
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

void tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_4x4vl_rvv(size_t m, size_t n, size_t k, const void *A,
                                                        size_t lda, size_t k_idx_a, const void *B,
                                                        size_t ldb, size_t k_idx_b, const void *C,
                                                        size_t ldc, void *D, size_t ldd,
                                                        const void *bias, float clamp_min,
                                                        float clamp_max)
{
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            // Packed A pointer: trailing partial m-block has stride actual_m (not m_step),
            // so when k_idx_a > 0 we compensate for the over-counted advancement.
            const float *a_ptr;
            if (actual_m == m_step || k_idx_a == 0) {
                a_ptr = (const float *)A + m_idx * lda;
            } else {
                a_ptr = (const float *)A + m_idx * lda - k_idx_a * (m_step - actual_m);
            }
            // Packed B pointer: same compensation for trailing partial n-block
            // (per-k row stride within the block equals actual_n, not n_step).
            const float *b_ptr;
            if (actual_n == n_step || k_idx_b == 0) {
                b_ptr = (const float *)B + n_idx * ldb;
            } else {
                b_ptr = (const float *)B + n_idx * ldb - k_idx_b * (n_step - actual_n);
            }
            const float *c_ptr = C ? (const float *)C + m_idx * ldc + n_idx : NULL;
            const float *bias_ptr = bias ? (const float *)bias + n_idx : NULL;
            float *d_ptr = (float *)D + m_idx * ldd + n_idx;

            // Single LMUL=4 kernel call: vl = actual_n. The kernel uses vl as the
            // per-k stride into packed B, matching the new flat block layout.
            gemm_mrxnr_4vl_rvv(actual_m, actual_n, k, d_ptr, a_ptr, b_ptr, c_ptr, bias_ptr, ldd,
                               lda, ldb, ldc, clamp_min, clamp_max, actual_n);
        }
    }
}
