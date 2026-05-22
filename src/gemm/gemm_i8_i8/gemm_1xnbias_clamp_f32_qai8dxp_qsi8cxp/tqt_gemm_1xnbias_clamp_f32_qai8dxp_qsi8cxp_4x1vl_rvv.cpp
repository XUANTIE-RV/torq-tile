//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB;
static const size_t m_step = 4;
static const size_t n_step = vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                size_t k_idx,
                                                                                size_t lda,
                                                                                size_t actual_m)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(int8_t);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n_idx,
                                                                                size_t k_idx,
                                                                                size_t ldb,
                                                                                size_t actual_n)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx * TQT_MIN(actual_n, n_step)) * sizeof(int8_t);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m, size_t k)
{
    return m * k * sizeof(int8_t);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n, size_t k)
{
    return n * k * sizeof(int8_t);
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t m, size_t k, size_t lda,
                                                                     size_t lda_packed,
                                                                     size_t k_idx, const void *A,
                                                                     void *A_packed)
{
    tqt_gemm_pack_mxk_4x1_e8_rvv(m, k, lda, lda_packed, k_idx, A, A_packed);
}

void tqt_run_b_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n, size_t k, size_t ldb,
                                                                     size_t ldb_packed,
                                                                     size_t k_idx, const void *B,
                                                                     void *B_packed)
{
    tqt_gemm_pack_kxn_1x1vl_rvv(n, k, ldb, ldb_packed, k_idx, B, B_packed, 8);
}

void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(size_t n, size_t k,
                                                                      size_t ldb, size_t ldb_packed,
                                                                      size_t k_idx, const void *B,
                                                                      void *B_packed)
{
    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, k, ldb, ldb_packed, k_idx, B, B_packed);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

// Single GEMM kernel iteration for MR x vl tile.
// A is packed: rows interleaved by MR. B is packed: cols interleaved by nr.
template <size_t MR>
static void gemm_kernel(size_t K, float *D, const int8_t *A, const int8_t *B, size_t ldd, size_t vl,
                        const float scale_a[MR], const int32_t neg_zp_a[MR], const float *scale_b,
                        const float *bias, float clamp_min, float clamp_max)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    const int8_t *a_ptr = A;
    const int8_t *b_ptr = B;
    const int8_t *a_ptrs[MR];

    tqt_init_zero_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv<MR>(vl);

    for (size_t k_idx = 0; k_idx < K; k_idx++) {
        // Within the packed A tile, MR rows are interleaved at byte granularity.
        tqt_get_a_ptr_gemm_qai8dx_qsi8cx_4x1vl_rvv<MR>(a_ptr, 1, a_ptrs);
        tqt_load_b_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(vl, b_ptr);
        tqt_widen_b_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(vl);
        tqt_vwmacc_zp_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv<MR>(vl, a_ptrs, neg_zp_a);
        a_ptr += MR;
        b_ptr += vl;
    }

    tqt_dequant_fp32_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv<MR>(vl, scale_a, scale_b, bias, clamp_min,
                                                             clamp_max, D, ldd);
}

using gemm_kernel_fn_t = void (*)(size_t K, float *D, const int8_t *A, const int8_t *B, size_t ldd,
                                  size_t vl, const float scale_a[], const int32_t neg_zp_a[],
                                  const float *scale_b, const float *bias, float clamp_min,
                                  float clamp_max);

static constexpr gemm_kernel_fn_t kernel_table[4] = {
    gemm_kernel<1>,
    gemm_kernel<2>,
    gemm_kernel<3>,
    gemm_kernel<4>,
};

// ============================================================================
// Gemm Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qai8dxp_qsi8cxp_params *params)
{
    TQT_UNUSED(C);
    TQT_UNUSED(ldc);

    if (!params) {
        return;
    }

    const float *bias_ptr = (const float *)bias;
    float *D_ptr = (float *)D;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const int8_t *a_ptr;
            const int8_t *b_ptr;
            if (actual_m == m_step || k_idx_a == 0) {
                a_ptr = (const int8_t *)A_packed + m_idx * lda;
            } else {
                a_ptr = (const int8_t *)A_packed + m_idx * lda - k_idx_a * (m_step - actual_m);
            }
            if (actual_n == n_step || k_idx_b == 0) {
                b_ptr = (const int8_t *)B_packed + n_idx * ldb;
            } else {
                b_ptr = (const int8_t *)B_packed + n_idx * ldb - k_idx_b * (n_step - actual_n);
            }
            const float *bias_tile_ptr = bias_ptr ? bias_ptr + n_idx : NULL;
            float *d_ptr = D_ptr + m_idx * ldd + n_idx;

            float scale_a_tile[m_step] = {0.f, 0.f, 0.f, 0.f};
            int32_t neg_zp_a_tile[m_step] = {0, 0, 0, 0};
            for (size_t i = 0; i < actual_m; ++i) {
                scale_a_tile[i] = params->scale_a[m_idx + i];
                neg_zp_a_tile[i] = -params->zero_point_a[m_idx + i];
            }

            const float *scale_b_tile = params->scale_b + n_idx;

            size_t mr_idx = actual_m - 1;
            kernel_table[mr_idx](k, d_ptr, a_ptr, b_ptr, ldd, actual_n, scale_a_tile, neg_zp_a_tile,
                                 scale_b_tile, bias_tile_ptr, clamp_min, clamp_max);
        }
    }
}
