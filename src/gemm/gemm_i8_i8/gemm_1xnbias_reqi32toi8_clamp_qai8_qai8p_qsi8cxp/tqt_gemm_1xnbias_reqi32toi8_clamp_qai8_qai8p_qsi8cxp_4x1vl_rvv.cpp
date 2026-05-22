//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_i8_i8_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/tqt_utils_i8_i8.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB;
static const size_t m_step = 4;
static const size_t n_step = vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(int8_t);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx * TQT_MIN(actual_n, n_step)) * sizeof(int8_t);
}

size_t tqt_get_c_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                   size_t n_idx,
                                                                                   size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(int32_t);
}

size_t tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(int32_t);
}

size_t tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                   size_t n_idx,
                                                                                   size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(int8_t);
}

size_t tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(int8_t);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t m,
                                                                                        size_t k)
{
    return m * k * sizeof(int8_t);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t n,
                                                                                        size_t k)
{
    return n * k * sizeof(int8_t);
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed)
{
    tqt_gemm_pack_mxk_4x1_e8_rvv(m, k, lda, lda_packed, k_idx, A, A_packed);
}

void tqt_run_b_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed)
{
    tqt_gemm_pack_kxn_1x1vl_rvv(n, k, ldb, ldb_packed, k_idx, B, B_packed, 8);
}

void tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed)
{
    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, k, ldb, ldb_packed, k_idx, B, B_packed);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

// Single GEMM kernel iteration for MRxvl tile with requantization
// Packed format: A rows are interleaved, B columns are contiguous
template <size_t MR>
static void gemm_kernel(size_t K, int8_t *D, const int8_t *A, const int8_t *B, const int32_t *C,
                        const int32_t *BIAS, size_t ldd, size_t lda, size_t ldb, size_t ldc,
                        int8_t clamp_min, int8_t clamp_max, size_t vl, const int32_t *mult,
                        const int32_t *shift, int32_t zp)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    TQT_UNUSED(lda);
    TQT_UNUSED(ldb);

    size_t a_stride_row = 1;
    size_t d_stride_row = ldd;

    const int8_t *a_ptr = A;
    const int8_t *b_ptr = B;
    const int8_t *a_ptrs[MR];

    if (C) {
        tqt_init_c_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, C, ldc, 1);
    } else {
        tqt_init_zero_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl);
    }

    for (size_t k_idx = 0; k_idx < K; k_idx++) {
        tqt_get_a_ptr_gemm_i8_i8_4x1vl_rvv<MR>(a_ptr, a_stride_row, a_ptrs);
        tqt_load_b_kernel_gemm_i8_i8_4x1vl_rvv(vl, b_ptr);
        tqt_vmacc_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, a_ptrs);
        a_ptr += MR;
        b_ptr += vl;
    }

    if (BIAS) {
        tqt_add_1xnbias_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, BIAS);
    }

    tqt_requantize_1xn_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, mult, shift, zp);
    tqt_clamp_i8_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, clamp_min, clamp_max);
    tqt_store_i8_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, D, d_stride_row);
}

using gemm_kernel_fn_t = void (*)(size_t K, int8_t *D, const int8_t *A, const int8_t *B,
                                  const int32_t *C, const int32_t *BIAS, size_t ldd, size_t lda,
                                  size_t ldb, size_t ldc, int8_t clamp_min, int8_t clamp_max,
                                  size_t vl, const int32_t *mult, const int32_t *shift, int32_t zp);

static constexpr gemm_kernel_fn_t kernel_table[4] = {
    gemm_kernel<1>,
    gemm_kernel<2>,
    gemm_kernel<3>,
    gemm_kernel<4>,
};

// ============================================================================
// Gemm Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
    const struct tqt_qai8_qai8_qsi8cx_params *params)
{
    if (!params) {
        return;
    }

    const int32_t *C_ptr = (const int32_t *)C;
    const int32_t *bias_ptr = (const int32_t *)bias;
    int8_t *D_ptr = (int8_t *)D;

    // Pre-compute requantization parameters for all columns
    // real_scale = scale_a * scale_b[i] / scale_d
    int32_t mult[n];
    int32_t shift[n];
    double real_scale =
        (double)params->scale_a * (double)params->scale_b[0] / (double)params->scale_d;
    tqt_quantize_multiplier(real_scale, &mult[0], &shift[0]);
    shift[0] = -shift[0] - 1;
    for (size_t i = 1; i < n; i++) {
        if (params->quant_channel_b == 1) {
            mult[i] = mult[0];
            shift[i] = shift[0];
        } else {
            real_scale =
                (double)params->scale_a * (double)params->scale_b[i] / (double)params->scale_d;
            tqt_quantize_multiplier(real_scale, &mult[i], &shift[i]);
            shift[i] = -shift[i] - 1;
        }
    }

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
            const int32_t *c_ptr = C_ptr ? C_ptr + m_idx * ldc + n_idx : NULL;
            const int32_t *bias_tile_ptr = bias_ptr ? bias_ptr + n_idx : NULL;
            int8_t *d_ptr = D_ptr + m_idx * ldd + n_idx;

            const int32_t *mult_ptr = mult + n_idx;
            const int32_t *shift_ptr = shift + n_idx;

            size_t mr_idx = actual_m - 1;
            kernel_table[mr_idx](k, d_ptr, a_ptr, b_ptr, c_ptr, bias_tile_ptr, ldd, lda, ldb, ldc,
                                 clamp_min, clamp_max, actual_n, mult_ptr, shift_ptr,
                                 params->zero_point_d);
        }
    }
}
