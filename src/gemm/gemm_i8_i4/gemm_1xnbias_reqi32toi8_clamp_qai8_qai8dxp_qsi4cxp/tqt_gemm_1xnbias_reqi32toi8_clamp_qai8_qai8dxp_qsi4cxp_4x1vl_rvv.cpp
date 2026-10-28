//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_i8_i4/tqt_kernel_gemm_i8_i4_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/tqt_requant_epilogue_i8_i4_4x1vl_rvv.h"

static const size_t m_step = 4;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(void)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    return n_step;
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(int8_t);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(n_idx % n_step == 0);
    TQT_ASSUME(k_idx % 2 == 0);
    const size_t ldb_e8 = ldb / 2;
    return n_idx * ldb_e8 + (k_idx / 2) * TQT_MIN(actual_n, n_step);
}

size_t tqt_get_c_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m_idx,
                                                                                     size_t n_idx,
                                                                                     size_t ldc)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float);
}

size_t tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
    size_t n_idx)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float);
}

size_t tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m_idx,
                                                                                     size_t n_idx,
                                                                                     size_t ldd)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(int8_t);
}

size_t tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m,
                                                                                   size_t n)
{
    return m * n * sizeof(int8_t);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(size_t m,
                                                                                          size_t k)
{
    return m * k * sizeof(int8_t);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(size_t n,
                                                                                          size_t k)
{
    return n * k / 2;
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed)
{
    tqt_gemm_pack_mxk_4x1_e8_rvv(m, k, lda, lda_packed, k_idx, A, A_packed);
}

void tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed)
{
    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, k / 2, ldb / 2, ldb_packed / 2, k_idx / 2, B, B_packed);
}

// ============================================================================
// Template GEMM kernel (packed A + packed B): i32 accumulator, int8 requantize
// ============================================================================

template <size_t MR>
static void gemm_kernel(size_t K, int8_t *D, const int8_t *A_packed, const uint8_t *B_packed,
                        const float *C, const float *bias, size_t ldd, size_t lda, size_t ldb,
                        size_t ldc, int32_t clamp_min, int32_t clamp_max, size_t vl,
                        const float *scale_a, const int32_t *zp_a, const float *scale_b,
                        float scale_d, int32_t zero_point_d, const int32_t *bias_i32)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");
    TQT_UNUSED(lda);
    TQT_UNUSED(ldb);

    const size_t d_stride_row = ldd;
    const size_t c_stride_row = ldc;

    tqt_init_zero_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl);
    tqt_init_rsum_b_kernel_gemm_i8_i4_4x1vl_rvv(vl);

    for (size_t k = 0; k < K; k += 2) {
        const size_t kp = k / 2;
        const uint8_t *b_col = B_packed + kp * vl;
        tqt_dequant_b_kernel_gemm_i8_i4_4x1vl_rvv(vl, b_col);
        tqt_accum_rsum_b_kernel_gemm_i8_i4_4x1vl_rvv(vl);

        const int8_t *a_ptrs_lo[MR];
        const int8_t *a_ptrs_hi[MR];
        const int8_t *a_lo_base = A_packed + k * MR;
        const int8_t *a_hi_base = A_packed + (k + 1) * MR;
        if constexpr (MR >= 1) {
            a_ptrs_lo[0] = a_lo_base + 0;
            a_ptrs_hi[0] = a_hi_base + 0;
        }
        if constexpr (MR >= 2) {
            a_ptrs_lo[1] = a_lo_base + 1;
            a_ptrs_hi[1] = a_hi_base + 1;
        }
        if constexpr (MR >= 3) {
            a_ptrs_lo[2] = a_lo_base + 2;
            a_ptrs_hi[2] = a_hi_base + 2;
        }
        if constexpr (MR >= 4) {
            a_ptrs_lo[3] = a_lo_base + 3;
            a_ptrs_hi[3] = a_hi_base + 3;
        }

        tqt_vmacc_lo_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, a_ptrs_lo);
        tqt_vmacc_hi_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, a_ptrs_hi);
    }

    tqt_apply_zp_correction_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, zp_a);
    if (bias_i32)
        tqt_add_1xnbias_i32_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, bias_i32);
    tqt_dequant_acc_to_f32_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, scale_a, scale_b);

    if (C) {
        tqt_add_c_f32_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, C, c_stride_row);
    }

    if (bias) {
        tqt_add_1xnbias_f32_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, bias);
    }

    tqt_requantize_f32_to_i8_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, scale_d, zero_point_d, clamp_min,
                                                             clamp_max, D, d_stride_row);
}

using gemm_kernel_fn_t = void (*)(size_t K, int8_t *D, const int8_t *A_packed,
                                  const uint8_t *B_packed, const float *C, const float *bias,
                                  size_t ldd, size_t lda, size_t ldb, size_t ldc, int32_t clamp_min,
                                  int32_t clamp_max, size_t vl, const float *scale_a,
                                  const int32_t *zp_a, const float *scale_b, float scale_d,
                                  int32_t zero_point_d, const int32_t *bias_i32);

static constexpr gemm_kernel_fn_t kernel_table[4] = {
    gemm_kernel<1>,
    gemm_kernel<2>,
    gemm_kernel<3>,
    gemm_kernel<4>,
};

// ============================================================================
// Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
    const struct tqt_qai8_qai8dx_qsi4cx_params *params)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    if (!params) {
        return;
    }

    const int8_t *A_ptr = (const int8_t *)A_packed;
    const uint8_t *B_ptr = (const uint8_t *)B_packed;
    const float *C_ptr = (const float *)C;
    const float *bias_ptr = (const float *)bias;
    int8_t *D_ptr = (int8_t *)D;

    TQT_ASSUME(k_idx_b % 2 == 0);
    const size_t ldb_e8 = ldb / 2;
    const size_t origin_m = TQT_MIN(m, m_step);
    const size_t origin_n = TQT_MIN(n, n_step);
    const tqt_scoped_rne_kernel_gemm_i8_i4_4x1vl_rvv rounding;
    const float scale_d = params->scale_d;
    const int32_t zero_point_d = params->zero_point_d;
    const int32_t clamp_min_i32 = (int32_t)clamp_min;
    const int32_t clamp_max_i32 = (int32_t)clamp_max;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const int8_t *a_tile = A_ptr + m_idx * lda + k_idx_a * actual_m - k_idx_a * origin_m;
            const uint8_t *b_tile =
                B_ptr + n_idx * ldb_e8 + (k_idx_b / 2) * actual_n - (k_idx_b / 2) * origin_n;
            const float *c_tile = C_ptr ? C_ptr + m_idx * ldc + n_idx : nullptr;
            const float *bias_tile = bias_ptr ? bias_ptr + n_idx : nullptr;
            int8_t *d_tile = D_ptr + m_idx * ldd + n_idx;

            const float *scale_a_tile = params->scale_a + m_idx;
            const int32_t *zp_a_tile = params->zero_point_a + m_idx;
            const float *scale_b_tile = params->scale_b + n_idx;

            const size_t mr_idx = actual_m - 1;
            kernel_table[mr_idx](k, d_tile, a_tile, b_tile, c_tile, bias_tile, ldd, lda, ldb, ldc,
                                 clamp_min_i32, clamp_max_i32, actual_n, scale_a_tile, zp_a_tile,
                                 scale_b_tile, scale_d, zero_point_d,
                                 params->bias_i32 ? params->bias_i32 + n_idx : nullptr);
        }
    }
}
