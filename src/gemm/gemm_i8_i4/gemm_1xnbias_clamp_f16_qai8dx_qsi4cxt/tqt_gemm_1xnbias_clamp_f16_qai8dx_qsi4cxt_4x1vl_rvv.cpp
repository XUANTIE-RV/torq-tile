//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i4/tqt_kernel_gemm_i8_i4_4x1vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB;
static const size_t m_step = 4;
static const size_t n_step = vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                        size_t lda)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx) * sizeof(int8_t);
}

size_t tqt_get_b_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                        size_t ldb)
{
    TQT_ASSUME(n_idx % n_step == 0);
    TQT_ASSUME(k_idx % 2 == 0);
    return n_idx * (ldb / 2) + k_idx / 2;
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float16_t);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float16_t);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float16_t);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float16_t);
}

// ============================================================================
// Template GEMM kernel: MR rows × vl columns, i32 accumulator, f32 epilogue
// ============================================================================

template <size_t MR>
static void gemm_kernel(size_t K, float16_t *D, const int8_t *A, const uint8_t *B,
                        const float16_t *C, const float16_t *bias, size_t ldd, size_t lda,
                        size_t ldb, size_t ldc, float16_t clamp_min, float16_t clamp_max, size_t vl,
                        const float *scale_a, const int32_t *zp_a, const float *scale_b)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    const size_t a_stride_row = lda;
    const size_t b_byte_stride = ldb / 2;
    const size_t d_stride_row = ldd;
    const size_t c_stride_row = ldc;

    tqt_init_zero_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl);
    tqt_init_rsum_b_kernel_gemm_i8_i4_4x1vl_rvv(vl);

    // K loop: process 2 K elements per iteration (one byte of B per N row).
    for (size_t k = 0; k < K; k += 2) {
        const uint8_t *b_col = B + k / 2;
        tqt_dequant_bt_kernel_gemm_i8_i4_4x1vl_rvv(vl, b_col, b_byte_stride);
        tqt_accum_rsum_b_kernel_gemm_i8_i4_4x1vl_rvv(vl);

        const int8_t *a_ptrs_lo[MR];
        const int8_t *a_ptrs_hi[MR];
        if constexpr (MR >= 1) {
            a_ptrs_lo[0] = A + 0 * a_stride_row + k;
            a_ptrs_hi[0] = A + 0 * a_stride_row + k + 1;
        }
        if constexpr (MR >= 2) {
            a_ptrs_lo[1] = A + 1 * a_stride_row + k;
            a_ptrs_hi[1] = A + 1 * a_stride_row + k + 1;
        }
        if constexpr (MR >= 3) {
            a_ptrs_lo[2] = A + 2 * a_stride_row + k;
            a_ptrs_hi[2] = A + 2 * a_stride_row + k + 1;
        }
        if constexpr (MR >= 4) {
            a_ptrs_lo[3] = A + 3 * a_stride_row + k;
            a_ptrs_hi[3] = A + 3 * a_stride_row + k + 1;
        }

        tqt_vmacc_lo_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, a_ptrs_lo);
        tqt_vmacc_hi_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, a_ptrs_hi);
    }

    // Epilogue: zp_a correction → dequant to f32 → +C → +bias → clamp → store.
    tqt_apply_zp_correction_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, zp_a);
    tqt_dequant_acc_to_f32_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, scale_a, scale_b);

    if (C) {
        tqt_add_c_f16_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, C, c_stride_row);
    }

    if (bias) {
        tqt_add_1xnbias_f16_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, bias);
    }

    const float min_f32 = static_cast<float>(clamp_min);
    const float max_f32 = static_cast<float>(clamp_max);
    tqt_clamp_f32_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, min_f32, max_f32);

    tqt_store_narrow_f16_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, D, d_stride_row);
}

using gemm_kernel_fn_t = void (*)(size_t K, float16_t *D, const int8_t *A, const uint8_t *B,
                                  const float16_t *C, const float16_t *bias, size_t ldd, size_t lda,
                                  size_t ldb, size_t ldc, float16_t clamp_min, float16_t clamp_max,
                                  size_t vl, const float *scale_a, const int32_t *zp_a,
                                  const float *scale_b);

static constexpr gemm_kernel_fn_t kernel_table[4] = {
    gemm_kernel<1>,
    gemm_kernel<2>,
    gemm_kernel<3>,
    gemm_kernel<4>,
};

// ============================================================================
// Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float16_t clamp_min, float16_t clamp_max, const struct tqt_qai8dx_qsi4cx_params *params)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);

    if (!params) {
        return;
    }

    const int8_t *A_ptr = (const int8_t *)A;
    const uint8_t *B_ptr = (const uint8_t *)B;
    const float16_t *C_ptr = (const float16_t *)C;
    const float16_t *bias_ptr = (const float16_t *)bias;
    float16_t *D_ptr = (float16_t *)D;

    const size_t b_byte_stride = ldb / 2;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const int8_t *a_tile = A_ptr + m_idx * lda;
            const uint8_t *b_tile = B_ptr + n_idx * b_byte_stride;
            const float16_t *c_tile = C_ptr ? C_ptr + m_idx * ldc + n_idx : nullptr;
            const float16_t *bias_tile = bias_ptr ? bias_ptr + n_idx : nullptr;
            float16_t *d_tile = D_ptr + m_idx * ldd + n_idx;

            const float *scale_a_tile = params->scale_a + m_idx;
            const int32_t *zp_a_tile = params->zero_point_a + m_idx;
            const float *scale_b_tile = params->scale_b + n_idx;

            const size_t mr_idx = actual_m - 1;
            kernel_table[mr_idx](k, d_tile, a_tile, b_tile, c_tile, bias_tile, ldd, lda, ldb, ldc,
                                 clamp_min, clamp_max, actual_n, scale_a_tile, zp_a_tile,
                                 scale_b_tile);
        }
    }
}
