//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_i8_i8_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/tqt_utils_i8_i8.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB;
static const size_t m_step = 4;
static const size_t n_step = vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(size_t m_idx,
                                                                                  size_t k_idx,
                                                                                  size_t lda)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx) * sizeof(int8_t);
}

size_t tqt_get_b_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(size_t n_idx,
                                                                                  size_t k_idx,
                                                                                  size_t ldb)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx) * sizeof(int8_t);
}

size_t tqt_get_c_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(size_t m_idx,
                                                                                  size_t n_idx,
                                                                                  size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(int32_t);
}

size_t tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(int32_t);
}

size_t tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(size_t m_idx,
                                                                                  size_t n_idx,
                                                                                  size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(int8_t);
}

size_t tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(int8_t);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

// Single GEMM kernel iteration for MRxvl tile with requantization
template <size_t MR>
static void gemm_kernel(size_t K, int8_t *D, const int8_t *A, const int8_t *B, const int32_t *C,
                        const int32_t *BIAS, size_t ldd, size_t lda, size_t ldb, size_t ldc,
                        int8_t clamp_min, int8_t clamp_max, size_t vl, const int32_t *mult,
                        const int32_t *shift, int32_t zp)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");
    size_t a_stride_row = lda;
    size_t b_stride_row = ldb;
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
        tqt_load_bt_kernel_gemm_i8_i8_4x1vl_rvv(vl, b_ptr, b_stride_row);
        tqt_vmacc_kernel_gemm_i8_i8_4x1vl_rvv<MR>(vl, a_ptrs);
        a_ptr += 1;
        b_ptr += 1;
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

void tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    int8_t clamp_min, int8_t clamp_max, const struct tqt_qai8_qai8_qsi8cx_params *params)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);

    if (!params) {
        return;
    }

    const int8_t *A_ptr = (const int8_t *)A;
    const int8_t *B_ptr = (const int8_t *)B;
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

            const int8_t *a_ptr = A_ptr + m_idx * lda;
            const int8_t *b_ptr = B_ptr + n_idx * ldb;
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
