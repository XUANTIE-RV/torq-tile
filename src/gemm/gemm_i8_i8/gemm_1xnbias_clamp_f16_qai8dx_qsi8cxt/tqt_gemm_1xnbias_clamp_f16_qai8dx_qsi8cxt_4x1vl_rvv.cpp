//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, fp16 && vec-fp16.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB;
static const size_t m_step = 4;
static const size_t n_step = vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                        size_t lda)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx) * sizeof(int8_t);
}

size_t tqt_get_b_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                        size_t ldb)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldb + k_idx) * sizeof(int8_t);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(float16_t);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float16_t);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float16_t);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float16_t);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

// Single GEMM kernel iteration for MR x vl tile.
// A: original layout (per-row stride), B: transposed [N, K] (strided B load).
template <size_t MR>
static void gemm_kernel(size_t K, float16_t *D, const int8_t *A, const int8_t *B, size_t ldd,
                        size_t lda, size_t ldb, size_t vl, const float scale_a[MR],
                        const int32_t neg_zp_a[MR], const float *scale_b, const float16_t *bias,
                        float16_t clamp_min, float16_t clamp_max)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    const int8_t *a_ptr = A;
    const int8_t *b_ptr = B;
    const int8_t *a_ptrs[MR];

    tqt_init_zero_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv<MR>(vl);

    for (size_t k_idx = 0; k_idx < K; k_idx++) {
        tqt_get_a_ptr_gemm_qai8dx_qsi8cx_4x1vl_rvv<MR>(a_ptr, lda, a_ptrs);
        tqt_load_bt_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(vl, b_ptr, ldb);
        tqt_widen_b_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv(vl);
        tqt_vwmacc_zp_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv<MR>(vl, a_ptrs, neg_zp_a);
        a_ptr += 1;
        b_ptr += 1;
    }

    tqt_dequant_fp16_kernel_gemm_qai8dx_qsi8cx_4x1vl_rvv<MR>(vl, scale_a, scale_b, bias, clamp_min,
                                                             clamp_max, D, ldd);
}

using gemm_kernel_fn_t = void (*)(size_t K, float16_t *D, const int8_t *A, const int8_t *B,
                                  size_t ldd, size_t lda, size_t ldb, size_t vl,
                                  const float scale_a[], const int32_t neg_zp_a[],
                                  const float *scale_b, const float16_t *bias, float16_t clamp_min,
                                  float16_t clamp_max);

static constexpr gemm_kernel_fn_t kernel_table[4] = {
    gemm_kernel<1>,
    gemm_kernel<2>,
    gemm_kernel<3>,
    gemm_kernel<4>,
};

// ============================================================================
// Gemm Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float16_t clamp_min, float16_t clamp_max, const struct tqt_qai8dxp_qsi8cxp_params *params)
{
    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);
    TQT_UNUSED(C);
    TQT_UNUSED(ldc);

    if (!params) {
        return;
    }

    const int8_t *A_ptr = (const int8_t *)A;
    const int8_t *B_ptr = (const int8_t *)B;
    const float16_t *bias_ptr = (const float16_t *)bias;
    float16_t *D_ptr = (float16_t *)D;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const int8_t *a_ptr = A_ptr + m_idx * lda;
            const int8_t *b_ptr = B_ptr + n_idx * ldb;
            const float16_t *bias_tile_ptr = bias_ptr ? bias_ptr + n_idx : NULL;
            float16_t *d_ptr = D_ptr + m_idx * ldd + n_idx;

            // Per-row scale_a / -zero_point_a for this tile (MR active rows)
            float scale_a_tile[m_step] = {0.f, 0.f, 0.f, 0.f};
            int32_t neg_zp_a_tile[m_step] = {0, 0, 0, 0};
            for (size_t i = 0; i < actual_m; ++i) {
                scale_a_tile[i] = params->scale_a[m_idx + i];
                neg_zp_a_tile[i] = -params->zero_point_a[m_idx + i];
            }

            const float *scale_b_tile = params->scale_b + n_idx;

            size_t mr_idx = actual_m - 1;
            kernel_table[mr_idx](k, d_ptr, a_ptr, b_ptr, ldd, lda, ldb, actual_n, scale_a_tile,
                                 neg_zp_a_tile, scale_b_tile, bias_tile_ptr, clamp_min, clamp_max);
        }
    }
}
