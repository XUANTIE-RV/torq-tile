//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i2/tqt_kernel_gemm_i8_i2_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i2/tqt_utils_i8_i2.h"
#include "src/gemm/gemm_i8_i4/tqt_requant_epilogue_i8_i4_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/tqt_kernel_gemm_i8_i8_4x1vl_rvv.h"

static const size_t m_step = 4;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv(void)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    return n_step;
}

size_t tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv(size_t m_idx,
                                                                                  size_t k_idx,
                                                                                  size_t lda)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx) * sizeof(int8_t);
}

size_t tqt_get_b_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv(size_t n_idx,
                                                                                  size_t k_idx,
                                                                                  size_t ldb)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(n_idx % n_step == 0);
    TQT_ASSUME(k_idx % 4 == 0);
    TQT_ASSUME(ldb % 4 == 0);
    // [N, K/4] bytes: byte offset = n_idx * (K/4) + k_idx/4
    return n_idx * (ldb / 4) + k_idx / 4;
}

size_t tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv(size_t n_idx)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(int32_t);
}

size_t tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv(size_t m_idx,
                                                                                  size_t n_idx,
                                                                                  size_t ldd)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(int8_t);
}

size_t tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(int8_t);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

template <size_t MR, bool USE_DEFAULT_LUT>
static void gemm_kernel(size_t K, int8_t *D, const int8_t *A, size_t lda, const uint8_t *B,
                        size_t ldb_bytes, const int32_t *bias, size_t ldd, int8_t clamp_min,
                        int8_t clamp_max, size_t actual_n, const int32_t *neg_zp_a,
                        const float *scale_a, const float *scale_b, const int32_t *sum_b,
                        const int8_t *lut, float scale_d, int32_t zero_point_d)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (!USE_DEFAULT_LUT) {
        tqt_load_custom_lut_kernel_gemm_i8_i2_4x1vl_rvv(actual_n, lut);
    } else {
        TQT_UNUSED(lut);
    }

    tqt_init_zero_kernel_gemm_i8_i2_4x1vl_rvv<MR>(actual_n);

    tqt_dot_bt_kernel_gemm_i8_i2_4x1vl_rvv<MR, USE_DEFAULT_LUT>(K, A, lda, B, ldb_bytes, actual_n);

    tqt_apply_zp_correction_kernel_gemm_i8_i2_4x1vl_rvv<MR>(actual_n, sum_b, neg_zp_a);
    if (bias)
        tqt_add_1xnbias_kernel_gemm_i8_i8_4x1vl_rvv<MR>(actual_n, bias);
    tqt_dequant_to_f32_kernel_gemm_i8_i2_4x1vl_rvv<MR>(actual_n, scale_a, scale_b);
    tqt_requantize_f32_to_i8_kernel_gemm_i8_i4_4x1vl_rvv<MR>(actual_n, scale_d, zero_point_d,
                                                             clamp_min, clamp_max, D, ldd);
}

using gemm_kernel_fn_t = void (*)(size_t K, int8_t *D, const int8_t *A, size_t lda,
                                  const uint8_t *B, size_t ldb_bytes, const int32_t *bias,
                                  size_t ldd, int8_t clamp_min, int8_t clamp_max, size_t actual_n,
                                  const int32_t *neg_zp_a, const float *scale_a,
                                  const float *scale_b, const int32_t *sum_b, const int8_t *lut,
                                  float scale_d, int32_t zero_point_d);

static constexpr gemm_kernel_fn_t kernel_table[4][2] = {
    {gemm_kernel<1, true>, gemm_kernel<1, false>},
    {gemm_kernel<2, true>, gemm_kernel<2, false>},
    {gemm_kernel<3, true>, gemm_kernel<3, false>},
    {gemm_kernel<4, true>, gemm_kernel<4, false>},
};

// ============================================================================
// Gemm Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi2cxt_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, void *D, size_t ldd, const int32_t *bias, int8_t clamp_min,
    int8_t clamp_max, const int8_t *lut, const struct tqt_qai8_qai8_qsi2cx_params *params)
{
    const size_t VLENB = tqt_get_vlenb();
    const size_t n_step = VLENB;

    TQT_UNUSED(k_idx_a);
    TQT_UNUSED(k_idx_b);

    if (!params || m == 0 || n == 0) {
        return;
    }
    TQT_ASSUME(k % 4 == 0);
    TQT_ASSUME(ldb % 4 == 0);

    const int8_t *A_ptr = (const int8_t *)A;
    const uint8_t *B_ptr = (const uint8_t *)B;
    const int32_t *bias_ptr = (const int32_t *)bias;
    int8_t *D_ptr = (int8_t *)D;

    const bool use_default_lut = (lut == nullptr);
    const size_t ldb_bytes = ldb / 4;

    const tqt_scoped_rne_kernel_gemm_i8_i4_4x1vl_rvv rounding;
    const float scale_d = params->scale_d;
    int32_t neg_zp_a_full[4];
    float scale_a_full[4];
    for (size_t i = 0; i < 4; ++i) {
        neg_zp_a_full[i] = -params->zero_point_a;
        scale_a_full[i] = params->scale_a;
    }

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const int8_t *a_ptr = A_ptr + m_idx * lda;
            const uint8_t *b_ptr = B_ptr + n_idx * ldb_bytes;
            const int32_t *bias_tile_ptr = bias_ptr ? bias_ptr + n_idx : nullptr;
            int8_t *d_ptr = D_ptr + m_idx * ldd + n_idx;

            const int32_t *neg_zp_tile = neg_zp_a_full;
            const float *scale_a_tile = scale_a_full;
            const float *scale_b_tile = params->scale_b + n_idx;
            int32_t sum_b_tile[actual_n];
            for (size_t col = 0; col < actual_n; ++col)
                sum_b_tile[col] = tqt_compute_sum_b_row_lut(b_ptr + col * ldb_bytes, k, 1, lut);

            const size_t mr_idx = actual_m - 1;
            const size_t lut_idx = use_default_lut ? 0 : 1;
            kernel_table[mr_idx][lut_idx](k, d_ptr, a_ptr, lda, b_ptr, ldb_bytes, bias_tile_ptr,
                                          ldd, clamp_min, clamp_max, actual_n, neg_zp_tile,
                                          scale_a_tile, scale_b_tile, sum_b_tile, lut, scale_d,
                                          params->zero_point_d);
        }
    }
}
