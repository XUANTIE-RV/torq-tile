//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_i8_i2/tqt_kernel_gemm_i8_i2_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i2/tqt_utils_i8_i2.h"

static inline size_t get_vlenb()
{
    static const size_t value = csrr_vlenb();
    return value;
}

static const size_t m_step = 4;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(void)
{
    const size_t VLENB = get_vlenb();
    const size_t n_step = VLENB;

    return n_step;
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m_idx,
                                                                                size_t k_idx,
                                                                                size_t lda,
                                                                                size_t actual_m)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(int8_t);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n_idx,
                                                                                size_t k_idx,
                                                                                size_t ldb,
                                                                                size_t actual_n)
{
    const size_t VLENB = get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(n_idx % n_step == 0);
    TQT_ASSUME(k_idx % 4 == 0);
    TQT_ASSUME(ldb % 4 == 0);
    // Packed B layout is [N/n_step, K/4, n_step] uint8.
    // ldb is given in K elements; convert to bytes by /4.
    const size_t ldb_bytes = ldb / 4;
    return n_idx * ldb_bytes + (k_idx / 4) * TQT_MIN(actual_n, n_step);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n_idx)
{
    const size_t VLENB = get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(float);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldd)
{
    const size_t VLENB = get_vlenb();
    const size_t n_step = VLENB;

    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(float);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m, size_t k)
{
    return m * k * sizeof(int8_t);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n, size_t k)
{
    TQT_ASSUME(k % 4 == 0);
    return n * (k / 4) * sizeof(uint8_t);
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t m, size_t k, size_t lda,
                                                                     size_t lda_packed,
                                                                     size_t k_idx, const void *A,
                                                                     void *A_packed)
{
    // A is plain int8 [M, K]; same layout as i8_i8 ap_bp.
    tqt_gemm_pack_mxk_4x1_e8_rvv(m, k, lda, lda_packed, k_idx, A, A_packed);
}

void tqt_run_b_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n, size_t k, size_t ldb,
                                                                     size_t ldb_packed,
                                                                     size_t k_idx, const void *B,
                                                                     void *B_packed)
{
    // B is [K/4, N] uint8 row-major (4 weights packed s3 s2 s1 s0 per byte). Treat each
    // K-byte (= 4 K elements) as one "k" step of the underlying byte packer.
    TQT_ASSUME(k % 4 == 0);
    TQT_ASSUME(k_idx % 4 == 0);
    TQT_ASSUME(ldb_packed % 4 == 0);
    tqt_gemm_pack_kxn_1x1vl_rvv(n, k / 4, ldb, ldb_packed / 4, k_idx / 4, B, B_packed, 8);
}

void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(size_t n, size_t k,
                                                                      size_t ldb, size_t ldb_packed,
                                                                      size_t k_idx, const void *B,
                                                                      void *B_packed)
{
    // B is [N, K/4] uint8 row-major (transposed). ldb is in K elements (multiple of 4),
    // ldb_packed likewise.
    TQT_ASSUME(k % 4 == 0);
    TQT_ASSUME(k_idx % 4 == 0);
    TQT_ASSUME(ldb % 4 == 0);
    TQT_ASSUME(ldb_packed % 4 == 0);
    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, k / 4, ldb / 4, ldb_packed / 4, k_idx / 4, B, B_packed);
}

// ===========================================================================
// Internal Helper Functions
// ===========================================================================

template <size_t MR, bool USE_DEFAULT_LUT>
static inline void gemm_kernel_kloop(size_t K, const int8_t *A, const uint8_t *B, size_t vl)
{
    const int8_t *a_ptr = A;
    const uint8_t *b_ptr = B;
    const int8_t *a_ptrs[MR];

    // K must be a multiple of 4. Each B byte covers 4 K elements (slots 0..3).
    const size_t k_bytes = K / 4;

    for (size_t kb = 0; kb < k_bytes; ++kb) {
        // Load one byte per N column into v0 (e8 m1).
        tqt_load_b_byte_kernel_gemm_i8_i2_4x1vl_rvv(vl, b_ptr);

        // Slot 0
        if constexpr (USE_DEFAULT_LUT) {
            tqt_decode_b_default_lut_kernel_gemm_i8_i2_4x1vl_rvv<0, false>(vl);
        } else {
            tqt_decode_b_custom_lut_kernel_gemm_i8_i2_4x1vl_rvv<0, false>(vl);
        }
        tqt_get_a_ptr_gemm_i8_i2_4x1vl_rvv<MR>(a_ptr + 0 * MR, 1, a_ptrs);
        tqt_vmacc_kernel_gemm_i8_i2_4x1vl_rvv<MR>(vl, a_ptrs);

        // Slot 1
        if constexpr (USE_DEFAULT_LUT) {
            tqt_decode_b_default_lut_kernel_gemm_i8_i2_4x1vl_rvv<1>(vl);
        } else {
            tqt_decode_b_custom_lut_kernel_gemm_i8_i2_4x1vl_rvv<1>(vl);
        }
        tqt_get_a_ptr_gemm_i8_i2_4x1vl_rvv<MR>(a_ptr + 1 * MR, 1, a_ptrs);
        tqt_vmacc_kernel_gemm_i8_i2_4x1vl_rvv<MR>(vl, a_ptrs);

        // Slot 2
        if constexpr (USE_DEFAULT_LUT) {
            tqt_decode_b_default_lut_kernel_gemm_i8_i2_4x1vl_rvv<2>(vl);
        } else {
            tqt_decode_b_custom_lut_kernel_gemm_i8_i2_4x1vl_rvv<2>(vl);
        }
        tqt_get_a_ptr_gemm_i8_i2_4x1vl_rvv<MR>(a_ptr + 2 * MR, 1, a_ptrs);
        tqt_vmacc_kernel_gemm_i8_i2_4x1vl_rvv<MR>(vl, a_ptrs);

        // Slot 3
        if constexpr (USE_DEFAULT_LUT) {
            tqt_decode_b_default_lut_kernel_gemm_i8_i2_4x1vl_rvv<3>(vl);
        } else {
            tqt_decode_b_custom_lut_kernel_gemm_i8_i2_4x1vl_rvv<3>(vl);
        }
        tqt_get_a_ptr_gemm_i8_i2_4x1vl_rvv<MR>(a_ptr + 3 * MR, 1, a_ptrs);
        tqt_vmacc_kernel_gemm_i8_i2_4x1vl_rvv<MR>(vl, a_ptrs);

        a_ptr += 4 * MR;  // 4 K steps × MR bytes per step (interleaved layout)
        b_ptr += vl;      // packed B layout has `actual_n` bytes per K-byte for this N-tile
    }
}

template <size_t MR, bool USE_DEFAULT_LUT>
static void gemm_kernel(size_t K, float *D, const int8_t *A, const uint8_t *B, const float *bias,
                        size_t ldd, float clamp_min, float clamp_max, bool apply_clamp,
                        size_t actual_n, const int32_t *neg_zp_a, const float *scale_a,
                        const float *scale_b, const int32_t *sum_b, const int8_t *lut)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");

    if constexpr (!USE_DEFAULT_LUT) {
        tqt_load_custom_lut_kernel_gemm_i8_i2_4x1vl_rvv(actual_n, lut);
    } else {
        TQT_UNUSED(lut);
    }

    tqt_init_zero_kernel_gemm_i8_i2_4x1vl_rvv<MR>(actual_n);

    gemm_kernel_kloop<MR, USE_DEFAULT_LUT>(K, A, B, actual_n);

    tqt_apply_zp_correction_kernel_gemm_i8_i2_4x1vl_rvv<MR>(actual_n, sum_b, neg_zp_a);
    tqt_dequant_to_f32_kernel_gemm_i8_i2_4x1vl_rvv<MR>(actual_n, scale_a, scale_b);

    if (bias) {
        tqt_add_1xnbias_f32_kernel_gemm_i8_i2_4x1vl_rvv<MR>(actual_n, bias);
    }
    // Skip min/max only for the explicitly unbounded range.
    if (apply_clamp) {
        tqt_clamp_f32_kernel_gemm_i8_i2_4x1vl_rvv<MR>(actual_n, clamp_min, clamp_max);
    }
    tqt_store_f32_kernel_gemm_i8_i2_4x1vl_rvv<MR>(actual_n, D, ldd);
}

using gemm_kernel_fn_t = void (*)(size_t K, float *D, const int8_t *A, const uint8_t *B,
                                  const float *bias, size_t ldd, float clamp_min, float clamp_max,
                                  bool apply_clamp, size_t actual_n, const int32_t *neg_zp_a,
                                  const float *scale_a, const float *scale_b, const int32_t *sum_b,
                                  const int8_t *lut);

// kernel_table[mr_idx][lut_path]
static constexpr gemm_kernel_fn_t kernel_table[4][2] = {
    {gemm_kernel<1, true>, gemm_kernel<1, false>},
    {gemm_kernel<2, true>, gemm_kernel<2, false>},
    {gemm_kernel<3, true>, gemm_kernel<3, false>},
    {gemm_kernel<4, true>, gemm_kernel<4, false>},
};

// ============================================================================
// Gemm Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, void *D, size_t ldd, const void *bias,
    float clamp_min, float clamp_max, const int8_t *lut,
    const struct tqt_qai8dx_qsi2cx_f32_params *params)
{
    const size_t VLENB = get_vlenb();
    const size_t n_step = VLENB;

    const bool apply_clamp = tqt_clamp_is_enabled((float)clamp_min, (float)clamp_max);

    if (!params) {
        return;
    }
    TQT_ASSUME(k % 4 == 0);
    TQT_ASSUME(k_idx_a % 4 == 0);
    TQT_ASSUME(k_idx_b % 4 == 0);
    TQT_ASSUME(ldb % 4 == 0);

    const int8_t *A_ptr = (const int8_t *)A_packed;
    const uint8_t *B_ptr = (const uint8_t *)B_packed;
    const float *bias_ptr = (const float *)bias;
    float *D_ptr = (float *)D;

    const bool use_default_lut = (lut == nullptr);
    const size_t ldb_bytes = ldb / 4;

    // Precompute -zp_a[m] for the whole M range.
    int32_t neg_zp_a_full[m];
    for (size_t i = 0; i < m; ++i) {
        neg_zp_a_full[i] = -params->zero_point_a[i];
    }

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const int8_t *a_ptr =
                A_ptr + m_idx * lda + k_idx_a * actual_m - k_idx_a * TQT_MIN(m, m_step);
            const uint8_t *b_ptr = B_ptr + n_idx * ldb_bytes + (k_idx_b / 4) * actual_n -
                                   (k_idx_b / 4) * TQT_MIN(n, n_step);

            const float *bias_tile_ptr = bias_ptr ? bias_ptr + n_idx : nullptr;
            float *d_ptr = D_ptr + m_idx * ldd + n_idx;

            const int32_t *neg_zp_tile = neg_zp_a_full + m_idx;
            const float *scale_a_tile = params->scale_a + m_idx;
            const float *scale_b_tile = params->scale_b + n_idx;
            const int32_t *sum_b_tile = params->sum_b + n_idx;

            const size_t mr_idx = actual_m - 1;
            const size_t lut_idx = use_default_lut ? 0 : 1;
            kernel_table[mr_idx][lut_idx](k, d_ptr, a_ptr, b_ptr, bias_tile_ptr, ldd, clamp_min,
                                          clamp_max, apply_clamp, actual_n, neg_zp_tile,
                                          scale_a_tile, scale_b_tile, sum_b_tile, lut);
        }
    }
}
