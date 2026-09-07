//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zvfbfwma)
#error This file must be compiled for riscv, riscv_vector, zvfbfwma.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv.h"

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_i8_i4/tqt_kernel_gemm_i8_i4_4x1vl_blockquant_rvv.h"

static const size_t VLENB = csrr_vlenb();
static const size_t vlmax = VLENB;
static const size_t m_step = 4;
static const size_t n_step = vlmax;

// ============================================================================
// Helper Functions (Get Methods)
// ============================================================================

size_t tqt_get_m_step_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(void)
{
    return n_step;
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m_idx,
                                                                                 size_t k_idx,
                                                                                 size_t lda,
                                                                                 size_t actual_m)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(int8_t);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t n_idx,
                                                                                 size_t k_idx,
                                                                                 size_t ldb,
                                                                                 size_t actual_n)
{
    TQT_ASSUME(n_idx % n_step == 0);
    TQT_ASSUME(k_idx % 2 == 0);
    const size_t ldb_e8 = ldb / 2;
    return n_idx * ldb_e8 + (k_idx / 2) * TQT_MIN(actual_n, n_step);
}

size_t tqt_get_scale_a_offset_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m_idx,
                                                                                size_t b_idx,
                                                                                size_t ldsa)
{
    TQT_ASSUME(m_idx % m_step == 0);
    return (m_idx * ldsa + b_idx) * sizeof(float);
}

size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t n_idx,
                                                                                size_t b_idx,
                                                                                size_t ldsb)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return (n_idx * ldsb + b_idx) * sizeof(float);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m_idx,
                                                                          size_t n_idx, size_t ldc)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldc + n_idx) * sizeof(bfloat16_t);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t n_idx)
{
    TQT_ASSUME(n_idx % n_step == 0);
    return n_idx * sizeof(bfloat16_t);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m_idx,
                                                                          size_t n_idx, size_t ldd)
{
    TQT_ASSUME(m_idx % m_step == 0);
    TQT_ASSUME(n_idx % n_step == 0);
    return (m_idx * ldd + n_idx) * sizeof(bfloat16_t);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(bfloat16_t);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m, size_t k)
{
    return m * k * sizeof(int8_t);
}

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t n, size_t k)
{
    return n * k / 2;
}

// ============================================================================
// Pack Functions
// ============================================================================

void tqt_run_a_pack_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(size_t m, size_t k,
                                                                      size_t lda, size_t lda_packed,
                                                                      size_t k_idx, const void *A,
                                                                      void *A_packed)
{
    tqt_gemm_pack_mxk_4x1_e8_rvv(m, k, lda, lda_packed, k_idx, A, A_packed);
}

void tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed)
{
    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, k / 2, ldb / 2, ldb_packed / 2, k_idx / 2, B, B_packed);
}

// ============================================================================
// Template GEMM kernel (packed A + packed B)
// ============================================================================

template <size_t MR>
static void gemm_kernel(size_t K, size_t bl, bfloat16_t *D, const int8_t *A_packed,
                        const uint8_t *B_packed, const float *scale_a, const float *scale_b,
                        const bfloat16_t *C, const bfloat16_t *bias, size_t ldd, size_t lda,
                        size_t ldb, size_t ldsa, size_t ldsb, size_t ldc, bfloat16_t clamp_min,
                        bfloat16_t clamp_max, bool apply_clamp, size_t vl)
{
    static_assert(MR >= 1 && MR <= 4, "MR must be in range [1, 4]");
    TQT_UNUSED(lda);
    TQT_UNUSED(ldb);

    const size_t scale_byte_stride = ldsb * sizeof(float);
    const size_t d_stride_row = ldd;
    const size_t c_stride_row = ldc;

    tqt_init_fzero_acc_kernel_bq<MR>(vl);

    const size_t num_blocks = K / bl;
    for (size_t blk = 0; blk < num_blocks; ++blk) {
        tqt_load_scale_b_block_kernel_bq(vl, scale_b + blk, scale_byte_stride);

        float sa_blk[MR];
        if constexpr (MR >= 1)
            sa_blk[0] = scale_a[0 * ldsa + blk];
        if constexpr (MR >= 2)
            sa_blk[1] = scale_a[1 * ldsa + blk];
        if constexpr (MR >= 3)
            sa_blk[2] = scale_a[2 * ldsa + blk];
        if constexpr (MR >= 4)
            sa_blk[3] = scale_a[3 * ldsa + blk];

        const size_t k_start = blk * bl;
        const size_t k_end = k_start + bl;
        for (size_t k = k_start; k < k_end; k += 2) {
            const size_t kp = k / 2;
            const uint8_t *b_col = B_packed + kp * vl;
            const int8_t *a_lo_base = A_packed + k * MR;
            const int8_t *a_hi_base = A_packed + (k + 1) * MR;

            // Lo nibble (k+0)
            float a_lo[MR];
            if constexpr (MR >= 1)
                a_lo[0] = (float)a_lo_base[0] * sa_blk[0];
            if constexpr (MR >= 2)
                a_lo[1] = (float)a_lo_base[1] * sa_blk[1];
            if constexpr (MR >= 3)
                a_lo[2] = (float)a_lo_base[2] * sa_blk[2];
            if constexpr (MR >= 4)
                a_lo[3] = (float)a_lo_base[3] * sa_blk[3];

            // Hi nibble (k+1)
            float a_hi[MR];
            if constexpr (MR >= 1)
                a_hi[0] = (float)a_hi_base[0] * sa_blk[0];
            if constexpr (MR >= 2)
                a_hi[1] = (float)a_hi_base[1] * sa_blk[1];
            if constexpr (MR >= 3)
                a_hi[2] = (float)a_hi_base[2] * sa_blk[2];
            if constexpr (MR >= 4)
                a_hi[3] = (float)a_hi_base[3] * sa_blk[3];
            tqt_dot_b_pair_fp32_qsi4c2_kernel_bq<MR>(vl, b_col, a_lo, a_hi);
        }
    }

    if (C) {
        tqt_add_c_bf16_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, C, c_stride_row);
    }

    if (bias) {
        tqt_add_1xnbias_bf16_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, bias);
    }

    const float min_f32 = static_cast<float>(clamp_min);
    const float max_f32 = static_cast<float>(clamp_max);
    // Skip min/max only for the explicitly unbounded range.
    if (apply_clamp) {
        tqt_clamp_f32_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, min_f32, max_f32);
    }

    tqt_store_narrow_bf16_kernel_gemm_i8_i4_4x1vl_rvv<MR>(vl, D, d_stride_row);
}

using gemm_kernel_fn_t = void (*)(size_t K, size_t bl, bfloat16_t *D, const int8_t *A_packed,
                                  const uint8_t *B_packed, const float *scale_a,
                                  const float *scale_b, const bfloat16_t *C, const bfloat16_t *bias,
                                  size_t ldd, size_t lda, size_t ldb, size_t ldsa, size_t ldsb,
                                  size_t ldc, bfloat16_t clamp_min, bfloat16_t clamp_max,
                                  bool apply_clamp, size_t vl);

static constexpr gemm_kernel_fn_t kernel_table[4] = {
    gemm_kernel<1>,
    gemm_kernel<2>,
    gemm_kernel<3>,
    gemm_kernel<4>,
};

// ============================================================================
// Main Function
// ============================================================================

void tqt_run_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, bfloat16_t clamp_min, bfloat16_t clamp_max,
    const struct tqt_qsi8d2_qsi4c2_params *params)
{
    const bool apply_clamp = tqt_gemm_clamp_is_enabled((float)clamp_min, (float)clamp_max);

    if (!params) {
        return;
    }

    const int8_t *A_ptr = (const int8_t *)A_packed;
    const uint8_t *B_ptr = (const uint8_t *)B_packed;
    const bfloat16_t *C_ptr = (const bfloat16_t *)C;
    const bfloat16_t *bias_ptr = (const bfloat16_t *)bias;
    bfloat16_t *D_ptr = (bfloat16_t *)D;

    TQT_ASSUME(k_idx_a % bl == 0);
    TQT_ASSUME(k_idx_b % bl == 0);
    TQT_ASSUME(k_idx_b % 2 == 0);
    const size_t ldb_e8 = ldb / 2;
    const size_t origin_m = TQT_MIN(m, m_step);
    const size_t origin_n = TQT_MIN(n, n_step);
    const size_t ldsa = params->ldsa;
    const size_t ldsb = params->ldsb;

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = TQT_MIN(m - m_idx, m_step);
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);

            const int8_t *a_tile = A_ptr + m_idx * lda + k_idx_a * actual_m - k_idx_a * origin_m;
            const uint8_t *b_tile =
                B_ptr + n_idx * ldb_e8 + (k_idx_b / 2) * actual_n - (k_idx_b / 2) * origin_n;
            const bfloat16_t *c_tile = C_ptr ? C_ptr + m_idx * ldc + n_idx : nullptr;
            const bfloat16_t *bias_tile = bias_ptr ? bias_ptr + n_idx : nullptr;
            bfloat16_t *d_tile = D_ptr + m_idx * ldd + n_idx;

            const float *scale_a_tile = params->scale_a + m_idx * ldsa + k_idx_a / bl;
            const float *scale_b_tile = params->scale_b + n_idx * ldsb + k_idx_b / bl;

            const size_t mr_idx = actual_m - 1;
            kernel_table[mr_idx](k, bl, d_tile, a_tile, b_tile, scale_a_tile, scale_b_tile, c_tile,
                                 bias_tile, ldd, lda, ldb, ldsa, ldsb, ldc, clamp_min, clamp_max,
                                 apply_clamp, actual_n);
        }
    }
}
