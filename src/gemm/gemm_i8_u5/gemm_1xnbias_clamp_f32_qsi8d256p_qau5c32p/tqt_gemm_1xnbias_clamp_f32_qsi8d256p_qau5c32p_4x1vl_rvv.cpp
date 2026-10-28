//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv.h"

#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_i8_u5/tqt_kernel_gemm_qsi8d256_qau5c32_4x1vl_rvv.h"

static inline size_t get_vlenb()
{
    static const size_t value = csrr_vlenb();
    return value;
}

static const size_t m_step = 4;

static inline size_t q5_bytes(size_t logical_k)
{
    return (logical_k / 32) * 20;
}

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(void)
{
    return m_step;
}

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(void)
{
    const size_t n_step = get_vlenb();

    return n_step;
}

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t m_idx,
                                                                                   size_t k_idx,
                                                                                   size_t lda,
                                                                                   size_t actual_m)
{
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(int8_t);
}

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t n_idx,
                                                                                   size_t k_idx,
                                                                                   size_t ldb,
                                                                                   size_t actual_n)
{
    const size_t n_step = get_vlenb();

    return n_idx * q5_bytes(ldb) + q5_bytes(k_idx) * TQT_MIN(actual_n, n_step);
}

size_t tqt_get_scale_a_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t m_idx,
                                                                                  size_t block_idx,
                                                                                  size_t ldsa)
{
    return (m_idx * ldsa + block_idx) * sizeof(float);
}

size_t tqt_get_bsums_a_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t m_idx,
                                                                                  size_t block_idx,
                                                                                  size_t ldbsuma)
{
    return (m_idx * ldbsuma + block_idx) * sizeof(int16_t);
}

size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t n_idx,
                                                                                  size_t block_idx,
                                                                                  size_t ldsb)
{
    return (n_idx * ldsb + block_idx) * sizeof(float);
}

size_t tqt_get_real_zero_b_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(
    size_t n_idx, size_t block_idx, size_t ldsb)
{
    return (n_idx * ldsb + block_idx) * sizeof(float);
}

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t m_idx,
                                                                            size_t n_idx,
                                                                            size_t ldc)
{
    return (m_idx * ldc + n_idx) * sizeof(float);
}

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t n_idx)
{
    return n_idx * sizeof(float);
}

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t m_idx,
                                                                            size_t n_idx,
                                                                            size_t ldd)
{
    return (m_idx * ldd + n_idx) * sizeof(float);
}

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t m, size_t k)
{
    return m * k;
}

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(size_t n, size_t k)
{
    return n * q5_bytes(k);
}

void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed)
{
    tqt_gemm_pack_mxk_4x1_e8_rvv(m, k, lda, lda_packed, k_idx, A, A_packed);
}

void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed)
{
    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, q5_bytes(k), q5_bytes(ldb), q5_bytes(ldb_packed),
                                   q5_bytes(k_idx), B, B_packed);
}

void tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qsi8d256_qau5c32_params *params)
{
    const size_t n_step = get_vlenb();

    const bool apply_clamp = tqt_clamp_is_enabled((float)clamp_min, (float)clamp_max);

    TQT_ASSUME(k % 256 == 0);
    TQT_ASSUME(k_idx_a == k_idx_b);
    TQT_ASSUME(k_idx_a % 256 == 0);
    if (!params)
        return;

    const int8_t *a_base = (const int8_t *)A_packed;
    const uint8_t *b_base = (const uint8_t *)B_packed;
    const size_t ldb_bytes = q5_bytes(ldb);
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t mr = TQT_MIN(m - m_idx, m_step);
        const int8_t *a_tile = a_base + m_idx * lda;
        if (m_idx != 0 && k_idx_a != 0 && mr != m_step)
            a_tile -= k_idx_a * (m_step - mr);
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t nr = TQT_MIN(n - n_idx, n_step);
            const uint8_t *b_tile = b_base + n_idx * ldb_bytes;
            if (n_idx != 0 && k_idx_b != 0 && nr != n_step)
                b_tile -= q5_bytes(k_idx_b) * (n_step - nr);
            tqt_compute_tile_kernel_gemm_qsi8d256_qau5c32_4x1vl_rvv(
                mr, nr, k, k_idx_a, (float *)D + m_idx * ldd + n_idx, a_tile, b_tile,
                params->scale_a + m_idx * params->ldsa, params->ldsa,
                params->bsums_a + m_idx * params->ldbsuma, params->ldbsuma,
                params->scale_b + n_idx * params->ldsb, params->real_zero_b + n_idx * params->ldsb,
                params->ldsb, C ? (const float *)C + m_idx * ldc + n_idx : nullptr,
                bias ? (const float *)bias + n_idx : nullptr, ldc, ldd, clamp_min, clamp_max,
                apply_clamp);
        }
    }
}
