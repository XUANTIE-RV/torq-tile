//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include "tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv.h"

#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_i8_i5/tqt_kernel_gemm_qsi8d32_qsi5c32_4x1vl_rvv.h"

static const size_t m_step = 4;

static inline size_t tqt_qsi5c32_bytes(size_t logical_k)
{
    return (logical_k / 32) * 20;
}

#define TQT_DEFINE_QSI5C32_GETTERS(BASE)                                                   \
    size_t tqt_get_m_step_##BASE(void)                                                     \
    {                                                                                      \
        return m_step;                                                                     \
    }                                                                                      \
    size_t tqt_get_n_step_##BASE(void)                                                     \
    {                                                                                      \
        const size_t n_step = tqt_get_vlenb();                                             \
        return n_step;                                                                     \
    }                                                                                      \
    size_t tqt_get_a_packed_offset_##BASE(size_t m, size_t k, size_t ld, size_t actual)    \
    {                                                                                      \
        return m * ld + k * TQT_MIN(actual, m_step);                                       \
    }                                                                                      \
    size_t tqt_get_b_packed_offset_##BASE(size_t n, size_t k, size_t ld, size_t actual)    \
    {                                                                                      \
        const size_t n_step = tqt_get_vlenb();                                             \
        return n * tqt_qsi5c32_bytes(ld) + tqt_qsi5c32_bytes(k) * TQT_MIN(actual, n_step); \
    }                                                                                      \
    size_t tqt_get_scale_a_offset_##BASE(size_t m, size_t block, size_t ld)                \
    {                                                                                      \
        return (m * ld + block) * sizeof(float);                                           \
    }                                                                                      \
    size_t tqt_get_scale_b_offset_##BASE(size_t n, size_t block, size_t ld)                \
    {                                                                                      \
        return (n * ld + block) * sizeof(float);                                           \
    }                                                                                      \
    size_t tqt_get_c_offset_##BASE(size_t m, size_t n, size_t ld)                          \
    {                                                                                      \
        return (m * ld + n) * sizeof(float);                                               \
    }                                                                                      \
    size_t tqt_get_bias_offset_##BASE(size_t n)                                            \
    {                                                                                      \
        return n * sizeof(float);                                                          \
    }                                                                                      \
    size_t tqt_get_d_offset_##BASE(size_t m, size_t n, size_t ld)                          \
    {                                                                                      \
        return (m * ld + n) * sizeof(float);                                               \
    }

TQT_DEFINE_QSI5C32_GETTERS(gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv)
#undef TQT_DEFINE_QSI5C32_GETTERS

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}
size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(size_t m, size_t k)
{
    return m * k;
}
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(size_t n, size_t k)
{
    return n * tqt_qsi5c32_bytes(k);
}
void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed)
{
    tqt_gemm_pack_mxk_4x1_e8_rvv(m, k, lda, lda_packed, k_idx, A, A_packed);
}
void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed)
{
    TQT_ASSUME(k % 32 == 0);
    TQT_ASSUME(ldb % 32 == 0);
    TQT_ASSUME(ldb_packed % 32 == 0);
    TQT_ASSUME(k_idx % 32 == 0);

    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, tqt_qsi5c32_bytes(k), tqt_qsi5c32_bytes(ldb),
                                   tqt_qsi5c32_bytes(ldb_packed), tqt_qsi5c32_bytes(k_idx), B,
                                   B_packed);
}
void tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qsi8d32_qsi5c32_params *params)
{
    const size_t n_step = tqt_get_vlenb();

    const bool apply_clamp = tqt_clamp_is_enabled((float)clamp_min, (float)clamp_max);

    TQT_ASSUME(k % 32 == 0);
    TQT_ASSUME(lda % 32 == 0);
    TQT_ASSUME(ldb % 32 == 0);
    TQT_ASSUME(k_idx_a == k_idx_b);
    TQT_ASSUME(k_idx_a % 32 == 0);
    if (params == nullptr)
        return;

    const int8_t *a_base = (const int8_t *)A_packed;
    const uint8_t *b_base = (const uint8_t *)B_packed;
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t mr = TQT_MIN(m - m_idx, m_step);
        const int8_t *a_tile = a_base + m_idx * lda;
        if (m_idx != 0 && k_idx_a != 0 && mr != m_step)
            a_tile -= k_idx_a * (m_step - mr);
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t nr = TQT_MIN(n - n_idx, n_step);
            const uint8_t *b_tile = b_base + n_idx * tqt_qsi5c32_bytes(ldb);
            if (n_idx != 0 && k_idx_b != 0 && nr != n_step)
                b_tile -= tqt_qsi5c32_bytes(k_idx_b) * (n_step - nr);
            tqt_compute_tile_kernel_gemm_qsi8d32_qsi5c32_4x1vl_rvv(
                mr, nr, k, k_idx_a, (float *)D + m_idx * ldd + n_idx, a_tile, b_tile,
                params->scale_a + m_idx * params->ldsa, params->ldsa,
                params->scale_b + n_idx * params->ldsb, params->ldsb,
                C ? (const float *)C + m_idx * ldc + n_idx : nullptr,
                bias ? (const float *)bias + n_idx : nullptr, ldc, ldd, clamp_min, clamp_max,
                apply_clamp);
        }
    }
}
