//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv.h"

#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_gemm_pack_rvv.h"
#include "src/gemm/gemm_i8_i3/tqt_kernel_gemm_qsi8d256_qsi3c16_4x1vl_rvv.h"

static const size_t m_step = 4;

static inline size_t qsi3_bytes(size_t k)
{
    return (k / 16) * 6;
}

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(void)
{
    return m_step;
}
size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(void)
{
    const size_t n_step = tqt_get_vlenb();

    return n_step;
}
size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t m_idx,
                                                                                   size_t k_idx,
                                                                                   size_t lda,
                                                                                   size_t actual_m)
{
    return (m_idx * lda + k_idx * TQT_MIN(actual_m, m_step)) * sizeof(int8_t);
}
size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t n_idx,
                                                                                   size_t k_idx,
                                                                                   size_t ldb,
                                                                                   size_t actual_n)
{
    const size_t n_step = tqt_get_vlenb();

    return n_idx * qsi3_bytes(ldb) + qsi3_bytes(k_idx) * TQT_MIN(actual_n, n_step);
}
size_t tqt_get_scale_a_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t m_idx,
                                                                                  size_t block_idx,
                                                                                  size_t ldsa)
{
    return (m_idx * ldsa + block_idx) * sizeof(float);
}
size_t tqt_get_qscale_b_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t n_idx,
                                                                                   size_t block_idx,
                                                                                   size_t ldqb)
{
    return (n_idx * ldqb + block_idx) * sizeof(int8_t);
}
size_t tqt_get_super_scale_b_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(
    size_t n_idx, size_t block_idx, size_t ldssb)
{
    return (n_idx * ldssb + block_idx) * sizeof(float);
}
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t m_idx,
                                                                            size_t n_idx,
                                                                            size_t ldc)
{
    return (m_idx * ldc + n_idx) * sizeof(float);
}
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t n_idx)
{
    return n_idx * sizeof(float);
}
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t m_idx,
                                                                            size_t n_idx,
                                                                            size_t ldd)
{
    return (m_idx * ldd + n_idx) * sizeof(float);
}
size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t m, size_t n)
{
    return m * n * sizeof(float);
}
size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t m, size_t k)
{
    return m * k;
}
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(size_t n, size_t k)
{
    return n * qsi3_bytes(k);
}
void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed)
{
    tqt_gemm_pack_mxk_4x1_e8_rvv(m, k, lda, lda_packed, k_idx, A, A_packed);
}
void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed)
{
    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, qsi3_bytes(k), qsi3_bytes(ldb), qsi3_bytes(ldb_packed),
                                   qsi3_bytes(k_idx), B, B_packed);
}
void tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qsi8d256_qsi3c16_params *params)
{
    const size_t n_step = tqt_get_vlenb();

    const bool apply_clamp = tqt_clamp_is_enabled((float)clamp_min, (float)clamp_max);

    TQT_ASSUME(k % 256 == 0);
    TQT_ASSUME(k_idx_a == k_idx_b);
    TQT_ASSUME(k_idx_a % 256 == 0);
    if (!params)
        return;

    const int8_t *a_base = (const int8_t *)A_packed;
    const uint8_t *b_base = (const uint8_t *)B_packed;
    const size_t ldb_bytes = qsi3_bytes(ldb);
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t mr = TQT_MIN(m - m_idx, m_step);
        const int8_t *a_tile = a_base + m_idx * lda;
        if (m_idx != 0 && k_idx_a != 0 && mr != m_step)
            a_tile -= k_idx_a * (m_step - mr);
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t nr = TQT_MIN(n - n_idx, n_step);
            const uint8_t *b_tile = b_base + n_idx * ldb_bytes;
            if (n_idx != 0 && k_idx_b != 0 && nr != n_step)
                b_tile -= qsi3_bytes(k_idx_b) * (n_step - nr);
            tqt_compute_tile_kernel_gemm_qsi8d256_qsi3c16_4x1vl_rvv(
                mr, nr, k, k_idx_a, (float *)D + m_idx * ldd + n_idx, a_tile, b_tile,
                params->scale_a + m_idx * params->ldsa, params->ldsa,
                params->qscale_b + n_idx * params->ldqb, params->ldqb,
                params->super_scale_b + n_idx * params->ldssb, params->ldssb,
                C ? (const float *)C + m_idx * ldc + n_idx : nullptr,
                bias ? (const float *)bias + n_idx : nullptr, ldc, ldd, clamp_min, clamp_max,
                apply_clamp);
        }
    }
}
