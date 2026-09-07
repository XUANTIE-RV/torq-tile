//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"

struct tqt_u8_u8_clamp_f32_params;

#ifdef __cplusplus
extern "C" {
#endif

size_t tqt_get_m_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(void);
size_t tqt_get_n_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(void);

size_t tqt_get_a_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t m_idx,
                                                                                size_t k_idx,
                                                                                size_t lda,
                                                                                size_t actual_m);
size_t tqt_get_b_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t n_idx,
                                                                                size_t k_idx,
                                                                                size_t ldb,
                                                                                size_t actual_n);
size_t tqt_get_c_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldc);
size_t tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t m_idx);
size_t tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t n_idx);
size_t tqt_get_dst_bias_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldd);
size_t tqt_get_d_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t m, size_t n);
size_t tqt_get_a_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t m, size_t k);
size_t tqt_get_b_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t n, size_t k);

void tqt_run_a_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t m, size_t k, size_t lda,
                                                                     size_t lda_packed,
                                                                     size_t k_idx, const void *A,
                                                                     void *A_packed);
void tqt_run_b_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t n, size_t k, size_t ldb,
                                                                     size_t ldb_packed,
                                                                     size_t k_idx, const void *B,
                                                                     void *B_packed);
void tqt_run_bt_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(size_t n, size_t k,
                                                                      size_t ldb, size_t ldb_packed,
                                                                      size_t k_idx, const void *B,
                                                                      void *B_packed);

/// D[m,n] = clamp(scvtf(C[m,n] + bias_m[m] + bias_n[n]
///                       + sum_k u8_A_packed * u8_B_packed) * scale
///                + dst_bias[n], min, max)
///
/// A_packed is [M,K] packed uint8. B_packed is [N,K] packed uint8.
/// C is optional [M,N] int32 residual (nullptr to skip). D is [M,N] float output.
/// bias_m is int32[actual_m]. bias_n is int32[actual_n].
/// dst_bias is optional float[actual_n].
/// @note If clamp_min is negative infinity and clamp_max is positive infinity,
///       clamping is skipped. All other bounds retain the existing clamp behavior.
///       This does not guarantee preservation of NaN payloads or exception flags.
void tqt_run_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias_m, const void *bias_n, const void *dst_bias, float clamp_min,
    float clamp_max, const struct tqt_u8_u8_clamp_f32_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif
