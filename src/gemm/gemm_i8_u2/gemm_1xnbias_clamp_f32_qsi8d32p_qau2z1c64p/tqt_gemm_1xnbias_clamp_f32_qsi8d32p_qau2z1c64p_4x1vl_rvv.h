//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/gemm/gemm_i8_u2/tqt_params_i8_u2.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(void);
size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(void);
size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);
size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
size_t tqt_get_scale_a_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(size_t m_idx,
                                                                                   size_t block_idx,
                                                                                   size_t ldsa);
size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(size_t n_idx,
                                                                                   size_t block_idx,
                                                                                   size_t ldsb);
size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldc);
size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldd);
size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(size_t m, size_t n);
size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(size_t m,
                                                                                  size_t k);
size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(size_t n,
                                                                                  size_t k);
void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(
    size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx, const void *A, void *A_packed);
void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);
void tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qsi8d32_qau2z1c64_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif
