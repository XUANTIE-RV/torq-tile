//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i2/tqt_params_i8_i2.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel: GEMM with 1xN bias and clamp for qai8dx (A row-major int8) x qsi2cxt
/// (B transposed row-major int2, [N, K/4] uint8 bytes, 4 weights packed `s3 s2 s1 s0`)
/// -> f32 (D).
///
/// 4 x vl tile, RVV. Same compute semantics as the qai8dxp x qsi2cxp variant; the only
/// differences are that A and B are not pre-packed (raw layouts) and the K loop uses
/// strided loads. K must be a multiple of 4. The default LUT is {-2, -1, 0, 1}; pass nullptr
/// for `lut` to use the default.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv(void);
size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv(void);

/// A is [M, K] int8 row-major. lda is in K elements.
size_t tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                        size_t lda);

/// B is [N, K/4] uint8 row-major (transposed). ldb is in K elements (multiple of 4).
size_t tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                        size_t ldb);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldd);
size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv(size_t m, size_t n);

/// Runs the GEMM micro-kernel. See the qai8dxp_qsi2cxp variant for full parameter docs.
void tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, void *D, size_t ldd, const void *bias, float clamp_min,
    float clamp_max, const int8_t *lut, const struct tqt_qai8dx_qsi2cx_f32_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
