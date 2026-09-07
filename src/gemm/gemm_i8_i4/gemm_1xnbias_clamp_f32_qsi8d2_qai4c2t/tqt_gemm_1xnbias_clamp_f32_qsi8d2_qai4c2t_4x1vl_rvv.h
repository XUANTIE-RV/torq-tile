//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel: W4A8 block-quantized GEMM with 1xN bias and clamp (D = f32 output)
///
/// rvv (4 x vl tile) implementation where vl = csrr_vlenb()
/// (i8 elements per vector at e8/m1).
/// Computes:
///   D = clamp(C + dequant(A_qsi8d2) * dequant(B_qai4c2t) + bias, min, max)
/// where:
///   - A is qsi8d2: per-row block-quantized symmetric int8 along K,
///     [m, k] row-major. scale_a is [m, k/bl] fp32. No zero point.
///   - B is qai4c2t: per-block ASYMMETRIC int4 along K (block size = bl),
///     [n, k] transposed (K-contiguous), stored as [n, k/2] uint8 with two
///     nibbles per byte (lo = k+0, hi = k+1) using ZP=0 unsigned encoding
///     ([0, 15] stored verbatim). scale_b is [n, k/bl] fp32, zp_b is
///     [n, k/bl] fp32 (per-block zero point in fp domain).
///   - bias is f32[n] broadcast across rows (optional).
///   - C is f32[m, n] (optional, can be nullptr).
///   - D is f32[m, n].
///
/// Algorithm (per output cell):
///   acc = sum_blk(scale_a[m, blk] * scale_b[n, blk]
///                   * sum_{k in blk}(qa[m, k] * (qb[n, k] - zp_b[n, blk])))
///   D[m, n] = clamp(C[m, n] + acc + bias[n], min, max)
///
/// k must be a multiple of bl. bl must be >= 2 and a multiple of 2.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(void);

size_t tqt_get_a_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(size_t m_idx, size_t k_idx,
                                                                        size_t lda, size_t bl);

size_t tqt_get_b_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(size_t n_idx, size_t k_idx,
                                                                        size_t ldb, size_t bl);

size_t tqt_get_scale_a_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(size_t m_idx,
                                                                              size_t b_idx,
                                                                              size_t ldsa);

size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(size_t n_idx,
                                                                              size_t b_idx,
                                                                              size_t ldsb);

size_t tqt_get_zp_b_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(size_t n_idx,
                                                                           size_t b_idx,
                                                                           size_t ldzpb);

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                        size_t ldd);

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(size_t m, size_t n);

/// @note If clamp_min is negative infinity and clamp_max is positive infinity,
///       clamping is skipped. All other bounds retain the existing clamp behavior.
///       This does not guarantee preservation of NaN payloads or exception flags.
void tqt_run_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qsi8d2_qai4c2_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
