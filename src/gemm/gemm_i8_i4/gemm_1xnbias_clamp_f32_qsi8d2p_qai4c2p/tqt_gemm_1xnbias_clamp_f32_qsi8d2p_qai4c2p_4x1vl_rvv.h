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

/// Micro-kernel: packed W4A8 block-quantized GEMM with 1xN bias and clamp
/// (D = f32 output).
///
/// rvv (4 x vl tile) implementation where vl = csrr_vlenb()
/// (i8 elements per vector at e8/m1).
/// Computes:
///   D = clamp(C + dequant(A_qsi8d2p) * dequant(B_qai4c2p) + bias, min, max)
/// where:
///   - A is qsi8d2p: per-row block-quantized symmetric int8 along K, packed
///     in [m/m_step, k, m_step] interleaved layout.
///   - B is qai4c2p: per-block asymmetric int4 along K, packed into
///     [n/n_step, k/2, n_step] uint8 layout (two nibbles per byte using ZP=0
///     unsigned encoding; zero point is carried in zp_b as fp32).
///   - bias is f32[n], C is f32[m, n], D is f32[m, n].
///
/// Algorithm matches the unpacked variant; only the data layouts differ.
/// k must be a multiple of bl. bl must be >= 2 and a multiple of 2.

size_t tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(void);

size_t tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(void);

size_t tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t m_idx,
                                                                                size_t k_idx,
                                                                                size_t lda,
                                                                                size_t actual_m);

size_t tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t n_idx,
                                                                                size_t k_idx,
                                                                                size_t ldb,
                                                                                size_t actual_n);

size_t tqt_get_scale_a_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t m_idx,
                                                                               size_t b_idx,
                                                                               size_t ldsa);

size_t tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t n_idx,
                                                                               size_t b_idx,
                                                                               size_t ldsb);

size_t tqt_get_zp_b_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t n_idx,
                                                                            size_t b_idx,
                                                                            size_t ldzpb);

size_t tqt_get_c_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldc);

size_t tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t n_idx);

size_t tqt_get_d_offset_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t m_idx, size_t n_idx,
                                                                         size_t ldd);

size_t tqt_get_d_size_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t m, size_t n);

size_t tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t m, size_t k);

size_t tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t n, size_t k);

void tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t m, size_t k, size_t lda,
                                                                     size_t lda_packed,
                                                                     size_t k_idx, const void *A,
                                                                     void *A_packed);

void tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(size_t n, size_t k,
                                                                      size_t ldb, size_t ldb_packed,
                                                                      size_t k_idx, const void *B,
                                                                      void *B_packed);

/// @note If clamp_min is negative infinity and clamp_max is positive infinity,
///       clamping is skipped. All other bounds retain the existing clamp behavior.
///       This does not guarantee preservation of NaN payloads or exception flags.
void tqt_run_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv(
    size_t m, size_t n, size_t k, size_t bl, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max,
    const struct tqt_qsi8d2_qai4c2_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
