//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Implicit GEMM micro-kernel: channel-first, Mx1 bias, clamp, non-packed weight, FP16
/// RVV 3vlx8 tile implementation.
///
/// Vectorization direction: M (OC) vectorized (3*vl), N (spatial) scalar (8).
/// A = weight [OC, K], input = [IC, ID, IH, IW] (fetched on-the-fly)
/// where K = IC_per_group * KD * KH * KW, N = OD * OH * OW

/// --------------------------------------------------

/// Gets m step value.
///
/// @return The m step value (3 * vl, where vl = csrr_vlenb() / sizeof(float16_t)).
size_t tqt_get_m_step_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv(void);

/// Gets n step value.
///
/// @return The n step value (8).
size_t tqt_get_n_step_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv(void);

/// Gets the offset in bytes to the data element in the A (weight) matrix buffer.
///
/// @param[in] m_idx Row index (OC offset, must be divisible by m_step).
/// @param[in] k_idx Column index (K offset).
/// @param[in] lda Leading dimension of A.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_offset_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv(size_t m_idx, size_t k_idx,
                                                                    size_t lda);

/// Gets the offset in bytes to the data element in the C (accumulation) buffer.
///
/// @param[in] m_idx Row index (OC offset, must be divisible by m_step).
/// @param[in] n_idx Column index (spatial offset).
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
///
/// @param[in] m_idx Row index (OC offset, must be divisible by m_step).
///
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv(size_t m_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index (OC offset, must be divisible by m_step).
/// @param[in] n_idx Column index (spatial offset).
/// @param[in] ldd Leading dimension of D (= OD * OH * OW).
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows (OC).
/// @param[in] n Number of columns (OD * OH * OW).
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv(size_t m, size_t n);

/// Runs the implicit GEMM convolution micro-kernel.
///
/// Fuses im2col + GEMM: fetches input elements on-the-fly from the original
/// input tensor using conv_params. M and N tiling handled internally.
///
/// @param[in]  m     Number of output channels to process.
/// @param[in]  n     Number of spatial positions to process.
/// @param[in]  k     K chunk length (IC * KD * KH * KW for full K).
/// @param[in]  input Original input tensor [IC, ID, IH, IW] (NOT im2col).
/// @param[in]  group_idx Group index for grouped convolution.
/// @param[in]  n_idx Starting spatial position (linear index into OD*OH*OW).
/// @param[in]  k_idx_b  K offset for input addressing (must be aligned to KD*KH*KW).
/// @param[in]  A     Weight matrix [OC, K] in row-major format.
/// @param[in]  lda   Leading dimension of A (= K).
/// @param[in]  k_idx_a  K offset in weight (must be aligned to KD*KH*KW).
/// @param[in]  C     Accumulation buffer or NULL (used when k_idx_b > 0).
/// @param[in]  ldc   Leading dimension of C (= OD * OH * OW).
/// @param[out] D     Output buffer [OC, OD, OH, OW].
/// @param[in]  ldd   Leading dimension of D (= OD * OH * OW).
/// @param[in]  bias  Bias vector [OC] or NULL.
/// @param[in]  clamp_min Minimum clamp value (applied on last K chunk).
/// @param[in]  clamp_max Maximum clamp value (applied on last K chunk).
/// @param[in]  params Convolution parameters (kernel shape, stride, dilation, pad).
void tqt_run_igemm_igemmcf_mx1bias_clamp_f16_f16_f16_3vlx8_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t n_idx, size_t k_idx_b,
    const void *A, size_t lda, size_t k_idx_a, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, float16_t clamp_min, float16_t clamp_max, const tqt_conv_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif
