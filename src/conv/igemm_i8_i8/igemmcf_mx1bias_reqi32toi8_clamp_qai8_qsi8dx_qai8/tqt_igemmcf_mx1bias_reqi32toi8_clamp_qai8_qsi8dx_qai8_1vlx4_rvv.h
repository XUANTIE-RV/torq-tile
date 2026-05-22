//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Implicit GEMM micro-kernel: channel-first, Mx1 bias, requantize i32->i8, clamp,
/// non-packed weight, quantized int8.
/// RVV 1vlx4 tile implementation.
///
/// Vectorization direction: M (OC) vectorized (1*vl), N (spatial) scalar (4).
/// A = weight [OC_per_group, K] row-major (non-packed),
/// input = [IC, ID, IH, IW] (fetched on-the-fly)
/// where K = IC_per_group * KD * KH * KW, N = OD * OH * OW
///
/// Quantization scheme:
/// - A (weight): symmetric per-channel quantization (scale_a[OC], quant_channel_a = OC)
/// - B (input):  asymmetric quantization (scale_b, zero_point_b)
/// - D (output): asymmetric quantization (scale_d, zero_point_d)

/// --------------------------------------------------

/// Gets m step value.
///
/// @return The m step value (vl, where vl = csrr_vlenb()).
size_t tqt_get_m_step_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(void);

/// Gets n step value.
///
/// @return The n step value (4).
size_t tqt_get_n_step_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(void);

/// Gets the offset in bytes to the data element in the A (weight) matrix buffer.
///
/// @param[in] m_idx Row index (OC offset, must be divisible by m_step).
/// @param[in] k_idx Column index (K offset).
/// @param[in] lda Leading dimension of A.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_offset_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(size_t m_idx,
                                                                                    size_t k_idx,
                                                                                    size_t lda);

/// Gets the offset in bytes to the data element in the C (accumulation) buffer.
///
/// @param[in] m_idx Row index (OC offset, must be divisible by m_step).
/// @param[in] n_idx Column index (spatial offset).
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(size_t m_idx,
                                                                                    size_t n_idx,
                                                                                    size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
///
/// @param[in] m_idx Row index (OC offset, must be divisible by m_step).
///
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(
    size_t m_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index (OC offset, must be divisible by m_step).
/// @param[in] n_idx Column index (spatial offset).
/// @param[in] ldd Leading dimension of D (= OD * OH * OW).
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(size_t m_idx,
                                                                                    size_t n_idx,
                                                                                    size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows (OC).
/// @param[in] n Number of columns (OD * OH * OW).
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(size_t m,
                                                                                  size_t n);

/// Runs the implicit GEMM convolution micro-kernel (channel-first, non-packed weight, int8).
///
/// Fuses im2col + GEMM: fetches input elements on-the-fly from the original
/// input tensor using conv_params. M and N tiling handled internally.
/// No K-tiling: full K is always processed in one call.
///
/// @param[in]  m     Number of output channels to process.
/// @param[in]  n     Number of spatial positions to process.
/// @param[in]  k     K chunk length (IC_per_group * KD * KH * KW for full K).
/// @param[in]  input Original input tensor [IC, ID, IH, IW] (int8, NOT im2col).
/// @param[in]  group_idx Group index for grouped convolution.
/// @param[in]  n_idx Starting spatial position (linear index into OD*OH*OW).
/// @param[in]  k_idx_b  K offset for input addressing (unused, must be 0).
/// @param[in]  A     Weight matrix [OC, K] in row-major format (int8).
/// @param[in]  lda   Leading dimension of A (= K).
/// @param[in]  k_idx_a  K offset in weight (unused, must be 0).
/// @param[in]  C     Accumulation buffer (int32) or NULL (unused for i8 igemm).
/// @param[in]  ldc   Leading dimension of C.
/// @param[out] D     Output buffer [OC, OD, OH, OW] (int8).
/// @param[in]  ldd   Leading dimension of D (= OD * OH * OW).
/// @param[in]  bias  Bias vector [OC] (int32) or NULL.
/// @param[in]  clamp_min Minimum clamp value (int8).
/// @param[in]  clamp_max Maximum clamp value (int8).
/// @param[in]  conv_params Convolution parameters (shapes, strides, padding, etc.).
/// @param[in]  quant_params Quantization parameters.
void tqt_run_igemm_igemmcf_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_1vlx4_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t n_idx, size_t k_idx_b,
    const void *A, size_t lda, size_t k_idx_a, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, int8_t clamp_min, int8_t clamp_max, const tqt_conv_params *conv_params,
    const struct tqt_qai8_qsi8dx_qai8_params *quant_params);

#ifdef __cplusplus
}  // extern "C"
#endif
