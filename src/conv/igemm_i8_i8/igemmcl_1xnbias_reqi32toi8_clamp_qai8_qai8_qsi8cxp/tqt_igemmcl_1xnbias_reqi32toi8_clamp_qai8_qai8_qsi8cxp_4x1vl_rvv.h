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

/// Implicit GEMM micro-kernel: channel-last, 1xN bias, requantize i32->i8, clamp,
/// packed weight, quantized int8.
/// RVV 4x1vl tile implementation.
///
/// Weight packed from [OC, K] to [OC/vl, K, vl] for contiguous vector loads.
///
/// Vectorization direction: N (OC) vectorized (1*vl), M (spatial) scalar (4).
///
/// Quantization scheme:
/// - A (input): asymmetric quantization (scale_a, zero_point_a)
/// - B (weight): symmetric per-channel quantization (scale_b[n], quant_channel_b = N)
/// - D (output): asymmetric quantization (scale_d, zero_point_d)

size_t tqt_get_m_step_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(void);
size_t tqt_get_n_step_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(void);

size_t tqt_get_b_packed_offset_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
size_t tqt_get_c_offset_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                     size_t n_idx,
                                                                                     size_t ldc);
size_t tqt_get_bias_offset_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
    size_t n_idx);
size_t tqt_get_d_offset_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                     size_t n_idx,
                                                                                     size_t ldd);
size_t tqt_get_d_size_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(size_t m,
                                                                                   size_t n);
size_t tqt_get_b_packed_size_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(size_t n,
                                                                                          size_t k);

/// Pack weight from [OC, K] to [OC/vl, K, vl] for contiguous vector loads.
///
/// @param[in]  n     Number of output channels.
/// @param[in]  k     K dimension (KD*KH*KW*IC).
/// @param[in]  ldw   Leading dimension of weight (= K).
/// @param[in]  ldw_packed Leading dimension of packed weight.
/// @param[in]  k_idx Starting K index for packing.
/// @param[in]  weight Input weight [OC, K] row-major (int8).
/// @param[out] weight_packed Packed weight buffer.
void tqt_run_bt_pack_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
    size_t n, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx, const void *weight,
    void *weight_packed);

/// Runs the implicit GEMM convolution micro-kernel (channel-last, packed weight, int8).
///
/// @param[in]  m     Number of spatial positions to process.
/// @param[in]  n     Number of output channels (OC) to process.
/// @param[in]  k     K chunk length (KD*KH*KW*IC for full K).
/// @param[in]  input Original input tensor [ID, IH, IW, IC] (int8).
/// @param[in]  group_idx Group index for grouped convolution.
/// @param[in]  m_idx Starting spatial position (linear index into OD*OH*OW).
/// @param[in]  k_idx_a  K offset for input addressing (must be aligned to IC).
/// @param[in]  B_packed Packed weight matrix (int8).
/// @param[in]  ldb   Leading dimension of B (= K).
/// @param[in]  k_idx_b  K offset in weight (must be aligned to IC).
/// @param[in]  C     Accumulation buffer (int32) or NULL (unused for i8 igemm).
/// @param[in]  ldc   Leading dimension of C.
/// @param[out] D     Output buffer [OD*OH*OW, OC] (int8).
/// @param[in]  ldd   Leading dimension of D (= OC).
/// @param[in]  bias  Bias vector [OC] (int32) or NULL.
/// @param[in]  clamp_min Minimum clamp value (int8).
/// @param[in]  clamp_max Maximum clamp value (int8).
/// @param[in]  conv_params Convolution parameters (shapes, strides, padding, etc.).
/// @param[in]  quant_params Quantization parameters.
void tqt_run_igemm_igemmcl_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t m_idx, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
    const tqt_conv_params *conv_params, const struct tqt_qai8_qai8_qsi8cx_params *quant_params);

#ifdef __cplusplus
}  // extern "C"
#endif
