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

/// Implicit GEMM micro-kernel: channel-last, 1xN bias, clamp, transposed weight, FP16
/// RVV 8x3vl tile implementation.
///
/// Vectorization direction: N (OC) vectorized (3*vl), M (spatial) scalar (8).
/// B = weight [OC, K] row-major (transposed), input = [ID, IH, IW, IC] (fetched on-the-fly)

size_t tqt_get_m_step_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(void);
size_t tqt_get_n_step_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(void);

size_t tqt_get_b_offset_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t n_idx, size_t k_idx,
                                                                     size_t ldb);
size_t tqt_get_c_offset_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc);
size_t tqt_get_bias_offset_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd);
size_t tqt_get_d_size_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(size_t m, size_t n);

/// Runs the implicit GEMM convolution micro-kernel (channel-last, transposed weight).
///
/// @param[in]  m     Number of spatial positions to process.
/// @param[in]  n     Number of output channels (OC) to process.
/// @param[in]  k     K chunk length (KD*KH*KW*IC for full K).
/// @param[in]  input Original input tensor [ID, IH, IW, IC].
/// @param[in]  group_idx Group index for grouped convolution.
/// @param[in]  m_idx Starting spatial position (linear index into OD*OH*OW).
/// @param[in]  k_idx_a  K offset for input addressing (must be aligned to IC).
/// @param[in]  B     Weight matrix [OC, K] row-major (transposed B).
/// @param[in]  ldb   Leading dimension of B (= K).
/// @param[in]  k_idx_b  K offset in weight (must be aligned to IC).
/// @param[in]  C     Accumulation buffer or NULL.
/// @param[in]  ldc   Leading dimension of C (= OC).
/// @param[out] D     Output buffer [OD*OH*OW, OC].
/// @param[in]  ldd   Leading dimension of D (= OC).
/// @param[in]  bias  Bias vector [OC] or NULL.
/// @param[in]  clamp_min Minimum clamp value.
/// @param[in]  clamp_max Maximum clamp value.
/// @param[in]  params Convolution parameters.
void tqt_run_igemm_igemmcl_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t m_idx, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, float16_t clamp_min, float16_t clamp_max, const tqt_conv_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif
