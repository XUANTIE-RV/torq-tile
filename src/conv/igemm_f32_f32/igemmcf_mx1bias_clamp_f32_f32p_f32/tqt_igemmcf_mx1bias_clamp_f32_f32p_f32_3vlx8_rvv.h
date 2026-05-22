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

/// Implicit GEMM micro-kernel: channel-first, Mx1 bias, clamp, packed weight, FP32
/// RVV 3vlx8 tile implementation.
///
/// Weight packed from [OC, K] to [OC/3vl, K, 3vl] for contiguous vector loads.
/// Input fetched on-the-fly (implicit im2col).

/// --------------------------------------------------

size_t tqt_get_m_step_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(void);
size_t tqt_get_n_step_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(void);

size_t tqt_get_a_packed_offset_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m_idx,
                                                                            size_t k_idx,
                                                                            size_t lda,
                                                                            size_t actual_m);
size_t tqt_get_c_offset_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc);
size_t tqt_get_bias_offset_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m_idx);
size_t tqt_get_d_offset_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd);
size_t tqt_get_d_size_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m, size_t n);
size_t tqt_get_a_packed_size_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m, size_t k);

/// Pack weight from [OC, K] row-major to [OC/3vl, K, 3vl] packed format.
void tqt_run_a_pack_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(size_t m, size_t k, size_t ldw,
                                                                 size_t ldw_packed, size_t k_idx,
                                                                 const void *weight,
                                                                 void *weight_packed);

/// Runs the implicit GEMM convolution micro-kernel with packed weight.
void tqt_run_igemm_igemmcf_mx1bias_clamp_f32_f32p_f32_3vlx8_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t n_idx, size_t k_idx_b,
    const void *A_packed, size_t lda, size_t k_idx_a, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max, const tqt_conv_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif
