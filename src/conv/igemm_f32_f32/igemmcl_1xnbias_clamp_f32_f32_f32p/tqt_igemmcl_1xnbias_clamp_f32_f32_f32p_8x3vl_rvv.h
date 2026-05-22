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

/// Implicit GEMM micro-kernel: channel-last, 1xN bias, clamp, packed weight, FP32
/// RVV 8x3vl tile implementation.
///
/// Weight packed from [OC, K] to [OC/(3vl), K, 3vl] for contiguous vector loads.

size_t tqt_get_m_step_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(void);
size_t tqt_get_n_step_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(void);

size_t tqt_get_b_packed_offset_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t n_idx,
                                                                            size_t k_idx,
                                                                            size_t ldb,
                                                                            size_t actual_n);
size_t tqt_get_c_offset_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc);
size_t tqt_get_bias_offset_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t n_idx);
size_t tqt_get_d_offset_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd);
size_t tqt_get_d_size_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t m, size_t n);
size_t tqt_get_b_packed_size_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t n, size_t k);

void tqt_run_bt_pack_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(size_t n, size_t k, size_t ldw,
                                                                  size_t ldw_packed, size_t k_idx,
                                                                  const void *weight,
                                                                  void *weight_packed);

void tqt_run_igemm_igemmcl_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t m_idx, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max, const tqt_conv_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif
