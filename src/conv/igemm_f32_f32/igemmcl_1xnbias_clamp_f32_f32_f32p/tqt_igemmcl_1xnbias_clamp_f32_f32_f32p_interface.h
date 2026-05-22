//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/common/tqt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

// Implicit GEMM micro-kernel type: igemmcl_1xnbias_clamp_f32_f32_f32p
// Channel-last implicit GEMM convolution (NDHWC), packed weight, 1xN bias.
//
// Weight packed from [OC, K] to [OC/(3vl), K, 3vl] for contiguous vector loads.
//
// K-dimension alignment requirement:
//   k_idx_a and k_idx_b MUST be multiples of IC.

/// Helper functions
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_m_step_func_t)(void);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_n_step_func_t)(void);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_b_packed_offset_func_t)(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_c_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldc);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_bias_offset_func_t)(size_t n_idx);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_d_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldd);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_d_size_func_t)(size_t m, size_t n);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_b_packed_size_func_t)(size_t n,
                                                                                  size_t k);

/// Weight pack function: [OC, K] -> [OC/(3vl), K, 3vl]
typedef void (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_run_bt_pack_func_t)(
    size_t n, size_t k, size_t ldw, size_t ldw_packed, size_t k_idx, const void *weight,
    void *weight_packed);

/// Core function
typedef void (*tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_run_igemm_func_t)(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t m_idx, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float clamp_min, float clamp_max, const tqt_conv_params *params);

/// Micro-kernel interface
struct tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_ukernel
{
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_m_step_func_t get_m_step;
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_n_step_func_t get_n_step;
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_b_packed_offset_func_t get_b_packed_offset;
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_c_offset_func_t get_c_offset;
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_bias_offset_func_t get_bias_offset;
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_d_offset_func_t get_d_offset;
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_d_size_func_t get_d_size;
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_get_b_packed_size_func_t get_b_packed_size;
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_run_bt_pack_func_t run_bt_pack;
    tqt_igemmcl_1xnbias_clamp_f32_f32_f32p_run_igemm_func_t run_igemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
