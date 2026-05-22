//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

// Implicit GEMM micro-kernel type: igemmcl_1xnbias_clamp_f16_f16_f16t
// Channel-last implicit GEMM convolution (NDHWC), non-packed weight (transposed B), 1xN bias.
//
// Fuses im2col + GEMM: input elements fetched on-the-fly from the original input tensor.
//
// Vectorization direction: N (OC) is vectorized, M (spatial) is scalar.
//   A = implicit im2col of input [M, K] (M = OD*OH*OW, K = KD*KH*KW*IC, NOT materialized)
//   B = weight [OC, K] row-major (transposed B in GEMM terms, K = KD*KH*KW*IC)
//
// K-dimension alignment requirement:
//   k_idx_a and k_idx_b MUST be multiples of IC.

/// Helper functions
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_m_step_func_t)(void);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_n_step_func_t)(void);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_b_offset_func_t)(size_t n_idx,
                                                                             size_t k_idx,
                                                                             size_t ldb);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_c_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldc);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_bias_offset_func_t)(size_t n_idx);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_d_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldd);
typedef size_t (*tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_d_size_func_t)(size_t m, size_t n);

/// Core function: fused implicit-im2col + GEMM convolution micro-kernel.
///
/// M and N tiling is handled internally.
/// K-tiling behavior same as igemmcf (first/last chunk bias/clamp logic).
typedef void (*tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_run_igemm_func_t)(
    size_t m, size_t n, size_t k, const void *input, size_t group_idx, size_t m_idx, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
    const void *bias, float16_t clamp_min, float16_t clamp_max, const tqt_conv_params *params);

/// Micro-kernel interface
struct tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_ukernel
{
    tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_m_step_func_t get_m_step;
    tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_n_step_func_t get_n_step;
    tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_b_offset_func_t get_b_offset;
    tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_c_offset_func_t get_c_offset;
    tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_bias_offset_func_t get_bias_offset;
    tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_d_offset_func_t get_d_offset;
    tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_get_d_size_func_t get_d_size;
    tqt_igemmcl_1xnbias_clamp_f16_f16_f16t_run_igemm_func_t run_igemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
