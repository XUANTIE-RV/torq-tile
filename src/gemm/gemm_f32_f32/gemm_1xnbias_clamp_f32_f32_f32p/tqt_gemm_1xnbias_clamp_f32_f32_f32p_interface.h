//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/common/tqt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// All micro-kernels variants of the same type share the same interfaces
// In this case, the micro-kernel type is: gemm_1xnbias_clamp_f32_f32_f32p
// B-only packed variant: A is in original row-major layout, B is pre-packed.

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_m_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_n_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_a_offset_func_t)(size_t m_idx,
                                                                          size_t k_idx, size_t lda);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_b_packed_offset_func_t)(size_t n_idx,
                                                                                 size_t k_idx,
                                                                                 size_t ldb,
                                                                                 size_t actual_n);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_c_offset_func_t)(size_t m_idx,
                                                                          size_t n_idx, size_t ldc);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_bias_offset_func_t)(size_t n_idx);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_d_offset_func_t)(size_t m_idx,
                                                                          size_t n_idx, size_t ldd);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_d_size_func_t)(size_t m, size_t n);
typedef size_t (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_b_packed_size_func_t)(size_t n, size_t k);

/// Micro-kernel pack functions
typedef void (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_run_b_pack_func_t)(size_t n, size_t k,
                                                                      size_t ldb, size_t ldb_packed,
                                                                      size_t k_idx, const void *B,
                                                                      void *B_packed);
typedef void (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_run_bt_pack_func_t)(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);

/// Micro-kernel core function ("run" method)
typedef void (*tqt_gemm_1xnbias_clamp_f32_f32_f32p_run_gemm_func_t)(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B_packed,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float clamp_min, float clamp_max);

/// Micro-kernel interface
struct tqt_gemm_1xnbias_clamp_f32_f32_f32p_ukernel
{
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_m_step_func_t get_m_step;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_n_step_func_t get_n_step;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_a_offset_func_t get_a_offset;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_b_packed_offset_func_t get_b_packed_offset;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_c_offset_func_t get_c_offset;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_bias_offset_func_t get_bias_offset;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_d_offset_func_t get_d_offset;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_d_size_func_t get_d_size;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_get_b_packed_size_func_t get_b_packed_size;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_run_b_pack_func_t run_b_pack;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_run_bt_pack_func_t run_bt_pack;
    tqt_gemm_1xnbias_clamp_f32_f32_f32p_run_gemm_func_t run_gemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
