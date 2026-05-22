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

// All micro-kernel variants of the same type share the same interfaces
// In this case, the micro-kernel type is: gemm_1xnbias_clamp_f16_f16_qsi4c2t

typedef size_t (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_m_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_n_step_func_t)(void);
typedef size_t (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_a_offset_func_t)(size_t m_idx,
                                                                             size_t k_idx,
                                                                             size_t lda, size_t bl);
typedef size_t (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_b_offset_func_t)(size_t n_idx,
                                                                             size_t k_idx,
                                                                             size_t ldb, size_t bl);
typedef size_t (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_scale_b_offset_func_t)(size_t n_idx,
                                                                                   size_t b_idx,
                                                                                   size_t ldsb);
typedef size_t (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_c_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldc);
typedef size_t (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_bias_offset_func_t)(size_t n_idx);
typedef size_t (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_d_offset_func_t)(size_t m_idx,
                                                                             size_t n_idx,
                                                                             size_t ldd);
typedef size_t (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_d_size_func_t)(size_t m, size_t n);

typedef void (*tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_run_gemm_func_t)(
    size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b, const void *scale_b, size_t ldsb, const void *C,
    size_t ldc, void *D, size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max);

struct tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_ukernel
{
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_m_step_func_t get_m_step;
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_n_step_func_t get_n_step;
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_a_offset_func_t get_a_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_b_offset_func_t get_b_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_scale_b_offset_func_t get_scale_b_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_c_offset_func_t get_c_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_bias_offset_func_t get_bias_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_d_offset_func_t get_d_offset;
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_get_d_size_func_t get_d_size;
    tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_run_gemm_func_t run_gemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
