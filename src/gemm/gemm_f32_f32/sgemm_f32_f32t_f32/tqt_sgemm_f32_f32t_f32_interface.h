//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// All micro-kernels variants of the same type share the same interfaces
// In this case, the micro-kernel type is: sgemm_f32_f32t_f32

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*tqt_sgemm_f32_f32t_f32_get_m_step_func_t)(void);
typedef size_t (*tqt_sgemm_f32_f32t_f32_get_n_step_func_t)(void);
typedef size_t (*tqt_sgemm_f32_f32t_f32_get_a_offset_func_t)(size_t m_idx, size_t k_idx,
                                                             size_t lda);
typedef size_t (*tqt_sgemm_f32_f32t_f32_get_b_offset_func_t)(size_t n_idx, size_t k_idx,
                                                             size_t ldb);
typedef size_t (*tqt_sgemm_f32_f32t_f32_get_c_offset_func_t)(size_t m_idx, size_t n_idx,
                                                             size_t ldc);
typedef size_t (*tqt_sgemm_f32_f32t_f32_get_d_offset_func_t)(size_t m_idx, size_t n_idx,
                                                             size_t ldd);
typedef size_t (*tqt_sgemm_f32_f32t_f32_get_d_size_func_t)(size_t m, size_t n);

/// Micro-kernel core function ("run" method)
typedef void (*tqt_sgemm_f32_f32t_f32_run_sgemm_func_t)(size_t m, size_t n, size_t k, const void *A,
                                                        size_t lda, size_t k_idx_a, const void *B,
                                                        size_t ldb, size_t k_idx_b, const void *C,
                                                        size_t ldc, void *D, size_t ldd,
                                                        float alpha, float beta);

/// Micro-kernel interface
struct tqt_sgemm_f32_f32t_f32_ukernel
{
    tqt_sgemm_f32_f32t_f32_get_m_step_func_t get_m_step;
    tqt_sgemm_f32_f32t_f32_get_n_step_func_t get_n_step;
    tqt_sgemm_f32_f32t_f32_get_a_offset_func_t get_a_offset;
    tqt_sgemm_f32_f32t_f32_get_b_offset_func_t get_b_offset;
    tqt_sgemm_f32_f32t_f32_get_c_offset_func_t get_c_offset;
    tqt_sgemm_f32_f32t_f32_get_d_offset_func_t get_d_offset;
    tqt_sgemm_f32_f32t_f32_get_d_size_func_t get_d_size;
    tqt_sgemm_f32_f32t_f32_run_sgemm_func_t run_sgemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
