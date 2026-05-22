//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// All micro-kernels variants of the same type share the same interfaces
// In this case, the micro-kernel type is: gemm_clamp_i32_i8_i8

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*tqt_gemm_clamp_i32_i8_i8_get_m_step_func)(void);
typedef size_t (*tqt_gemm_clamp_i32_i8_i8_get_n_step_func)(void);
typedef size_t (*tqt_gemm_clamp_i32_i8_i8_get_a_offset_func)(size_t m_idx, size_t k_idx,
                                                             size_t lda);
typedef size_t (*tqt_gemm_clamp_i32_i8_i8_get_b_offset_func)(size_t n_idx, size_t k_idx,
                                                             size_t ldb);
typedef size_t (*tqt_gemm_clamp_i32_i8_i8_get_c_offset_func)(size_t m_idx, size_t n_idx,
                                                             size_t ldc);
typedef size_t (*tqt_gemm_clamp_i32_i8_i8_get_d_offset_func)(size_t m_idx, size_t n_idx,
                                                             size_t ldd);
typedef size_t (*tqt_gemm_clamp_i32_i8_i8_get_d_size_func)(size_t m, size_t n);

/// Micro-kernel core function ("run" method)
typedef void (*tqt_gemm_clamp_i32_i8_i8_run_gemm_func)(size_t m, size_t n, size_t k, const void *A,
                                                       size_t lda, size_t k_idx_a, const void *B,
                                                       size_t ldb, size_t k_idx_b, const void *C,
                                                       size_t ldc, void *D, size_t ldd,
                                                       int32_t clamp_min, int32_t clamp_max);

/// Micro-kernel interface
struct tqt_gemm_clamp_i32_i8_i8_ukernel
{
    tqt_gemm_clamp_i32_i8_i8_get_m_step_func get_m_step;
    tqt_gemm_clamp_i32_i8_i8_get_n_step_func get_n_step;
    tqt_gemm_clamp_i32_i8_i8_get_a_offset_func get_a_offset;
    tqt_gemm_clamp_i32_i8_i8_get_b_offset_func get_b_offset;
    tqt_gemm_clamp_i32_i8_i8_get_c_offset_func get_c_offset;
    tqt_gemm_clamp_i32_i8_i8_get_d_offset_func get_d_offset;
    tqt_gemm_clamp_i32_i8_i8_get_d_size_func get_d_size;
    tqt_gemm_clamp_i32_i8_i8_run_gemm_func run_gemm;
};

#ifdef __cplusplus
}  // extern "C"
#endif
