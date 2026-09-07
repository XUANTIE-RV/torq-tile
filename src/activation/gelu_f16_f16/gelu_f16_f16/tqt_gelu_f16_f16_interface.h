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

// Element-wise FP16 GeLU using the standard tanh approximation:
//   u = sqrt(2 / pi) * (x + 0.044715 * x^3)
//   c = 0.5 * x * (1 + tanh(u))
// A and C are row-major [M, N] matrices with independent leading dimensions.
// C may alias A when ldc == lda.

typedef size_t (*tqt_gelu_f16_f16_get_m_step_func_t)(void);
typedef size_t (*tqt_gelu_f16_f16_get_n_step_func_t)(void);
typedef void (*tqt_gelu_f16_f16_run_gelu_func_t)(size_t m, size_t n, const void *a, size_t lda,
                                                 void *c, size_t ldc);

struct tqt_gelu_f16_f16_ukernel
{
    tqt_gelu_f16_f16_get_m_step_func_t get_m_step;
    tqt_gelu_f16_f16_get_n_step_func_t get_n_step;
    tqt_gelu_f16_f16_run_gelu_func_t run_gelu;
};

#ifdef __cplusplus
}  // extern "C"
#endif
