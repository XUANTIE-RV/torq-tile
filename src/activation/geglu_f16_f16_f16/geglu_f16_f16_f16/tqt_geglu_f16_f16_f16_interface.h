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

// Element-wise GeGLU gated activation, FP16.
//   a layout: [M, N] row-major, stride lda (>= n) - the gate operand
//   b layout: [M, N] row-major, stride ldb (>= n) - the linear operand
//   c layout: [M, N] row-major, stride ldc (>= n)
//   c[i, j] = gelu(a[i, j]) * b[i, j]
//
// The operands carry independent leading dimensions. A fused [M, 2N]
// gate-and-up buffer is addressed with a = base, lda = 2 * n and
// b = base + n, ldb = 2 * n. The output may alias either input when its
// leading dimension matches the aliased operand.

typedef size_t (*tqt_geglu_f16_f16_f16_get_m_step_func_t)(void);
typedef size_t (*tqt_geglu_f16_f16_f16_get_n_step_func_t)(void);
typedef void (*tqt_geglu_f16_f16_f16_run_geglu_func_t)(size_t m, size_t n, const void *a,
                                                       size_t lda, const void *b, size_t ldb,
                                                       void *c, size_t ldc);

struct tqt_geglu_f16_f16_f16_ukernel
{
    tqt_geglu_f16_f16_f16_get_m_step_func_t get_m_step;
    tqt_geglu_f16_f16_f16_get_n_step_func_t get_n_step;
    tqt_geglu_f16_f16_f16_run_geglu_func_t run_geglu;
};

#ifdef __cplusplus
}  // extern "C"
#endif
