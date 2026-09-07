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

/// Micro-kernel: element-wise GeGLU gated activation for FP32
///
/// Computes c[i, j] = gelu(a[i, j]) * b[i, j].
/// A is the gate operand and B is the linear operand. All matrices have
/// independent leading dimensions, and C may alias A or B.
/// The RVV implementation reuses the standard tanh-approximate GeLU path.
///
/// A fused [M, 2N] gate-and-up buffer requires no repacking: pass
/// a = base, lda = 2 * n and b = base + n, ldb = 2 * n.
///
/// Accuracy: a single-thread sweep of 200,001 uniformly spaced gate values over
/// [-10, 10] and integer b in [-10, 10] in the tested QEMU configuration measures
/// a maximum absolute error of 1.04e-5 against a double-precision tanh-approximate
/// GeGLU reference.
/// Non-finite inputs are not covered.

size_t tqt_get_m_step_geglu_f32_f32_f32_1x4vl_rvv(void);
size_t tqt_get_n_step_geglu_f32_f32_f32_1x4vl_rvv(void);
void tqt_run_geglu_f32_f32_f32_1x4vl_rvv(size_t m, size_t n, const void *a, size_t lda,
                                         const void *b, size_t ldb, void *c, size_t ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
