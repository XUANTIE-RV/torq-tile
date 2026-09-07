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

/// Micro-kernel: element-wise ReGLU gated activation for FP32
///
/// Computes c[i, j] = relu(a[i, j]) * b[i, j].
/// A is the gate operand and B is the linear operand. All matrices have
/// independent leading dimensions, and C may alias A or B.
/// The RVV implementation uses LMUL=8 because max and multiply need only a small number of live
/// vector groups.
///
/// A fused [M, 2N] gate-and-up buffer requires no repacking: pass
/// a = base, lda = 2 * n and b = base + n, ldb = 2 * n.
///
/// Accuracy: all 4,200,021 pairs from 200,001 uniformly spaced finite gate values
/// over [-10, 10] and integer b in [-10, 10] match the correctly rounded FP32
/// reference bit for bit in the tested QEMU configuration. The maximum error
/// against the unrounded double-precision value is 3.82e-6. This is a bounded
/// measurement, not a guarantee for non-finite or untested signed-zero inputs.

size_t tqt_get_m_step_reglu_f32_f32_f32_1x8vl_rvv(void);
size_t tqt_get_n_step_reglu_f32_f32_f32_1x8vl_rvv(void);
void tqt_run_reglu_f32_f32_f32_1x8vl_rvv(size_t m, size_t n, const void *a, size_t lda,
                                         const void *b, size_t ldb, void *c, size_t ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
