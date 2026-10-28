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

/// FP32 GeLU using the standard tanh approximation on an RVV 1x4vl tile.
///
/// The cubic argument and the shared rational tanh approximation are evaluated
/// in FP32. The tanh input is symmetrically saturated at +/-9. Fully packed
/// matrices are collapsed into one contiguous run; otherwise lda and ldc are
/// honoured row by row. Finite inputs in the normal activation range are
/// supported, and c may alias a when ldc == lda.
///
/// Accuracy: a single-thread sweep of 200,001 uniformly spaced FP32 inputs over
/// [-10, 10] under the xt-c9501fdvkmt QEMU model measures a maximum absolute
/// error of 9.2e-7 against a double-precision tanh-approximate GeLU reference.
/// Non-finite inputs and values outside that interval are not characterized.

size_t tqt_get_m_step_gelu_f32_f32_1x4vl_rvv(void);
size_t tqt_get_n_step_gelu_f32_f32_1x4vl_rvv(void);
void tqt_run_gelu_f32_f32_1x4vl_rvv(size_t m, size_t n, const void *a, size_t lda, void *c,
                                    size_t ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
