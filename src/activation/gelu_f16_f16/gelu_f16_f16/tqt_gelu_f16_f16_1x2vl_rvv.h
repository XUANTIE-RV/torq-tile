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

/// FP16 GeLU using the standard tanh approximation on an RVV 1x2vl tile.
///
/// Each f16m2 input group is widened to f32m4. The cubic argument, rational tanh
/// approximation and output expression are evaluated in FP32, followed by one
/// narrowing store. Fully packed matrices are collapsed into one contiguous run;
/// otherwise lda and ldc are honoured row by row. C may alias A when ldc == lda.
///
/// Accuracy: an exhaustive single-thread sweep of every finite FP16 input in
/// [-10, 10] under the xt-c9501fdvkmt QEMU model measures a maximum absolute
/// error of 9.8e-4 against a double-precision tanh-approximate GeLU reference. Of
/// the 37,378 inputs, 35,955 produce the correctly rounded FP16 result.

size_t tqt_get_m_step_gelu_f16_f16_1x2vl_rvv(void);
size_t tqt_get_n_step_gelu_f16_f16_1x2vl_rvv(void);
void tqt_run_gelu_f16_f16_1x2vl_rvv(size_t m, size_t n, const void *a, size_t lda, void *c,
                                    size_t ldc);

#ifdef __cplusplus
}  // extern "C"
#endif
