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

/// Micro-kernel: elementwise vector-scalar divide for FP16
///
/// This is an RVV (8vl tile) implementation using LMUL=8 configuration.
/// n_step = 8 * csrr_vlenb() / sizeof(float16_t).
/// Computes: c[i] = a[i] / b, i in [0, n)
///
/// Implemented as a reciprocal multiply — the scalar reciprocal 1/b is computed
/// once (in fp32, rounded to fp16) and each element is multiplied by it
/// (c[i] = a[i] * (1/b)) since a multiply is much cheaper than a divide. Results
/// may differ from a true per-element division by a small number of ULPs.

/// Gets n step value (processing width per inner step).
///
/// @return The n step value (8 * vl, where vl = csrr_vlenb() / sizeof(float16_t)).
size_t tqt_get_n_step_divvs_f16_f16_f16_8vl_rvv(void);

/// Runs the vector-scalar divide micro-kernel.
///
/// @param[in]  n Number of elements.
/// @param[in]  a Input vector A [n], contiguous float16_t array.
/// @param[in]  b Scalar value to divide by.
/// @param[out] c Output vector C [n], contiguous float16_t array.
void tqt_run_divvs_f16_f16_f16_8vl_rvv(size_t n, const void *a, float16_t b, void *c);

#ifdef __cplusplus
}  // extern "C"
#endif
