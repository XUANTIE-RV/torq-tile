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

/// Micro-kernel: elementwise vector-scalar add for FP32
///
/// This is an RVV (8vl tile) implementation using LMUL=8 configuration.
/// n_step = 8 * csrr_vlenb() / sizeof(float).
/// Computes: c[i] = a[i] + b, i in [0, n)

/// Gets n step value (processing width per inner step).
///
/// @return The n step value (8 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_addvs_f32_f32_f32_8vl_rvv(void);

/// Runs the vector-scalar add micro-kernel.
///
/// @param[in]  n Number of elements.
/// @param[in]  a Input vector A [n], contiguous float array.
/// @param[in]  b Scalar value to add.
/// @param[out] c Output vector C [n], contiguous float array.
void tqt_run_addvs_f32_f32_f32_8vl_rvv(size_t n, const void *a, float b, void *c);

#ifdef __cplusplus
}  // extern "C"
#endif
