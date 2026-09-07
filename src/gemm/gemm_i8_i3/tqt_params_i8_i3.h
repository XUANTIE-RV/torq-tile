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

/// Parameters for qsi8d256 A x qsi3c16 B -> f32.
///
/// scale_a is [M, K/256], qscale_b is [N, K/16], and super_scale_b is
/// [N, K/256].
struct tqt_qsi8d256_qsi3c16_params
{
    const float *scale_a;
    size_t ldsa;
    const int8_t *qscale_b;
    size_t ldqb;
    const float *super_scale_b;
    size_t ldssb;
};

#ifdef __cplusplus
}  // extern "C"
#endif
