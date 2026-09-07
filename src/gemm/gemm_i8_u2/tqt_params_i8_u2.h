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

/// Parameters for qsi8d256 A x qau2c16 B -> f32.
///
/// scale_a is [M, K/256], and bsums_a is [M, K/16]. qscale_b and qmin_b are
/// [N, K/16]. super_scale_b and super_min_b are [N, K/256].
struct tqt_qsi8d256_qau2c16_params
{
    const float *scale_a;
    size_t ldsa;
    const int16_t *bsums_a;
    size_t ldbsuma;
    const uint8_t *qscale_b;
    const uint8_t *qmin_b;
    size_t ldqb;
    const float *super_scale_b;
    const float *super_min_b;
    size_t ldssb;
};

/// Parameters for qsi8d32 A x normalized qau2z1c64 B -> f32.
///
/// scale_a is [M, K/32], and scale_b is [N, K/64]. The unsigned 2-bit code
/// maps to the signed logical value code - 1.
struct tqt_qsi8d32_qau2z1c64_params
{
    const float *scale_a;
    size_t ldsa;
    const float *scale_b;
    size_t ldsb;
};

#ifdef __cplusplus
}  // extern "C"
#endif
