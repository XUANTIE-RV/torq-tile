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

/// Parameters for qsi8d256 A x qau4c32 B -> f32.
///
/// scale_a is [M, K/256], and bsums_a is [M, K/16]. Each bsums_a entry is
/// the exact sum of the corresponding 16 quantized activation values.
/// scale_b and real_zero_b are [N, K/32].
struct tqt_qsi8d256_qau4c32_params
{
    const float *scale_a;
    size_t ldsa;
    const int16_t *bsums_a;
    size_t ldbsuma;
    const float *scale_b;
    const float *real_zero_b;
    size_t ldsb;
};

/// Parameters for qsi8d256 A x Q2_K-derived qau4c16 B -> f32.
///
/// scale_a is [M, K/256], and bsums_a is [M, K/16]. qscale_b and qmin_b are
/// [N, K/16]. super_scale_b and super_min_b are [N, K/256].
struct tqt_qsi8d256_qau4c16_params
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

#ifdef __cplusplus
}  // extern "C"
#endif
