//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Quantization parameters for qai8dx (A) x qsi2cx (B) -> f32 (D)
///
/// A is per-dimension asymmetric int8: M rows each carry their own scale and zero_point.
/// B is per-channel symmetric int2: each output channel (N) carries one scale.
///
/// `sum_b[n] = sum_k lut[u2_b[n][k]]` is the row-sum of the dequantized int2 weights for the
/// n-th output channel; it is used by the kernel's zero-point correction
/// `iacc -= zp_a[m] * sum_b[n]`. It must be precomputed by the caller (typically together with
/// the int2 quantization step) and passed in alongside the other quantization parameters.
struct tqt_qai8dx_qsi2cx_f32_params
{
    const float *scale_a;  // [M] reciprocal applied at output: dst = ... * scale_a[m] * scale_b[n]
    const int32_t *zero_point_a;  // [M] nudged zero points used during A quantization
    const float *scale_b;         // [N] per-channel scale of B
    const int32_t *sum_b;         // [N] precomputed row-sum of dequantized int2 weights
};

/// Parameters for per-tensor asymmetric I8 A and per-channel LUT-decoded I2 B.
/// Raw I32 bias is added to the centered dot before scale_a * scale_b / scale_d.
struct tqt_qai8_qai8_qsi2cx_params
{
    float scale_a;
    int32_t zero_point_a;
    const float *scale_b;  // [N]
    float scale_d;
    int32_t zero_point_d;
};

#ifdef __cplusplus
}  // extern "C"
#endif
