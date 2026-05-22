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

/// Quantization parameters for qai8 (A) x qsi8cx (B) -> qai8 (D)
struct tqt_qai8_qai8_qsi8cx_params
{
    float scale_a;
    int32_t zero_point_a;
    float *scale_b;
    int32_t quant_channel_b;
    float scale_d;
    int32_t zero_point_d;
};

/// Quantization parameters for qsi8dx (A) x qai8 (B) -> qai8 (D)
/// A is per-dimension symmetric quantized (M scales), B is per-tensor asymmetric.
struct tqt_qai8_qsi8dx_qai8_params
{
    float *scale_a;
    int32_t quant_channel_a;
    float scale_b;
    int32_t zero_point_b;
    float scale_d;
    int32_t zero_point_d;
};

/// Quantization parameters for qai8dx (A) x qsi8cx (B) -> f32 / f16 (D)
///
/// A is per-row asymmetric int8 (each of M rows has its own scale + zero_point):
///     a_{m,k} ~= (q_a_{m,k} - zero_point_a[m]) * scale_a[m]
/// B is per-channel symmetric int8 (each of N channels has its own scale, no zp):
///     b_{k,n} ~= q_b_{k,n} * scale_b[n]
/// D is fp32 / fp16 output (no quantization), only clamped by [clamp_min, clamp_max].
struct tqt_qai8dxp_qsi8cxp_params
{
    float *scale_a;
    int32_t *zero_point_a;
    float *scale_b;
};

#ifdef __cplusplus
}  // extern "C"
#endif
