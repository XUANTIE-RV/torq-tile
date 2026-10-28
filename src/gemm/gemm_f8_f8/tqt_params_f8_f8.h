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

/// Quantization parameters for qsf8e4 (A) x qsf8e4 (B) -> bf16 (D), per-tensor.
///
/// A and B are FP8 E4M3 quantized symmetrically with a single fp32 scalar each:
///     a_{m,k} ~= q_a_{m,k} * scale_a
///     b_{k,n} ~= q_b_{k,n} * scale_b
/// D is bf16 output (no quantization), only clamped by [clamp_min, clamp_max].
struct tqt_qsf8e4_qsf8e4_params
{
    float scale_a;
    float scale_b;
};

/// Quantization parameters for qsf8e4dx (A) x qsf8e4cx (B) -> bf16 (D),
/// per-token (A) + per-channel (B).
///
/// A is per-row (per-dimension) symmetric FP8 (one fp32 scale per row of M):
///     a_{m,k} ~= q_a_{m,k} * scale_a[m]
/// B is per-channel symmetric FP8 (one fp32 scale per output channel of N):
///     b_{k,n} ~= q_b_{k,n} * scale_b[n]
/// D is bf16 output (no quantization), only clamped by [clamp_min, clamp_max].
struct tqt_qsf8e4dx_qsf8e4cx_params
{
    const float *scale_a;
    const float *scale_b;
};

#ifdef __cplusplus
}  // extern "C"
#endif
