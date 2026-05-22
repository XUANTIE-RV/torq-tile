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

/// Quantization parameters for qai8dx (A) x qsi4cx (B) -> f16/f32 (D)
///
/// A is per-dimension (per-row) asymmetric quantized int8:
///   - scale_a:       [m] per-row scales (m taken from the run_* m argument)
///   - zero_point_a:  [m] per-row zero points
/// B is per-channel (per-column) symmetric quantized int4:
///   - scale_b:       [n] per-channel scales (n taken from the run_* n argument)
/// D is float (f16 or f32), no quantization.
struct tqt_qai8dx_qsi4cx_params
{
    const float *scale_a;
    const int32_t *zero_point_a;
    const float *scale_b;
};

#ifdef __cplusplus
}  // extern "C"
#endif
