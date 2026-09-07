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

/// Quantization parameters for qai8dx (A) x qsi4c2 (B, block-quantized) -> f16/f32/bf16 (D)
///
/// A is per-row asymmetric int8 (no block dimension):
///   - scale_a       [m]            fp32 per-row scale
///   - zero_point_a  [m]            int32 per-row zero point
/// B is per-block symmetric int4 along K (block_size = bl, multiple of 2):
///   - scale_b       [n, k/bl]      fp32 row-major per-block scale
///   - ldsb          leading dim of scale_b in fp32 elements (typically k/bl)
/// Nibbles in B are stored unsigned in [0, 15] with ZP=8 offset; dequant
/// recovers signed value via (nibble - 8) * scale_b[n, blk].
struct tqt_qai8dx_qsi4c2_params
{
    const float *scale_a;
    const int32_t *zero_point_a;
    const float *scale_b;
    size_t ldsb;
};

/// Quantization parameters for qsi8d2 (A, block sym) x qsi4c2 (B, block sym) -> f16/f32/bf16 (D)
///
/// A is per-row block-quantized symmetric int8 along K (no zero point):
///   - scale_a       [m, k/bl]      fp32 row-major per-block scale
///   - ldsa          leading dim of scale_a in fp32 elements (typically k/bl)
/// B is per-block symmetric int4 along K (same encoding as in
/// tqt_qai8dx_qsi4c2_params):
///   - scale_b       [n, k/bl]      fp32 row-major
///   - ldsb          leading dim of scale_b in fp32 elements
struct tqt_qsi8d2_qsi4c2_params
{
    const float *scale_a;
    size_t ldsa;
    const float *scale_b;
    size_t ldsb;
};

/// Quantization parameters for qsi8d2 (A, block sym) x qai4c2 (B, block ASYM)
///                                                                 -> f16/f32/bf16 (D)
///
/// A is per-row block-quantized symmetric int8 along K (no zero point):
///   - scale_a       [m, k/bl]      fp32 row-major (ldsa)
/// B is per-block ASYMMETRIC int4 along K. Nibbles are stored unsigned in
/// [0, 15] (ZP=0 encoding); zero point is carried in zp_b as fp32.
/// Dequant: (nibble - zp_b[n, blk]) * scale_b[n, blk].
///   - scale_b       [n, k/bl]      fp32 row-major (ldsb)
///   - zp_b          [n, k/bl]      fp32 row-major (ldzpb)
struct tqt_qsi8d2_qai4c2_params
{
    const float *scale_a;
    size_t ldsa;
    const float *scale_b;
    size_t ldsb;
    const float *zp_b;
    size_t ldzpb;
};

#ifdef __cplusplus
}  // extern "C"
#endif
