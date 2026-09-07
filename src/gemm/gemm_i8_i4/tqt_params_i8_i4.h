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

/// Parameters for qsi8d256 A x Q3_K-derived qsi4c16 B -> f32.
///
/// scale_a is [M, K/256], qscale_b is [N, K/16], and super_scale_b is
/// [N, K/256].
struct tqt_qsi8d256_qsi4c16_params
{
    const float *scale_a;
    size_t ldsa;
    const int8_t *qscale_b;
    size_t ldqb;
    const float *super_scale_b;
    size_t ldssb;
};

/// Parameters for qsi8d32 A x normalized qsi4c32 B -> f32.
///
/// scale_a is [M, K/32], and scale_b is [N, K/32]. The qsi4 values use
/// signed two's-complement nibbles in [-8, 7].
struct tqt_qsi8d32_qsi4c32_params
{
    const float *scale_a;
    size_t ldsa;
    const float *scale_b;
    size_t ldsb;
};

/// Quantization parameters for qai8dx (A) x qsi4cx (B) -> f16/f32/bf16 (D)
///
/// A is per-dimension (per-row) asymmetric quantized int8:
///   - scale_a:       [m] per-row scales (m taken from the run_* m argument)
///   - zero_point_a:  [m] per-row zero points
/// B is per-channel (per-column) symmetric quantized int4:
///   - scale_b:       [n] per-channel scales (n taken from the run_* n argument)
/// D is float (f16, f32, or bf16), no quantization.
struct tqt_qai8dx_qsi4cx_params
{
    const float *scale_a;
    const int32_t *zero_point_a;
    const float *scale_b;
};

/// Quantization parameters for qai8dx (A) x qsi4cx (B) -> qai8 (D, requantized int8)
///
/// A is per-row (per-dimension) asymmetric quantized int8:
///   - scale_a:       [m] per-row scales
///   - zero_point_a:  [m] per-row zero points
/// B is per-channel (per-column) symmetric quantized int4:
///   - scale_b:       [n] per-channel scales
/// D is per-tensor asymmetric quantized int8 (requantize path):
///   - scale_d:       scalar output scale
///   - zero_point_d:  scalar output zero point
/// The i32 accumulator is dequantized to fp32 (scale_a[m] * scale_b[n]), the
/// optional fp32 C / 1xN bias are added in real domain, then the result is
/// requantized to int8 via (round(D_real / scale_d) + zero_point_d) and clamped.
struct tqt_qai8_qai8dx_qsi4cx_params
{
    const float *scale_a;
    const int32_t *zero_point_a;
    const float *scale_b;
    float scale_d;
    int32_t zero_point_d;
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
