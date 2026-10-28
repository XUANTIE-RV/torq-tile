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

/// Default LUT for qsi2 -> int8 dequantization.
/// u2 ∈ {0, 1, 2, 3}  ->  i8 ∈ {-2, -1, 0, 1}.  Equivalent to `u2 - 2`.
static const int8_t tqt_qsi2_default_lut[4] = {-2, -1, 0, 1};

/// Decode a single 2-bit slot from a packed byte using the default LUT.
///
/// The byte layout is `s3 s2 s1 s0` with s0 in the lowest 2 bits.
static inline int8_t tqt_qsi2_decode_default(uint8_t byte, size_t slot)
{
    return (int8_t)(((byte >> (slot * 2)) & 0x3) - 2);
}

/// Compute per-channel sum_b[n] for the n-th output channel of B.
///
/// B is laid out as [N, K/4] uint8 row-major (transposed view), each byte containing four
/// 2-bit weights packed `s3 s2 s1 s0`. The returned value is `Σ_k lut[u2_b[n][k]]`.
/// This must be called by the caller before invoking the matmul kernel and the result stored
/// in `tqt_qai8dx_qsi2cx_f32_params::sum_b[n]`.
static inline int32_t tqt_compute_sum_b_row(const uint8_t *b_row, size_t k)
{
    int32_t s = 0;
    size_t k_bytes = k / 4;
    for (size_t kb = 0; kb < k_bytes; ++kb) {
        uint8_t byte = b_row[kb];
        s += tqt_qsi2_decode_default(byte, 0);
        s += tqt_qsi2_decode_default(byte, 1);
        s += tqt_qsi2_decode_default(byte, 2);
        s += tqt_qsi2_decode_default(byte, 3);
    }
    return s;
}

/// Compute sum_b[0..n) for an N×K int2 weight matrix laid out as [N, K/4].
static inline void tqt_compute_sum_b(const uint8_t *B, size_t n, size_t k, size_t ldb_bytes,
                                     int32_t *sum_b)
{
    for (size_t i = 0; i < n; ++i) {
        sum_b[i] = tqt_compute_sum_b_row(B + i * ldb_bytes, k);
    }
}

/// Sum the decoded values of a K window using the same LUT as the dot product.
/// byte_stride is one for a transposed row, or the packed panel width for Bp.
static inline int32_t tqt_compute_sum_b_row_lut(const uint8_t *row, size_t k, size_t byte_stride,
                                                const int8_t *lut)
{
    const int8_t *values = lut ? lut : tqt_qsi2_default_lut;
    int32_t sum = 0;
    for (size_t i = 0; i < k; ++i)
        sum += values[(row[(i / 4) * byte_stride] >> (2 * (i % 4))) & 3];
    return sum;
}

#ifdef __cplusplus
}  // extern "C"
#endif
