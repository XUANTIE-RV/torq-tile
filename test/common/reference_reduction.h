//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// Sum reductions
// ============================================================================

/// Reference row-wise sum reduction (FP32).
///
/// For each row m, computes: c[m] = (init ? init[m] : 0) + sum(a[m * lda + 0 .. n-1]).
///
/// @param m     Number of rows
/// @param n     Number of columns
/// @param a     Input matrix A [m, n], row-major
/// @param lda   Leading dimension of A (>= n)
/// @param c     Output vector C [m]
/// @param init  Optional initial values [m] (can be nullptr)
void reference_sumrow_f32(size_t m, size_t n, const float *a, size_t lda, float *c,
                          const float *init);

/// Reference column-wise sum reduction (FP32).
///
/// For each column n, computes: c[n] = (init ? init[n] : 0) + sum(a[row * lda + n]) over all rows.
///
/// @param m     Number of rows
/// @param n     Number of columns
/// @param a     Input matrix A [m, n], row-major
/// @param lda   Leading dimension of A (>= n)
/// @param c     Output vector C [n]
/// @param init  Optional initial values [n] (can be nullptr)
void reference_sumcol_f32(size_t m, size_t n, const float *a, size_t lda, float *c,
                          const float *init);

/// Reference row-wise sum reduction (FP16 in / FP16 out, FP32 accumulation).
///
/// For each row m, computes: c[m] = (init ? init[m] : 0) + sum(a[m * lda + 0 .. n-1]).
/// Accumulation is performed in FP32, then narrowed back to FP16 on store.
void reference_sumrow_f16(size_t m, size_t n, const float16_t *a, size_t lda, float16_t *c,
                          const float16_t *init);

/// Reference column-wise sum reduction (FP16 in / FP16 out, FP32 accumulation).
///
/// For each column n, computes: c[n] = (init ? init[n] : 0) + sum(a[row * lda + n]) over all rows.
/// Accumulation is performed in FP32, then narrowed back to FP16 on store.
void reference_sumcol_f16(size_t m, size_t n, const float16_t *a, size_t lda, float16_t *c,
                          const float16_t *init);

// ============================================================================
// Mean reductions
//   Mean applies the 1/N scaling to the reduced sum only; init is added as an
//   output bias after the averaging (mirrors the sum init semantics):
//     meanrow: c[m] = (init ? init[m] : 0) + (1/n) * sum_n a[m, n]
//     meancol: c[n] = (init ? init[n] : 0) + (1/m) * sum_m a[m, n]
// ============================================================================

/// Reference row-wise mean reduction (FP32).
void reference_meanrow_f32(size_t m, size_t n, const float *a, size_t lda, float *c,
                           const float *init);

/// Reference column-wise mean reduction (FP32).
void reference_meancol_f32(size_t m, size_t n, const float *a, size_t lda, float *c,
                           const float *init);

/// Reference row-wise mean reduction (FP16 in / FP16 out, FP32 accumulation).
void reference_meanrow_f16(size_t m, size_t n, const float16_t *a, size_t lda, float16_t *c,
                           const float16_t *init);

/// Reference column-wise mean reduction (FP16 in / FP16 out, FP32 accumulation).
void reference_meancol_f16(size_t m, size_t n, const float16_t *a, size_t lda, float16_t *c,
                           const float16_t *init);

}  // namespace test
}  // namespace torq_tile
