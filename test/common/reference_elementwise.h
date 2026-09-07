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

// Element-wise reference kernels. All arithmetic is performed in the float
// domain and cast back to T, so the same definition serves both float and
// float16_t. Explicit instantiations for {float, float16_t} live in the .cpp.

/// Reference element-wise vector-vector add: c[i] = a[i] + b[i], i in [0, n).
template <typename T>
void reference_addvv(size_t n, const T *a, const T *b, T *c);

/// Reference element-wise vector-vector subtract: c[i] = a[i] - b[i], i in [0, n).
template <typename T>
void reference_subvv(size_t n, const T *a, const T *b, T *c);

/// Reference element-wise vector-vector multiply: c[i] = a[i] * b[i], i in [0, n).
template <typename T>
void reference_mulvv(size_t n, const T *a, const T *b, T *c);

/// Reference element-wise vector-vector divide: c[i] = a[i] / b[i], i in [0, n).
template <typename T>
void reference_divvv(size_t n, const T *a, const T *b, T *c);

/// Reference element-wise vector-scalar add: c[i] = a[i] + b, i in [0, n).
template <typename T>
void reference_addvs(size_t n, const T *a, T b, T *c);

/// Reference element-wise vector-scalar subtract: c[i] = a[i] - b, i in [0, n).
template <typename T>
void reference_subvs(size_t n, const T *a, T b, T *c);

/// Reference element-wise vector-scalar multiply: c[i] = a[i] * b, i in [0, n).
template <typename T>
void reference_mulvs(size_t n, const T *a, T b, T *c);

/// Reference element-wise vector-scalar divide: c[i] = a[i] / b, i in [0, n).
template <typename T>
void reference_divvs(size_t n, const T *a, T b, T *c);

}  // namespace test
}  // namespace torq_tile
