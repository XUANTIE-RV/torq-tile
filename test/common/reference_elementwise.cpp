//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/reference_elementwise.h"

namespace torq_tile
{
namespace test
{

template <typename T>
void reference_addvv(size_t n, const T *a, const T *b, T *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
    }
}

template <typename T>
void reference_subvv(size_t n, const T *a, const T *b, T *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T>(static_cast<float>(a[i]) - static_cast<float>(b[i]));
    }
}

template <typename T>
void reference_mulvv(size_t n, const T *a, const T *b, T *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T>(static_cast<float>(a[i]) * static_cast<float>(b[i]));
    }
}

template <typename T>
void reference_divvv(size_t n, const T *a, const T *b, T *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T>(static_cast<float>(a[i]) / static_cast<float>(b[i]));
    }
}

template <typename T>
void reference_addvs(size_t n, const T *a, T b, T *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T>(static_cast<float>(a[i]) + static_cast<float>(b));
    }
}

template <typename T>
void reference_subvs(size_t n, const T *a, T b, T *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T>(static_cast<float>(a[i]) - static_cast<float>(b));
    }
}

template <typename T>
void reference_mulvs(size_t n, const T *a, T b, T *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T>(static_cast<float>(a[i]) * static_cast<float>(b));
    }
}

template <typename T>
void reference_divvs(size_t n, const T *a, T b, T *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = static_cast<T>(static_cast<float>(a[i]) / static_cast<float>(b));
    }
}

// Explicit instantiations for the supported element types.
#define TQT_INSTANTIATE_ELEMENTWISE(T)                                   \
    template void reference_addvv<T>(size_t, const T *, const T *, T *); \
    template void reference_subvv<T>(size_t, const T *, const T *, T *); \
    template void reference_mulvv<T>(size_t, const T *, const T *, T *); \
    template void reference_divvv<T>(size_t, const T *, const T *, T *); \
    template void reference_addvs<T>(size_t, const T *, T, T *);         \
    template void reference_subvs<T>(size_t, const T *, T, T *);         \
    template void reference_mulvs<T>(size_t, const T *, T, T *);         \
    template void reference_divvs<T>(size_t, const T *, T, T *)

TQT_INSTANTIATE_ELEMENTWISE(float);
TQT_INSTANTIATE_ELEMENTWISE(float16_t);
TQT_INSTANTIATE_ELEMENTWISE(bfloat16_t);

#undef TQT_INSTANTIATE_ELEMENTWISE

}  // namespace test
}  // namespace torq_tile
