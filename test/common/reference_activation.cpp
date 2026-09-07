//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/reference_activation.h"

#include <cmath>

namespace torq_tile
{
namespace test
{

template <typename T>
void reference_relu(size_t m, size_t n, const T *a, size_t lda, T *c, size_t ldc)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float x = static_cast<float>(a[i * lda + j]);
            c[i * ldc + j] = static_cast<T>(x > 0.0f ? x : 0.0f);
        }
    }
}

template <typename T>
void reference_silu(size_t m, size_t n, const T *a, size_t lda, T *c, size_t ldc)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double x = static_cast<double>(static_cast<float>(a[i * lda + j]));
            c[i * ldc + j] = static_cast<T>(x / (1.0 + std::exp(-x)));
        }
    }
}

template <typename T>
void reference_gelu(size_t m, size_t n, const T *a, size_t lda, T *c, size_t ldc)
{
    constexpr double sqrt_2_over_pi = 0.7978845608028654;
    constexpr double cubic_coefficient = 0.044715;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double x = static_cast<double>(static_cast<float>(a[i * lda + j]));
            const double u = sqrt_2_over_pi * (x + cubic_coefficient * x * x * x);
            c[i * ldc + j] = static_cast<T>(0.5 * x * (1.0 + std::tanh(u)));
        }
    }
}

template <typename T>
void reference_reglu(size_t m, size_t n, const T *a, size_t lda, const T *b, size_t ldb, T *c,
                     size_t ldc)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float x = static_cast<float>(a[i * lda + j]);
            const float y = static_cast<float>(b[i * ldb + j]);
            c[i * ldc + j] = static_cast<T>((x > 0.0f ? x : 0.0f) * y);
        }
    }
}

template <typename T>
void reference_geglu(size_t m, size_t n, const T *a, size_t lda, const T *b, size_t ldb, T *c,
                     size_t ldc)
{
    constexpr double sqrt_2_over_pi = 0.7978845608028654;
    constexpr double cubic_coefficient = 0.044715;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double x = static_cast<double>(static_cast<float>(a[i * lda + j]));
            const double y = static_cast<double>(static_cast<float>(b[i * ldb + j]));
            const double u = sqrt_2_over_pi * (x + cubic_coefficient * x * x * x);
            c[i * ldc + j] = static_cast<T>((0.5 * x * (1.0 + std::tanh(u))) * y);
        }
    }
}

template <typename T>
void reference_swiglu(size_t m, size_t n, const T *a, size_t lda, const T *b, size_t ldb, T *c,
                      size_t ldc)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double x = static_cast<double>(static_cast<float>(a[i * lda + j]));
            const double y = static_cast<double>(static_cast<float>(b[i * ldb + j]));
            c[i * ldc + j] = static_cast<T>((x / (1.0 + std::exp(-x))) * y);
        }
    }
}

// Explicit instantiations for the supported element types.
#define TQT_INSTANTIATE_ACTIVATION(T)                                                            \
    template void reference_relu<T>(size_t, size_t, const T *, size_t, T *, size_t);             \
    template void reference_silu<T>(size_t, size_t, const T *, size_t, T *, size_t);             \
    template void reference_gelu<T>(size_t, size_t, const T *, size_t, T *, size_t);             \
    template void reference_reglu<T>(size_t, size_t, const T *, size_t, const T *, size_t, T *,  \
                                     size_t);                                                    \
    template void reference_geglu<T>(size_t, size_t, const T *, size_t, const T *, size_t, T *,  \
                                     size_t);                                                    \
    template void reference_swiglu<T>(size_t, size_t, const T *, size_t, const T *, size_t, T *, \
                                      size_t)

TQT_INSTANTIATE_ACTIVATION(float);
TQT_INSTANTIATE_ACTIVATION(float16_t);
TQT_INSTANTIATE_ACTIVATION(bfloat16_t);

#undef TQT_INSTANTIATE_ACTIVATION

}  // namespace test
}  // namespace torq_tile
