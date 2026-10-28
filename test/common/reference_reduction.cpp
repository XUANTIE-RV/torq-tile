//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/reference_reduction.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// Sum reductions
// ============================================================================

void reference_sumrow_f32(size_t m, size_t n, const float *a, size_t lda, float *c,
                          const float *init)
{
    for (size_t i = 0; i < m; ++i) {
        float acc = (init != nullptr) ? init[i] : 0.0f;
        for (size_t j = 0; j < n; ++j) {
            acc += a[i * lda + j];
        }
        c[i] = acc;
    }
}

void reference_sumcol_f32(size_t m, size_t n, const float *a, size_t lda, float *c,
                          const float *init)
{
    for (size_t j = 0; j < n; ++j) {
        c[j] = (init != nullptr) ? init[j] : 0.0f;
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            c[j] += a[i * lda + j];
        }
    }
}

void reference_sumrow_f16(size_t m, size_t n, const float16_t *a, size_t lda, float16_t *c,
                          const float16_t *init)
{
    for (size_t i = 0; i < m; ++i) {
        float acc = (init != nullptr) ? (float)init[i] : 0.0f;
        for (size_t j = 0; j < n; ++j) {
            acc += (float)a[i * lda + j];
        }
        c[i] = (float16_t)acc;
    }
}

void reference_sumcol_f16(size_t m, size_t n, const float16_t *a, size_t lda, float16_t *c,
                          const float16_t *init)
{
    for (size_t j = 0; j < n; ++j) {
        float acc = (init != nullptr) ? (float)init[j] : 0.0f;
        for (size_t i = 0; i < m; ++i) {
            acc += (float)a[i * lda + j];
        }
        c[j] = (float16_t)acc;
    }
}

void reference_sumrow_bf16(size_t m, size_t n, const bfloat16_t *a, size_t lda, bfloat16_t *c,
                           const bfloat16_t *init)
{
    for (size_t i = 0; i < m; ++i) {
        float acc = (init != nullptr) ? (float)init[i] : 0.0f;
        for (size_t j = 0; j < n; ++j) {
            acc += (float)a[i * lda + j];
        }
        c[i] = (bfloat16_t)acc;
    }
}

void reference_sumcol_bf16(size_t m, size_t n, const bfloat16_t *a, size_t lda, bfloat16_t *c,
                           const bfloat16_t *init)
{
    for (size_t j = 0; j < n; ++j) {
        float acc = (init != nullptr) ? (float)init[j] : 0.0f;
        for (size_t i = 0; i < m; ++i) {
            acc += (float)a[i * lda + j];
        }
        c[j] = (bfloat16_t)acc;
    }
}

// ============================================================================
// Mean reductions
// ============================================================================

void reference_meanrow_f32(size_t m, size_t n, const float *a, size_t lda, float *c,
                           const float *init)
{
    const float recip = (n != 0) ? 1.0f / (float)n : 0.0f;
    for (size_t i = 0; i < m; ++i) {
        float acc = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            acc += a[i * lda + j];
        }
        const float base = (init != nullptr) ? init[i] : 0.0f;
        c[i] = base + acc * recip;
    }
}

void reference_meancol_f32(size_t m, size_t n, const float *a, size_t lda, float *c,
                           const float *init)
{
    const float recip = (m != 0) ? 1.0f / (float)m : 0.0f;
    for (size_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            acc += a[i * lda + j];
        }
        const float base = (init != nullptr) ? init[j] : 0.0f;
        c[j] = base + acc * recip;
    }
}

void reference_meanrow_f16(size_t m, size_t n, const float16_t *a, size_t lda, float16_t *c,
                           const float16_t *init)
{
    const float recip = (n != 0) ? 1.0f / (float)n : 0.0f;
    for (size_t i = 0; i < m; ++i) {
        float acc = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            acc += (float)a[i * lda + j];
        }
        const float base = (init != nullptr) ? (float)init[i] : 0.0f;
        c[i] = (float16_t)(base + acc * recip);
    }
}

void reference_meancol_f16(size_t m, size_t n, const float16_t *a, size_t lda, float16_t *c,
                           const float16_t *init)
{
    const float recip = (m != 0) ? 1.0f / (float)m : 0.0f;
    for (size_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            acc += (float)a[i * lda + j];
        }
        const float base = (init != nullptr) ? (float)init[j] : 0.0f;
        c[j] = (float16_t)(base + acc * recip);
    }
}

void reference_meanrow_bf16(size_t m, size_t n, const bfloat16_t *a, size_t lda, bfloat16_t *c,
                            const bfloat16_t *init)
{
    const float recip = (n != 0) ? 1.0f / (float)n : 0.0f;
    for (size_t i = 0; i < m; ++i) {
        float acc = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            acc += (float)a[i * lda + j];
        }
        const float base = (init != nullptr) ? (float)init[i] : 0.0f;
        c[i] = (bfloat16_t)(base + acc * recip);
    }
}

void reference_meancol_bf16(size_t m, size_t n, const bfloat16_t *a, size_t lda, bfloat16_t *c,
                            const bfloat16_t *init)
{
    const float recip = (m != 0) ? 1.0f / (float)m : 0.0f;
    for (size_t j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (size_t i = 0; i < m; ++i) {
            acc += (float)a[i * lda + j];
        }
        const float base = (init != nullptr) ? (float)init[j] : 0.0f;
        c[j] = (bfloat16_t)(base + acc * recip);
    }
}

}  // namespace test
}  // namespace torq_tile
