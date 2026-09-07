//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "src/common/tqt_common.h"

// Include micro-kernel variants
#include "tqt_swiglu_f32_f32_f32_1x4vl_rvv.h"
#include "tqt_swiglu_f32_f32_f32_interface.h"

namespace
{
/// Micro-kernel interface
const tqt_swiglu_f32_f32_f32_ukernel ukernel{tqt_get_m_step_swiglu_f32_f32_f32_1x4vl_rvv,
                                             tqt_get_n_step_swiglu_f32_f32_f32_1x4vl_rvv,
                                             tqt_run_swiglu_f32_f32_f32_1x4vl_rvv};

// Fill buffer with random values
void fill_random(size_t n, float *vec, float min_val, float max_val)
{
    for (size_t i = 0; i < n; ++i) {
        vec[i] = (float)(min_val + (max_val - min_val) * ((float)rand() / (float)RAND_MAX));
    }
}

// Reference implementation: c[i, j] = silu(a[i, j]) * b[i, j], honouring all three
// leading dimensions
void run_swiglu_ref(size_t m, size_t n, const float *a, size_t lda, const float *b, size_t ldb,
                    float *c, size_t ldc)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double x = (double)a[i * lda + j];
            const double y = (double)b[i * ldb + j];
            c[i * ldc + j] = (float)((x / (1.0 + exp(-x))) * y);
        }
    }
}

// Verify results element-wise over the [m, n] window
bool verify_results(size_t m, size_t n, size_t ldc, float tolerance, const float *ref,
                    const float *act)
{
    bool passed = true;
    float max_diff = 0.0f;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float r = (float)ref[i * ldc + j];
            const float v = (float)act[i * ldc + j];
            const float diff = fabsf(r - v);

            if (diff > max_diff) {
                max_diff = diff;
            }

            // Use relative tolerance for large values
            const float ref_abs = fabsf(r);
            const float tol = (ref_abs > 1.0f) ? tolerance * ref_abs : tolerance;
            if (diff > tol) {
#ifdef TQT_DEBUG
                printf("[%zu, %zu]: ref=%.6f vs act=%.6f (diff=%.6f) [FAIL]\n", i, j, r, v, diff);
#endif
                passed = false;
            }
        }
    }

#ifdef TQT_DEBUG
    printf("Max difference: %.6f\n", max_diff);
#endif

    return passed;
}

// Calculate cosine similarity over the [m, n] window
float calculate_cosine_similarity(size_t m, size_t n, size_t ldc, const float *ref,
                                  const float *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double r = (double)(float)ref[i * ldc + j];
            const double v = (double)(float)act[i * ldc + j];
            dot_product += r * v;
            ref_norm += r * r;
            act_norm += v * v;
        }
    }

    ref_norm = sqrt(ref_norm);
    act_norm = sqrt(act_norm);

    if (ref_norm > 0.0 && act_norm > 0.0) {
        return (float)(dot_product / (ref_norm * act_norm));
    }

    return 0.0f;
}

}  // namespace

int main()
{
    const size_t M = 16;
    const size_t N = 100;

    // Independent leading dimensions for both operands and the output. The padding
    // is what makes the strided traversal the one under test, which is the point of
    // this contract.
    const size_t lda = N + 4;
    const size_t ldb = N + 2;
    const size_t ldc = N + 3;

    printf("TEST[swiglu_f32_f32_f32]\n");
    printf("- ukernel: swiglu_f32_f32_f32_1x4vl_rvv\n");
    printf("- m_step: %zu, n_step: %zu\n", ukernel.get_m_step(), ukernel.get_n_step());
    printf("- shape: M=%zu N=%zu lda=%zu ldb=%zu ldc=%zu\n", M, N, lda, ldb, ldc);

    // Allocate memory
    float *a = (float *)malloc(M * lda * sizeof(float));
    float *b = (float *)malloc(M * ldb * sizeof(float));
    float *c_ref = (float *)malloc(M * ldc * sizeof(float));
    float *c_actual = (float *)malloc(M * ldc * sizeof(float));

    if (!a || !b || !c_ref || !c_actual) {
        fprintf(stderr, "Memory allocation failed\n");
        free(a);
        free(b);
        free(c_ref);
        free(c_actual);
        return 1;
    }

    // Initialize the whole extent of both operands, padding included
    fill_random(M * lda, a, -8.0f, 8.0f);
    fill_random(M * ldb, b, -4.0f, 4.0f);

    // Initialize both output buffers, including row padding
    for (size_t i = 0; i < M * ldc; ++i) {
        c_ref[i] = (float)0.0f;
        c_actual[i] = (float)0.0f;
    }

    // Run reference implementation
    run_swiglu_ref(M, N, a, lda, b, ldb, c_ref, ldc);

    // Run micro-kernel implementation
    ukernel.run_swiglu(M, N, a, lda, b, ldb, c_actual, ldc);

    // Verify results
    const float tolerance = 1e-4f;
    const bool exact = verify_results(M, N, ldc, tolerance, c_ref, c_actual);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M, N, ldc, c_ref, c_actual);
    const bool passed = exact && cosine_sim > 0.9999f;

    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(a);
    free(b);
    free(c_ref);
    free(c_actual);

    return passed ? 0 : 1;
}
