//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "src/common/tqt_common.h"

// Include micro-kernel variants
#include "tqt_sumrow_f16_f16_4x2vl_rvv.h"
#include "tqt_sumrow_f16_f16_interface.h"

namespace
{
/// Micro-kernel interface
const tqt_sumrow_f16_f16_ukernel ukernel{tqt_get_m_step_sumrow_f16_f16_4x2vl_rvv,
                                         tqt_get_n_step_sumrow_f16_f16_4x2vl_rvv,
                                         tqt_run_sumrow_f16_f16_4x2vl_rvv};

// Fill matrix with random values (stored as FP16)
void fill_random(size_t n, float16_t *vec, float min_val, float max_val)
{
    for (size_t i = 0; i < n; ++i) {
        vec[i] = (float16_t)(min_val + (max_val - min_val) * ((float)rand() / (float)RAND_MAX));
    }
}

// Reference implementation (FP32 accumulation): c[m] = sum over n of a[m * lda + n]
void run_sumrow_ref(size_t m, size_t n, const float16_t *a, size_t lda, float *c)
{
    for (size_t i = 0; i < m; ++i) {
        float acc = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            acc += (float)a[i * lda + j];
        }
        c[i] = acc;
    }
}

// Verify results element-wise (act comes from FP16 output, converted to float)
bool verify_results(size_t n, float tolerance, const float *ref, const float16_t *act)
{
    bool passed = true;
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;

    for (size_t i = 0; i < n; ++i) {
        float act_f = (float)act[i];
        float diff = fabsf(ref[i] - act_f);

        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }

        // Use relative tolerance for large values
        float ref_abs = fabsf(ref[i]);
        float tol = (ref_abs > 1.0f) ? tolerance * ref_abs : tolerance;
        if (diff > tol) {
#ifdef TQT_DEBUG
            printf("[%zu]: ref=%.6f vs act=%.6f (diff=%.6f) [FAIL]\n", i, ref[i], act_f, diff);
#endif
            passed = false;
        }
    }

#ifdef TQT_DEBUG
    printf("Max difference at [%zu]: diff=%.6f\n", max_diff_idx, max_diff);
#else
    (void)max_diff_idx;
#endif

    return passed;
}

// Calculate cosine similarity between reference (float) and actual (FP16) vectors
float calculate_cosine_similarity(size_t n, const float *ref, const float16_t *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double act_f = (double)(float)act[i];
        dot_product += ref[i] * act_f;
        ref_norm += ref[i] * ref[i];
        act_norm += act_f * act_f;
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
    const size_t M = 8;
    const size_t N = 256;

    // Allocate memory
    float16_t *a = (float16_t *)malloc(M * N * sizeof(float16_t));
    float *c_ref = (float *)malloc(M * sizeof(float));
    float16_t *c_actual = (float16_t *)malloc(M * sizeof(float16_t));

    if (!a || !c_ref || !c_actual) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize input matrix
    fill_random(M * N, a, -1.0f, 1.0f);

    // Run reference implementation
    run_sumrow_ref(M, N, a, N, c_ref);

    // Run micro-kernel implementation
    tqt_run_sumrow_f16_f16_4x2vl_rvv(M, N, a, N, c_actual, nullptr);

    // Verify results
    const float tolerance = 1e-2f;
    verify_results(M, tolerance, c_ref, c_actual);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M, c_ref, c_actual);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[sumrow_f16_f16]\n");
    printf("- ukernel: sumrow_f16_f16_4x2vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(a);
    free(c_ref);
    free(c_actual);

    return passed ? 0 : 1;
}
