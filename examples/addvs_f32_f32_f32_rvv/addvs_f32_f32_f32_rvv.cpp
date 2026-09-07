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
#include "tqt_addvs_f32_f32_f32_8vl_rvv.h"
#include "tqt_addvs_f32_f32_f32_interface.h"

namespace
{
/// Micro-kernel interface
const tqt_addvs_f32_f32_f32_ukernel ukernel{tqt_get_n_step_addvs_f32_f32_f32_8vl_rvv,
                                            tqt_run_addvs_f32_f32_f32_8vl_rvv};

// Fill vector with random float values
void fill_random(size_t n, float *vec, float min_val, float max_val)
{
    for (size_t i = 0; i < n; ++i) {
        vec[i] = min_val + (max_val - min_val) * ((float)rand() / (float)RAND_MAX);
    }
}

// Reference implementation: c[i] = a[i] + b
void run_addvs_ref(size_t n, const float *a, float b, float *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b;
    }
}

// Verify results element-wise
bool verify_results(size_t n, float tolerance, const float *ref, const float *act)
{
    bool passed = true;
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;
    float max_diff_ref = 0.0f;
    float max_diff_act = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float diff = fabsf(ref[i] - act[i]);

        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
            max_diff_ref = ref[i];
            max_diff_act = act[i];
        }

        if (diff > tolerance) {
#ifdef TQT_DEBUG
            printf("[%zu]: ref=%.6f vs act=%.6f (diff=%.6f) [FAIL]\n", i, ref[i], act[i], diff);
#endif
            passed = false;
        } else {
#ifdef TQT_DEBUG
            printf("[%zu]: ref=%.6f vs act=%.6f (diff=%.6f)\n", i, ref[i], act[i], diff);
#endif
        }
    }

#ifdef TQT_DEBUG
    printf("Max difference at [%zu]: ref=%.6f, act=%.6f, diff=%.6f\n", max_diff_idx, max_diff_ref,
           max_diff_act, max_diff);
#endif

    return passed;
}

// Calculate cosine similarity between two vectors
float calculate_cosine_similarity(size_t n, const float *ref, const float *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < n; ++i) {
        dot_product += ref[i] * act[i];
        ref_norm += ref[i] * ref[i];
        act_norm += act[i] * act[i];
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
    const size_t N = 1024;
    const float scalar_b = 2.5f;

    // Allocate memory
    float *a = (float *)malloc(N * sizeof(float));
    float *c_ref = (float *)malloc(N * sizeof(float));
    float *c_actual = (float *)malloc(N * sizeof(float));

    if (!a || !c_ref || !c_actual) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize input vector
    fill_random(N, a, -10.0f, 10.0f);

    // Run reference implementation
    run_addvs_ref(N, a, scalar_b, c_ref);

    // Run micro-kernel implementation
    ukernel.run_addvs(N, a, scalar_b, c_actual);

    // Verify results
    const float tolerance = 1e-6f;
    verify_results(N, tolerance, c_ref, c_actual);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(N, c_ref, c_actual);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[addvs_f32_f32_f32]\n");
    printf("- ukernel: addvs_f32_f32_f32_8vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(a);
    free(c_ref);
    free(c_actual);

    return passed ? 0 : 1;
}
