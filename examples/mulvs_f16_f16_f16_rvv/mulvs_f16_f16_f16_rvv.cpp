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
#include "tqt_mulvs_f16_f16_f16_8vl_rvv.h"
#include "tqt_mulvs_f16_f16_f16_interface.h"

namespace
{
/// Micro-kernel interface
const tqt_mulvs_f16_f16_f16_ukernel ukernel{tqt_get_n_step_mulvs_f16_f16_f16_8vl_rvv,
                                            tqt_run_mulvs_f16_f16_f16_8vl_rvv};

// Fill vector with random float16 values
void fill_random(size_t n, float16_t *vec, float min_val, float max_val)
{
    for (size_t i = 0; i < n; ++i) {
        vec[i] = (float16_t)(min_val + (max_val - min_val) * ((float)rand() / (float)RAND_MAX));
    }
}

// Reference implementation: c[i] = a[i] * b
void run_mulvs_ref(size_t n, const float16_t *a, float16_t b, float16_t *c)
{
    for (size_t i = 0; i < n; ++i) {
        c[i] = (float16_t)((float)a[i] * (float)b);
    }
}

// Verify results element-wise
bool verify_results(size_t n, float tolerance, const float16_t *ref, const float16_t *act)
{
    bool passed = true;
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;
    float max_diff_ref = 0.0f;
    float max_diff_act = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float ref_val = (float)ref[i];
        float act_val = (float)act[i];
        float diff = fabsf(ref_val - act_val);

        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
            max_diff_ref = ref_val;
            max_diff_act = act_val;
        }

        if (diff > tolerance) {
#ifdef TQT_DEBUG
            printf("[%zu]: ref=%.6f vs act=%.6f (diff=%.6f) [FAIL]\n", i, ref_val, act_val, diff);
#endif
            passed = false;
        } else {
#ifdef TQT_DEBUG
            printf("[%zu]: ref=%.6f vs act=%.6f (diff=%.6f)\n", i, ref_val, act_val, diff);
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
float calculate_cosine_similarity(size_t n, const float16_t *ref, const float16_t *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < n; ++i) {
        float ref_val = (float)ref[i];
        float act_val = (float)act[i];

        dot_product += ref_val * act_val;
        ref_norm += ref_val * ref_val;
        act_norm += act_val * act_val;
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
    const float16_t scalar_b = (float16_t)2.5f;

    // Allocate memory
    float16_t *a = (float16_t *)malloc(N * sizeof(float16_t));
    float16_t *c_ref = (float16_t *)malloc(N * sizeof(float16_t));
    float16_t *c_actual = (float16_t *)malloc(N * sizeof(float16_t));

    if (!a || !c_ref || !c_actual) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize input vector
    fill_random(N, a, -10.0f, 10.0f);

    // Run reference implementation
    run_mulvs_ref(N, a, scalar_b, c_ref);

    // Run micro-kernel implementation
    ukernel.run_mulvs(N, a, scalar_b, c_actual);

    // Verify results
    const float tolerance = 1e-1f;
    verify_results(N, tolerance, c_ref, c_actual);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(N, c_ref, c_actual);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[mulvs_f16_f16_f16]\n");
    printf("- ukernel: mulvs_f16_f16_f16_8vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(a);
    free(c_ref);
    free(c_actual);

    return passed ? 0 : 1;
}
