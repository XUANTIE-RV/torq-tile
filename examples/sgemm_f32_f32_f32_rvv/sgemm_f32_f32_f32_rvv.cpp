//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/common/tqt_common.h"

// Include micro-kernel variants
#include "tqt_sgemm_f32_f32_f32_8x3vl_rvv.h"
#include "tqt_sgemm_f32_f32_f32_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_sgemm_f32_f32_f32_ukernel ukernel{
    tqt_get_m_step_sgemm_f32_f32_f32_8x3vl_rvv,   tqt_get_n_step_sgemm_f32_f32_f32_8x3vl_rvv,
    tqt_get_a_offset_sgemm_f32_f32_f32_8x3vl_rvv, tqt_get_b_offset_sgemm_f32_f32_f32_8x3vl_rvv,
    tqt_get_c_offset_sgemm_f32_f32_f32_8x3vl_rvv, tqt_get_d_offset_sgemm_f32_f32_f32_8x3vl_rvv,
    tqt_get_d_size_sgemm_f32_f32_f32_8x3vl_rvv,   tqt_run_sgemm_f32_f32_f32_8x3vl_rvv};

// Reference SGEMM implementation for verification
// D = alpha * A * B + beta * C
// A is in row-major format [M, K], B is in row-major format [K, N]
inline void run_sgemm_ref(size_t m, size_t n, size_t k, const float *A, size_t lda, const float *B,
                          size_t ldb, const float *C, size_t ldc, float *D, size_t ldd, float alpha,
                          float beta)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += A[i * lda + p] * B[p * ldb + j];
            }
            float c_val = (C != nullptr) ? C[i * ldc + j] : 0.0f;
            D[i * ldd + j] = alpha * acc + beta * c_val;
        }
    }
}

// Fill matrix with random values
void fill_matrix_random(size_t rows, size_t cols, float *matrix, const float min, const float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float rand_val = min + (max - min) * ((float)rand() / (float)RAND_MAX);
        matrix[i] = rand_val;
    }
}

// Verify results
bool verify_results(size_t rows, size_t cols, const float tolerance, const float *ref,
                    const float *act)
{
    bool passed = true;
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;
    float max_diff_ref = 0.0f;
    float max_diff_act = 0.0f;

    for (size_t i = 0; i < rows * cols; ++i) {
        float ref_val = ref[i];
        float act_val = act[i];
        float diff = fabsf(ref_val - act_val);

        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
            max_diff_ref = ref_val;
            max_diff_act = act_val;
        }

        if (diff > tolerance) {
#ifdef TQT_DEBUG
            size_t row = i / cols;
            size_t col = i % cols;
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f) [FAIL]\n", row, col, ref_val,
                   act_val, diff);
#endif
            passed = false;
        } else {
#ifdef TQT_DEBUG
            size_t row = i / cols;
            size_t col = i % cols;
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f)\n", row, col, ref_val, act_val,
                   diff);
#endif
        }
    }

#ifdef TQT_DEBUG
    size_t max_row = max_diff_idx / cols;
    size_t max_col = max_diff_idx % cols;
    printf("Max difference at [%zu][%zu]: ref=%.6f, act=%.6f, diff=%.6f\n", max_row, max_col,
           max_diff_ref, max_diff_act, max_diff);
#endif

    return passed;
}

// Calculate cosine similarity between two vectors
float calculate_cosine_similarity(size_t size, const float *ref, const float *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double r = ref[i];
        double a = act[i];
        dot_product += r * a;
        ref_norm += r * r;
        act_norm += a * a;
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
    // Matrix dimensions
    const size_t M = 60;
    const size_t N = 63;
    const size_t K = 127;

    // BLAS-style scalars
    const float alpha = 1.5f;
    const float beta = 0.75f;

    // Allocate the memory
    // A [M, K] row-major, B [K, N] row-major
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    float *D_ref = (float *)malloc(M * N * sizeof(float));
    float *D = (float *)malloc(M * N * sizeof(float));

    if (!A || !B || !C || !D_ref || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    fill_matrix_random(M, K, A, -1.0f, 1.0f);
    fill_matrix_random(K, N, B, -1.0f, 1.0f);
    fill_matrix_random(M, N, C, -5.0f, 5.0f);

    // Run reference implementation
    run_sgemm_ref(M, N, K, A, K, B, N, C, N, D_ref, N, alpha, beta);

    // Run micro-kernel implementation using ukernel interface
    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_m = std::min(M - m_idx, m_step);
            const size_t actual_n = std::min(N - n_idx, n_step);

            const uint8_t *a_ptr = (const uint8_t *)A + ukernel.get_a_offset(m_idx, 0, K);
            const uint8_t *b_ptr = (const uint8_t *)B + ukernel.get_b_offset(n_idx, 0, N);
            const uint8_t *c_ptr = (const uint8_t *)C + ukernel.get_c_offset(m_idx, n_idx, N);
            uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, N);

            ukernel.run_sgemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, N, 0, c_ptr, N, d_ptr, N,
                              alpha, beta);
        }
    }

    // Verify results
    const float tolerance = 0.01f;
    verify_results(M, N, tolerance, D_ref, D);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref, D);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[sgemm_f32_f32_f32]\n");
    printf("- ukernel: sgemm_f32_f32_f32_8x3vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    free(D_ref);
    free(D);

    return passed ? 0 : 1;
}
