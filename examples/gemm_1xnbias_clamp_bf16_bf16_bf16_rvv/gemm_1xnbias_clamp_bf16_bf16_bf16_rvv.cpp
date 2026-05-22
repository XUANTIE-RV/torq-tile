//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zvfbfwma)
#error This file must be compiled for riscv, riscv_vector, zvfbfwma.
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
#include "tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_interface.h"

namespace
{
/// Micro-kernel interface
const tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv,
    tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv};

// Reference implementation for verification
// Note: B is in standard format (stored as k x n)
void run_gemm_ref(size_t m, size_t n, size_t k, const bfloat16_t *A, size_t lda,
                  const bfloat16_t *B, size_t ldb, const bfloat16_t *C, size_t ldc, bfloat16_t *D,
                  size_t ldd, const bfloat16_t *bias, bfloat16_t clamp_min, bfloat16_t clamp_max)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            bfloat16_t acc = static_cast<bfloat16_t>(0.0f);
            // B is in standard format: B[p, j]
            for (size_t p = 0; p < k; ++p) {
                bfloat16_t a_val = A[i * lda + p];
                bfloat16_t b_val = B[p * ldb + j];  // B is k x n
                acc =
                    static_cast<bfloat16_t>(static_cast<float>(acc) +
                                            static_cast<float>(a_val) * static_cast<float>(b_val));
            }

            // Add C if provided
            if (C != nullptr) {
                acc = static_cast<bfloat16_t>(static_cast<float>(acc) +
                                              static_cast<float>(C[i * ldc + j]));
            }

            // Add bias if provided
            if (bias != nullptr) {
                acc =
                    static_cast<bfloat16_t>(static_cast<float>(acc) + static_cast<float>(bias[j]));
            }

            // Apply clamp
            float acc_f = static_cast<float>(acc);
            if (acc_f < static_cast<float>(clamp_min)) {
                acc = clamp_min;
            } else if (acc_f > static_cast<float>(clamp_max)) {
                acc = clamp_max;
            }

            D[i * ldd + j] = acc;
        }
    }
}

// Fill matrix with random values
void fill_matrix_random(size_t rows, size_t cols, bfloat16_t *matrix, const float min,
                        const float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float rand_val = min + (max - min) * ((float)rand() / (float)RAND_MAX);
        matrix[i] = (bfloat16_t)rand_val;
    }
}

// Print matrix
void print_matrix(size_t rows, size_t cols, const char *name, const bfloat16_t *matrix)
{
    printf("%s = [\n", name);
    for (size_t i = 0; i < rows; ++i) {
        printf("  [");
        for (size_t j = 0; j < cols; ++j) {
            printf("%.2f", (float)matrix[i * cols + j]);
            if (j < cols - 1)
                printf(", ");
        }
        printf("],\n");
    }
    printf("]\n\n");
}

// Verify results
bool verify_results(size_t rows, size_t cols, const float tolerance, const bfloat16_t *ref,
                    const bfloat16_t *act)
{
    bool passed = true;
    float max_diff = 0.0f;
    size_t max_diff_idx = 0;
    float max_diff_ref = 0.0f;
    float max_diff_act = 0.0f;

    for (size_t i = 0; i < rows * cols; ++i) {
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
            size_t row = i / cols;
            size_t col = i % cols;
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f)\n", row, col, ref_val, act_val,
                   diff);
#endif
            passed = false;
        } else {
#ifdef TQT_DEBUG
            size_t row = i / cols;
            size_t col = i % cols;
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f) ✅\n", row, col, ref_val, act_val,
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
float calculate_cosine_similarity(size_t size, const bfloat16_t *ref, const bfloat16_t *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < size; ++i) {
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
    // Matrix dimensions
    const size_t M = 60;
    const size_t N = 63;
    const size_t K = 127;

    // Allocate the memory
    bfloat16_t *A = (bfloat16_t *)malloc(M * K * sizeof(bfloat16_t));
    bfloat16_t *B = (bfloat16_t *)malloc(K * N * sizeof(bfloat16_t));  // B is k x n
    bfloat16_t *C = (bfloat16_t *)malloc(M * N * sizeof(bfloat16_t));
    bfloat16_t *bias = (bfloat16_t *)malloc(N * sizeof(bfloat16_t));
    bfloat16_t *D_ref = (bfloat16_t *)malloc(M * N * sizeof(bfloat16_t));
    bfloat16_t *D = (bfloat16_t *)malloc(M * N * sizeof(bfloat16_t));

    if (!A || !B || !C || !bias || !D_ref || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    fill_matrix_random(M, K, A, -1.0f, 1.0f);
    fill_matrix_random(K, N, B, -1.0f, 1.0f);  // B is k x n
    fill_matrix_random(M, N, C, -5.0f, 5.0f);
    fill_matrix_random(1, N, bias, -10.0f, 10.0f);

    // Clamp range
    bfloat16_t clamp_min = -60000;
    bfloat16_t clamp_max = 60000;

#ifdef TQT_DEBUG
    print_matrix(M, K, "A", A);
    print_matrix(K, N, "B", B);
    print_matrix(M, N, "C", C);
    print_matrix(1, N, "bias", bias);
#endif  // TQT_DEBUG

    // Run reference implementation
    run_gemm_ref(M, N, K, A, K, B, N, C, N, D_ref, N, bias, clamp_min, clamp_max);

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
            const uint8_t *bias_ptr = (const uint8_t *)bias + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, N);
#ifdef TQT_DEBUG
            printf("Processing a %zux%zu output block starting at (%zu, %zu)\n", actual_m, actual_n,
                   m_idx, n_idx);
#endif
            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, N, 0, c_ptr, N, d_ptr, N,
                             bias_ptr, clamp_min, clamp_max);
        }
    }

#ifdef TQT_DEBUG
    print_matrix(M, N, "D_ref", D_ref);
    print_matrix(M, N, "D", D);
#endif  // TQT_DEBUG

    // Verify results
    const float tolerance = 1.0f;
    verify_results(M, N, tolerance, D_ref, D);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref, D);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[gemm_1xnbias_clamp_bf16_bf16_bf16]\n");
    printf("- ukernel: gemm_1xnbias_clamp_bf16_bf16_bf16_8x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    free(bias);
    free(D_ref);
    free(D);

    return passed ? 0 : 1;
}
