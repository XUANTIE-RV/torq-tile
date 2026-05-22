//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/common/tqt_common.h"

// Include micro-kernel variants
#include "tqt_gemm_clamp_i32_i8_i8_4x1vl_rvv.h"
#include "tqt_gemm_clamp_i32_i8_i8_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_gemm_clamp_i32_i8_i8_ukernel ukernel{
    tqt_get_m_step_gemm_clamp_i32_i8_i8_4x1vl_rvv,
    tqt_get_n_step_gemm_clamp_i32_i8_i8_4x1vl_rvv,
    tqt_get_a_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
    tqt_get_b_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
    tqt_get_c_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
    tqt_get_d_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
    tqt_get_d_size_gemm_clamp_i32_i8_i8_4x1vl_rvv,
    tqt_run_gemm_clamp_i32_i8_i8_4x1vl_rvv,
};

// Reference implementation for verification
void run_gemm_ref(size_t m, size_t n, size_t k, const int8_t *A, size_t lda, const int8_t *B,
                  size_t ldb, const int32_t *C, size_t ldc, int32_t *D, size_t ldd,
                  int32_t clamp_min, int32_t clamp_max)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (size_t p = 0; p < k; ++p) {
                int32_t a_val = (int32_t)A[i * lda + p];
                int32_t b_val = (int32_t)B[p * ldb + j];
                acc += a_val * b_val;
            }

            if (C != nullptr) {
                acc += C[i * ldc + j];
            }

            if (acc < clamp_min) {
                acc = clamp_min;
            } else if (acc > clamp_max) {
                acc = clamp_max;
            }

            D[i * ldd + j] = acc;
        }
    }
}

// Fill matrix with random int8 values
void fill_matrix_random_int8(size_t rows, size_t cols, int8_t *matrix, int8_t min, int8_t max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        int r = rand() % (max - min + 1) + min;
        matrix[i] = (int8_t)r;
    }
}

// Print int8 matrix
void print_matrix_int8(size_t rows, size_t cols, const char *name, const int8_t *matrix)
{
    printf("%s = [\n", name);
    for (size_t i = 0; i < rows; ++i) {
        printf("  [");
        for (size_t j = 0; j < cols; ++j) {
            printf("%d", matrix[i * cols + j]);
            if (j < cols - 1)
                printf(", ");
        }
        printf("],\n");
    }
    printf("]\n\n");
}

// Print int32 matrix
void print_matrix_int32(size_t rows, size_t cols, const char *name, const int32_t *matrix)
{
    printf("%s = [\n", name);
    for (size_t i = 0; i < rows; ++i) {
        printf("  [");
        for (size_t j = 0; j < cols; ++j) {
            printf("%d", matrix[i * cols + j]);
            if (j < cols - 1)
                printf(", ");
        }
        printf("],\n");
    }
    printf("]\n\n");
}

// Fill matrix with random int32 values
void fill_matrix_random_int32(size_t rows, size_t cols, int32_t *matrix, int32_t min, int32_t max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        int32_t r = (int32_t)(rand() % (int)(max - min + 1)) + min;
        matrix[i] = r;
    }
}

// Verify results
bool verify_results(size_t rows, size_t cols, const int32_t *ref, const int32_t *act)
{
    bool passed = true;
    int32_t max_diff = 0;
    size_t max_diff_idx = 0;
    int32_t max_diff_ref = 0;
    int32_t max_diff_act = 0;

    for (size_t i = 0; i < rows * cols; ++i) {
        int32_t ref_val = ref[i];
        int32_t act_val = act[i];
        int32_t diff = ref_val - act_val;
        if (diff < 0)
            diff = -diff;

        // Track maximum difference
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
            max_diff_ref = ref_val;
            max_diff_act = act_val;
        }

        if (diff != 0) {
#ifdef TQT_DEBUG
            size_t row = i / cols;
            size_t col = i % cols;
            printf("[%zu][%zu]: ref=%d vs act=%d (diff=%d) ❌\n", row, col, ref_val, act_val, diff);
#endif
            passed = false;
        } else {
#ifdef TQT_DEBUG
            size_t row = i / cols;
            size_t col = i % cols;
            printf("[%zu][%zu]: ref=%d vs act=%d (diff=%d) ✅\n", row, col, ref_val, act_val, diff);
#endif
        }
    }

#ifdef TQT_DEBUG
    // Print maximum difference
    size_t max_row = max_diff_idx / cols;
    size_t max_col = max_diff_idx % cols;
    printf("Max difference at [%zu][%zu]: ref=%d, act=%d, diff=%d\n", max_row, max_col,
           max_diff_ref, max_diff_act, max_diff);
#endif

    return passed;
}

// Calculate cosine similarity between two vectors
float calculate_cosine_similarity(size_t size, const int32_t *ref, const int32_t *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double ref_val = (double)ref[i];
        double act_val = (double)act[i];

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

    // Allocate memory
    int8_t *A = (int8_t *)malloc(M * K * sizeof(int8_t));
    int8_t *B = (int8_t *)malloc(K * N * sizeof(int8_t));
    int32_t *C = (int32_t *)malloc(M * N * sizeof(int32_t));
    int32_t *D_ref = (int32_t *)malloc(M * N * sizeof(int32_t));
    int32_t *D = (int32_t *)malloc(M * N * sizeof(int32_t));

    if (!A || !B || !C || !D_ref || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    fill_matrix_random_int8(M, K, A, -50, 50);
    fill_matrix_random_int8(K, N, B, -50, 50);
    fill_matrix_random_int32(M, N, C, -1000, 1000);

    // Clamp range
    int32_t clamp_min = -100000;
    int32_t clamp_max = 100000;

#ifdef TQT_DEBUG
    print_matrix_int8(M, K, "A", A);
    print_matrix_int8(K, N, "B", B);
    print_matrix_int32(M, N, "C", C);
#endif

    // Run reference implementation
    run_gemm_ref(M, N, K, A, K, B, N, C, N, D_ref, N, clamp_min, clamp_max);

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
#ifdef TQT_DEBUG
            printf("Processing a %zux%zu output block starting at (%zu, %zu)\n", actual_m, actual_n,
                   m_idx, n_idx);
#endif

            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, N, 0, c_ptr, N, d_ptr, N,
                             clamp_min, clamp_max);
        }
    }

#ifdef TQT_DEBUG
    print_matrix_int32(M, N, "D", D);
#endif

    // Verify results
    verify_results(M, N, D_ref, D);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref, D);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[gemm_clamp_i32_i8_i8_rvv]\n");
    printf("- ukernel: gemm_clamp_i32_i8_i8_4x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    free(D_ref);
    free(D);

    return passed ? 0 : 1;
}
