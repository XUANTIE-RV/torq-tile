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
#include "tqt_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv.h"
#include "tqt_gemm_mx1bias_1xnbias_i32_u8p_u8p_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_gemm_mx1bias_1xnbias_i32_u8p_u8p_ukernel ukernel{
    tqt_get_m_step_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_n_step_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_a_packed_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_b_packed_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_c_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_d_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_d_size_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_a_packed_size_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_get_b_packed_size_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_run_a_pack_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_run_b_pack_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_run_bt_pack_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
    tqt_run_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
};

// Reference implementation for verification
// D[m,n] = C[m,n] + bias_m[m] + bias_n[n] + sum_k( A[m,k] * B[n,k] )
// A is [M,K] row-major uint8. B is [N,K] transposed uint8.
void run_gemm_ref(size_t m, size_t n, size_t k, const uint8_t *A, size_t lda, const uint8_t *B,
                  size_t ldb, const int32_t *C, size_t ldc, int32_t *D, size_t ldd,
                  const int32_t *bias_m, const int32_t *bias_n)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (size_t p = 0; p < k; ++p) {
                int32_t a_val = (int32_t)A[i * lda + p];
                int32_t b_val = (int32_t)B[j * ldb + p];
                acc += a_val * b_val;
            }

            if (C != nullptr) {
                acc += C[i * ldc + j];
            }

            acc += bias_m[i] + bias_n[j];
            D[i * ldd + j] = acc;
        }
    }
}

// Fill matrix with random uint8 values
void fill_matrix_random_uint8(size_t rows, size_t cols, uint8_t *matrix)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = (uint8_t)(rand() % 256);
    }
}

// Fill matrix with random int32 values
void fill_matrix_random_int32(size_t rows, size_t cols, int32_t *matrix, int32_t min, int32_t max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        int32_t r = (int32_t)(rand() % (int)(max - min + 1)) + min;
        matrix[i] = r;
    }
}

// Verify results (exact match for i32)
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
            printf("[%zu][%zu]: ref=%d vs act=%d (diff=%d) [FAIL]\n", row, col, ref_val, act_val,
                   diff);
#endif
            passed = false;
        } else {
#ifdef TQT_DEBUG
            size_t row = i / cols;
            size_t col = i % cols;
            printf("[%zu][%zu]: ref=%d vs act=%d (diff=%d)\n", row, col, ref_val, act_val, diff);
#endif
        }
    }

#ifdef TQT_DEBUG
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
    uint8_t *A = (uint8_t *)malloc(M * K * sizeof(uint8_t));
    uint8_t *B = (uint8_t *)malloc(N * K * sizeof(uint8_t));  // B is [N,K] transposed
    int32_t *C = (int32_t *)malloc(M * N * sizeof(int32_t));
    int32_t *bias_m = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *bias_n = (int32_t *)malloc(N * sizeof(int32_t));
    int32_t *D_ref = (int32_t *)malloc(M * N * sizeof(int32_t));
    int32_t *D = (int32_t *)malloc(M * N * sizeof(int32_t));

    if (!A || !B || !C || !bias_m || !bias_n || !D_ref || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    fill_matrix_random_uint8(M, K, A);
    fill_matrix_random_uint8(N, K, B);
    fill_matrix_random_int32(M, N, C, -1000, 1000);
    fill_matrix_random_int32(M, 1, bias_m, -1000, 1000);
    fill_matrix_random_int32(N, 1, bias_n, -1000, 1000);

    // Run reference implementation
    run_gemm_ref(M, N, K, A, K, B, K, C, N, D_ref, N, bias_m, bias_n);

    // Allocate memory for packed matrices
    const size_t a_packed_size = ukernel.get_a_packed_size(M, K);
    const size_t b_packed_size = ukernel.get_b_packed_size(N, K);

    uint8_t *A_packed = (uint8_t *)malloc(a_packed_size);
    uint8_t *B_packed = (uint8_t *)malloc(b_packed_size);

    if (!A_packed || !B_packed) {
        fprintf(stderr, "Packed matrix memory allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(bias_m);
        free(bias_n);
        free(D_ref);
        free(D);
        return 1;
    }

    // Pack A and B (B from [N, K] transposed layout)
    ukernel.run_a_pack(M, K, K, K, 0, A, A_packed);
    ukernel.run_bt_pack(N, K, K, K, 0, B, B_packed);

    // Run micro-kernel implementation using ukernel interface
    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_m = std::min(M - m_idx, m_step);
            const size_t actual_n = std::min(N - n_idx, n_step);

            const uint8_t *a_packed_ptr =
                (const uint8_t *)A_packed + ukernel.get_a_packed_offset(m_idx, 0, K, actual_m);
            const uint8_t *b_packed_ptr =
                (const uint8_t *)B_packed + ukernel.get_b_packed_offset(n_idx, 0, K, actual_n);
            const uint8_t *c_ptr = (const uint8_t *)C + ukernel.get_c_offset(m_idx, n_idx, N);
            const uint8_t *bias_m_ptr = (const uint8_t *)bias_m + ukernel.get_bias_m_offset(m_idx);
            const uint8_t *bias_n_ptr = (const uint8_t *)bias_n + ukernel.get_bias_n_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, N);
#ifdef TQT_DEBUG
            printf("Processing a %zux%zu output block starting at (%zu, %zu)\n", actual_m, actual_n,
                   m_idx, n_idx);
#endif

            ukernel.run_gemm(actual_m, actual_n, K, a_packed_ptr, K, 0, b_packed_ptr, K, 0, c_ptr,
                             N, d_ptr, N, bias_m_ptr, bias_n_ptr);
        }
    }

    // Verify results
    const bool exact_match = verify_results(M, N, D_ref, D);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref, D);
    const bool passed = exact_match && cosine_sim > 0.9999f;

    printf("TEST[gemm_mx1bias_1xnbias_i32_u8p_u8p_rvv]\n");
    printf("- ukernel: gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv\n");
    printf("- Exact match: %s\n", exact_match ? "YES" : "NO");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    free(bias_m);
    free(bias_n);
    free(D_ref);
    free(D);
    free(A_packed);
    free(B_packed);

    return passed ? 0 : 1;
}
