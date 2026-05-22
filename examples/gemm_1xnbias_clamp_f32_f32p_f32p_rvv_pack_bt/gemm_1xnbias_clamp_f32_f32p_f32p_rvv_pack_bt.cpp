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
#include "tqt_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_f32_f32p_f32p_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_gemm_1xnbias_clamp_f32_f32p_f32p_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv,
    tqt_run_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv};

void run_gemm_ref(size_t m, size_t n, size_t k, const float *A, size_t lda, const float *B,
                  size_t ldb, const float *C, size_t ldc, float *D, size_t ldd, const float *bias,
                  float clamp_min, float clamp_max)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                float a_val = A[i * lda + p];
                float b_val = B[j * ldb + p];
                acc += a_val * b_val;
            }

            if (C != nullptr) {
                acc += C[i * ldc + j];
            }

            if (bias != nullptr) {
                acc += bias[j];
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

void fill_matrix_random(size_t rows, size_t cols, float *matrix, const float min, const float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float rand_val = min + (max - min) * ((float)rand() / (float)RAND_MAX);
        matrix[i] = rand_val;
    }
}

void print_matrix(size_t rows, size_t cols, const char *name, const float *matrix)
{
    printf("%s = [\n", name);
    for (size_t i = 0; i < rows; ++i) {
        printf("  [");
        for (size_t j = 0; j < cols; ++j) {
            printf("%.2f", matrix[i * cols + j]);
            if (j < cols - 1)
                printf(", ");
        }
        printf("],\n");
    }
    printf("]\n\n");
}

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
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f) ❌\n", row, col, ref_val, act_val,
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

float calculate_cosine_similarity(size_t size, const float *ref, const float *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;

    for (size_t i = 0; i < size; ++i) {
        float ref_val = ref[i];
        float act_val = act[i];

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
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(N * K * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    float *bias = (float *)malloc(N * sizeof(float));
    float *D_ref = (float *)malloc(M * N * sizeof(float));
    float *D = (float *)malloc(M * N * sizeof(float));

    if (!A || !B || !C || !bias || !D_ref || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    fill_matrix_random(M, K, A, -1.0f, 1.0f);
    fill_matrix_random(N, K, B, -1.0f, 1.0f);
    fill_matrix_random(M, N, C, -5.0f, 5.0f);
    fill_matrix_random(1, N, bias, -10.0f, 10.0f);

    float clamp_min = -1e6f;
    float clamp_max = 1e6f;

#ifdef TQT_DEBUG
    print_matrix(M, K, "A", A);
    print_matrix(N, K, "B", B);
    print_matrix(M, N, "C", C);
    print_matrix(1, N, "bias", bias);
#endif

    // Run reference implementation
    run_gemm_ref(M, N, K, A, K, B, K, C, N, D_ref, N, bias, clamp_min, clamp_max);

    const size_t a_packed_size = ukernel.get_a_packed_size(M, K);
    const size_t b_packed_size = ukernel.get_b_packed_size(N, K);

    float *A_packed = (float *)malloc(a_packed_size);
    float *B_packed = (float *)malloc(b_packed_size);

    if (!A_packed || !B_packed) {
        fprintf(stderr, "Packed matrix memory allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(bias);
        free(D_ref);
        free(D);
        return 1;
    }

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
            const uint8_t *bias_ptr = (const uint8_t *)bias + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, N);
#ifdef TQT_DEBUG
            printf("Processing a %zux%zu output block starting at (%zu, %zu)\n", actual_m, actual_n,
                   m_idx, n_idx);
#endif

            ukernel.run_gemm(actual_m, actual_n, K, a_packed_ptr, K, 0, b_packed_ptr, K, 0, c_ptr,
                             N, d_ptr, N, bias_ptr, clamp_min, clamp_max);
        }
    }

#ifdef TQT_DEBUG
    print_matrix(M, N, "D", D);
#endif

    verify_results(M, N, 1.0f, D_ref, D);

    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref, D);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[gemm_1xnbias_clamp_f32_f32p_f32p]\n");
    printf("- ukernel: gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    free(bias);
    free(D_ref);
    free(D);
    free(A_packed);
    free(B_packed);

    return passed ? 0 : 1;
}
