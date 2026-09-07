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
#include "src/gemm/gemm_u8_u8/tqt_params_u8_u8.h"

// Include micro-kernel variants
#include "tqt_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv.h"
#include "tqt_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_ukernel ukernel{
    tqt_get_m_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_n_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_a_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_b_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_c_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_dst_bias_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_d_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_d_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_a_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_get_b_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_run_a_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_run_b_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_run_bt_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
    tqt_run_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
};

// Reference implementation for verification
// D[m,n] = clamp(scvtf(C[m,n] + bias_m[m] + bias_n[n] + sum_k u8_A * u8_B) * scale
//                + dst_bias[n], min, max)
// A is [M,K] row-major uint8. B is [N,K] transposed uint8.
void run_gemm_ref(size_t m, size_t n, size_t k, const uint8_t *A, size_t lda, const uint8_t *B,
                  size_t ldb, const int32_t *C, size_t ldc, float *D, size_t ldd,
                  const int32_t *bias_m, const int32_t *bias_n, const float *dst_bias, float scale,
                  float clamp_min, float clamp_max)
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

            float f32_out = (float)acc * scale;

            if (dst_bias != nullptr) {
                f32_out += dst_bias[j];
            }

            if (f32_out < clamp_min) {
                f32_out = clamp_min;
            } else if (f32_out > clamp_max) {
                f32_out = clamp_max;
            }

            D[i * ldd + j] = f32_out;
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

// Fill matrix with random float values
void fill_matrix_random_float(size_t rows, size_t cols, float *matrix, float min, float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float r = min + (float)rand() / (float)RAND_MAX * (max - min);
        matrix[i] = r;
    }
}

// Calculate cosine similarity between two float vectors
float calculate_cosine_similarity(size_t size, const float *ref, const float *act)
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
    float *dst_bias = (float *)malloc(N * sizeof(float));
    float *D_ref = (float *)malloc(M * N * sizeof(float));
    float *D = (float *)malloc(M * N * sizeof(float));

    if (!A || !B || !C || !bias_m || !bias_n || !dst_bias || !D_ref || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    fill_matrix_random_uint8(M, K, A);
    fill_matrix_random_uint8(N, K, B);
    fill_matrix_random_int32(M, N, C, -1000, 1000);
    fill_matrix_random_int32(M, 1, bias_m, -1000, 1000);
    fill_matrix_random_int32(N, 1, bias_n, -1000, 1000);
    fill_matrix_random_float(N, 1, dst_bias, -1.0f, 1.0f);

    // Dequant scale and clamp
    const float scale = 0.001f;
    const float clamp_min = -FLT_MAX;
    const float clamp_max = FLT_MAX;

    // Params struct
    struct tqt_u8_u8_clamp_f32_params params;
    params.scale = scale;

    // Run reference implementation
    run_gemm_ref(M, N, K, A, K, B, K, C, N, D_ref, N, bias_m, bias_n, dst_bias, scale, clamp_min,
                 clamp_max);

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
        free(dst_bias);
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
            const uint8_t *dst_bias_ptr =
                (const uint8_t *)dst_bias + ukernel.get_dst_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, N);
#ifdef TQT_DEBUG
            printf("Processing a %zux%zu output block starting at (%zu, %zu)\n", actual_m, actual_n,
                   m_idx, n_idx);
#endif

            ukernel.run_gemm(actual_m, actual_n, K, a_packed_ptr, K, 0, b_packed_ptr, K, 0, c_ptr,
                             N, d_ptr, N, bias_m_ptr, bias_n_ptr, dst_bias_ptr, clamp_min,
                             clamp_max, &params);
        }
    }

    // Calculate cosine similarity
#ifdef TQT_DEBUG
    for (size_t dbg_i = 0; dbg_i < M * N; ++dbg_i) {
        size_t dbg_row = dbg_i / N;
        size_t dbg_col = dbg_i % N;
        float dbg_ref = (float)D_ref[dbg_i];
        float dbg_act = (float)D[dbg_i];
        float dbg_diff = fabsf(dbg_ref - dbg_act);
        if (dbg_diff > 1.0f)
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f) [FAIL]\n", dbg_row, dbg_col,
                   dbg_ref, dbg_act, dbg_diff);
        else
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f)\n", dbg_row, dbg_col, dbg_ref,
                   dbg_act, dbg_diff);
    }
#endif

    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref, D);
    const bool passed = cosine_sim > 0.999f;

    printf("TEST[gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_rvv]\n");
    printf("- ukernel: gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    free(bias_m);
    free(bias_n);
    free(dst_bias);
    free(D_ref);
    free(D);
    free(A_packed);
    free(B_packed);

    return passed ? 0 : 1;
}
