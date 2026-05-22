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
#include "tqt_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_f32_f32_f32p_interface.h"

namespace
{

/// Micro-kernel interface (B-only packed: A original, B packed)
const tqt_gemm_1xnbias_clamp_f32_f32_f32p_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv,
    tqt_run_gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv};

void run_gemm_ref(size_t m, size_t n, size_t k, const float *A, size_t lda, const float *B,
                  size_t ldb, const float *C, size_t ldc, float *D, size_t ldd, const float *bias,
                  float clamp_min, float clamp_max)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += A[i * lda + p] * B[j * ldb + p];
            }
            if (C)
                acc += C[i * ldc + j];
            if (bias)
                acc += bias[j];
            acc = std::max(clamp_min, std::min(clamp_max, acc));
            D[i * ldd + j] = acc;
        }
    }
}

void fill_matrix_random(size_t rows, size_t cols, float *matrix, float min, float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = min + (max - min) * ((float)rand() / (float)RAND_MAX);
    }
}

float calculate_cosine_similarity(size_t size, const float *ref, const float *act)
{
    double dot = 0, rn = 0, an = 0;
    for (size_t i = 0; i < size; ++i) {
        dot += ref[i] * act[i];
        rn += ref[i] * ref[i];
        an += act[i] * act[i];
    }
    rn = sqrt(rn);
    an = sqrt(an);
    return (rn > 0 && an > 0) ? (float)(dot / (rn * an)) : 0.0f;
}

}  // namespace

int main()
{
    const size_t M = 60, N = 63, K = 127;

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

    float clamp_min = -1e6f, clamp_max = 1e6f;

    run_gemm_ref(M, N, K, A, K, B, K, C, N, D_ref, N, bias, clamp_min, clamp_max);

    // B-only packed: only pack B, A stays in original layout
    const size_t b_packed_size = ukernel.get_b_packed_size(N, K);
    float *B_packed = (float *)malloc(b_packed_size);
    if (!B_packed) {
        fprintf(stderr, "Packed matrix memory allocation failed\n");
        return 1;
    }

    ukernel.run_bt_pack(N, K, K, K, 0, B, B_packed);

    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_m = std::min(M - m_idx, m_step);
            const size_t actual_n = std::min(N - n_idx, n_step);

            // A: original layout
            const uint8_t *a_ptr = (const uint8_t *)A + ukernel.get_a_offset(m_idx, 0, K);
            // B: packed layout
            const uint8_t *b_ptr =
                (const uint8_t *)B_packed + ukernel.get_b_packed_offset(n_idx, 0, K, actual_n);
            const uint8_t *c_ptr = (const uint8_t *)C + ukernel.get_c_offset(m_idx, n_idx, N);
            const uint8_t *bias_ptr = (const uint8_t *)bias + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, N);

            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, K, 0, c_ptr, N, d_ptr, N,
                             bias_ptr, clamp_min, clamp_max);
        }
    }

    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref, D);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[gemm_1xnbias_clamp_f32_f32_f32p_pack_bt]\n");
    printf("- ukernel: gemm_1xnbias_clamp_f32_f32_f32p_8x3vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    free(bias);
    free(D_ref);
    free(D);
    free(B_packed);

    return passed ? 0 : 1;
}
