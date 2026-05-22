//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zvfbfwma)
#error This file must be compiled for riscv, riscv_vector, zvfbfwma.
#endif  // Architectural features check.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "src/common/tqt_common.h"
#include "tqt_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_f32_bf16_bf16_interface.h"

namespace
{
const tqt_gemm_1xnbias_clamp_f32_bf16_bf16_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    tqt_get_a_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    tqt_get_b_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv,
    tqt_run_gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv};

// Reference implementation (B is in standard format [k, n], A bf16, D/C/bias f16)
void run_gemm_ref(size_t m, size_t n, size_t k, const bfloat16_t *A, size_t lda,
                  const bfloat16_t *B, size_t ldb, const float *C, size_t ldc, float *D, size_t ldd,
                  const float *bias, float clamp_min, float clamp_max)
{
    const float clamp_min_f = static_cast<float>(clamp_min);
    const float clamp_max_f = static_cast<float>(clamp_max);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += static_cast<float>(A[i * lda + p]) * static_cast<float>(B[p * ldb + j]);
            }
            if (C)
                acc += static_cast<float>(C[i * ldc + j]);
            if (bias)
                acc += static_cast<float>(bias[j]);
            acc = std::fmax(acc, clamp_min_f);
            acc = std::fmin(acc, clamp_max_f);
            D[i * ldd + j] = static_cast<float>(acc);
        }
    }
}

template <typename T>
void fill_random(size_t count, T *data, float lo, float hi)
{
    for (size_t i = 0; i < count; ++i) {
        float r = lo + (hi - lo) * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        data[i] = static_cast<T>(r);
    }
}

float cosine_similarity(size_t size, const float *ref, const float *act)
{
    double dot = 0.0, n_ref = 0.0, n_act = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float r = static_cast<float>(ref[i]);
        float a = static_cast<float>(act[i]);
        dot += r * a;
        n_ref += r * r;
        n_act += a * a;
    }
    n_ref = std::sqrt(n_ref);
    n_act = std::sqrt(n_act);
    return (n_ref > 0.0 && n_act > 0.0) ? static_cast<float>(dot / (n_ref * n_act)) : 0.0f;
}
}  // namespace

int main()
{
    const size_t M = 60;
    const size_t N = 63;
    const size_t K = 127;

    bfloat16_t *A = (bfloat16_t *)malloc(M * K * sizeof(bfloat16_t));
    bfloat16_t *B = (bfloat16_t *)malloc(K * N * sizeof(bfloat16_t));
    float *C = (float *)malloc(M * N * sizeof(float));
    float *bias = (float *)malloc(N * sizeof(float));
    float *D_ref = (float *)malloc(M * N * sizeof(float));
    float *D = (float *)malloc(M * N * sizeof(float));

    fill_random(M * K, A, -1.0f, 1.0f);
    fill_random(K * N, B, -1.0f, 1.0f);
    fill_random(M * N, C, -5.0f, 5.0f);
    fill_random(N, bias, -10.0f, 10.0f);

    float clamp_min = static_cast<float>(-60000.0f);
    float clamp_max = static_cast<float>(60000.0f);

    run_gemm_ref(M, N, K, A, K, B, N, C, N, D_ref, N, bias, clamp_min, clamp_max);

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
            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, N, 0, c_ptr, N, d_ptr, N,
                             bias_ptr, clamp_min, clamp_max);
        }
    }

    const float cos = cosine_similarity(M * N, D_ref, D);
    const bool passed = cos > 0.9999f;

    printf("TEST[gemm_1xnbias_clamp_f32_bf16_bf16]\n");
    printf("- ukernel: gemm_1xnbias_clamp_f32_bf16_bf16_8x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cos);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A);
    free(B);
    free(C);
    free(bias);
    free(D_ref);
    free(D);
    return passed ? 0 : 1;
}
