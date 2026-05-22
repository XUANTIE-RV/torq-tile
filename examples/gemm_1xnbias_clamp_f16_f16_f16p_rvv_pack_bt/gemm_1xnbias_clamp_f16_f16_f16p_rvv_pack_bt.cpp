//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
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
#include "tqt_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_f16_f16_f16p_interface.h"

namespace
{

/// Micro-kernel interface (B-only packed: A original, B packed)
const tqt_gemm_1xnbias_clamp_f16_f16_f16p_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv,
    tqt_run_gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv};

void run_gemm_ref(size_t m, size_t n, size_t k, const float16_t *A, size_t lda, const float16_t *B,
                  size_t ldb, const float16_t *C, size_t ldc, float16_t *D, size_t ldd,
                  const float16_t *bias, float16_t clamp_min, float16_t clamp_max)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += (float)A[i * lda + p] * (float)B[j * ldb + p];
            }
            if (C)
                acc += (float)C[i * ldc + j];
            if (bias)
                acc += (float)bias[j];
            if (acc < (float)clamp_min)
                acc = (float)clamp_min;
            else if (acc > (float)clamp_max)
                acc = (float)clamp_max;
            D[i * ldd + j] = (float16_t)acc;
        }
    }
}

void fill_matrix_random(size_t rows, size_t cols, float16_t *matrix, float min, float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = (float16_t)(min + (max - min) * ((float)rand() / (float)RAND_MAX));
    }
}

float calculate_cosine_similarity(size_t size, const float16_t *ref, const float16_t *act)
{
    double dot = 0, rn = 0, an = 0;
    for (size_t i = 0; i < size; ++i) {
        float r = (float)ref[i], a = (float)act[i];
        dot += r * a;
        rn += r * r;
        an += a * a;
    }
    rn = sqrt(rn);
    an = sqrt(an);
    return (rn > 0 && an > 0) ? (float)(dot / (rn * an)) : 0.0f;
}

}  // namespace

int main()
{
    const size_t M = 60, N = 63, K = 127;

    float16_t *A = (float16_t *)malloc(M * K * sizeof(float16_t));
    float16_t *B = (float16_t *)malloc(N * K * sizeof(float16_t));
    float16_t *C = (float16_t *)malloc(M * N * sizeof(float16_t));
    float16_t *bias = (float16_t *)malloc(N * sizeof(float16_t));
    float16_t *D_ref = (float16_t *)malloc(M * N * sizeof(float16_t));
    float16_t *D = (float16_t *)malloc(M * N * sizeof(float16_t));

    if (!A || !B || !C || !bias || !D_ref || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    fill_matrix_random(M, K, A, -1.0f, 1.0f);
    fill_matrix_random(N, K, B, -1.0f, 1.0f);
    fill_matrix_random(M, N, C, -5.0f, 5.0f);
    fill_matrix_random(1, N, bias, -10.0f, 10.0f);

    float16_t clamp_min = (float16_t)(-60000.0f), clamp_max = (float16_t)(60000.0f);

    run_gemm_ref(M, N, K, A, K, B, K, C, N, D_ref, N, bias, clamp_min, clamp_max);

    // B-only packed: only pack B, A stays in original layout
    const size_t b_packed_size = ukernel.get_b_packed_size(N, K);
    float16_t *B_packed = (float16_t *)malloc(b_packed_size);
    if (!B_packed) {
        fprintf(stderr, "Packed matrix memory allocation failed\n");
        return 1;
    }

    ukernel.run_bt_pack(N, K, K, K, 0, B, B_packed);

    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_n = std::min(N - n_idx, n_step);

            const uint8_t *a_ptr = (const uint8_t *)A + ukernel.get_a_offset(m_idx, 0, K);
            const uint8_t *b_ptr =
                (const uint8_t *)B_packed + ukernel.get_b_packed_offset(n_idx, 0, K, actual_n);
            const uint8_t *c_ptr = (const uint8_t *)C + ukernel.get_c_offset(m_idx, n_idx, N);
            const uint8_t *bias_ptr = (const uint8_t *)bias + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, N);

            ukernel.run_gemm(std::min(M - m_idx, m_step), actual_n, K, a_ptr, K, 0, b_ptr, K, 0,
                             c_ptr, N, d_ptr, N, bias_ptr, clamp_min, clamp_max);
        }
    }

    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref, D);
    const bool passed = cosine_sim > 0.999f;

    printf("TEST[gemm_1xnbias_clamp_f16_f16_f16p_pack_bt]\n");
    printf("- ukernel: gemm_1xnbias_clamp_f16_f16_f16p_8x3vl_rvv\n");
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
