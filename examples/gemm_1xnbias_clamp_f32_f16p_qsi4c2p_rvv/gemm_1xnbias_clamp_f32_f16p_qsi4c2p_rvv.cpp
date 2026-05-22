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
#include "tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv,
    tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv};

// Reference implementation for verification (fp32 output, using unpacked B layout).
void run_gemm_ref(size_t m, size_t n, size_t k, size_t bl, const float16_t *A, size_t lda,
                  const uint8_t *B, size_t ldb, const float16_t *scale_b, size_t ldsb,
                  const float *C, size_t ldc, const float *bias, float *D, size_t ldd)
{
    const size_t b_row_bytes = ldb / 2;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = 0.0f;
            if (C != nullptr) {
                acc = C[i * ldc + j];
            }
            for (size_t p = 0; p < k; p++) {
                float a_val = static_cast<float>(A[i * lda + p]);
                uint8_t byte_val = B[j * b_row_bytes + p / 2];
                int8_t b_raw;
                if (p % 2 == 0) {
                    b_raw = static_cast<int8_t>(byte_val & 0x0F) - 8;
                } else {
                    b_raw = static_cast<int8_t>(byte_val >> 4) - 8;
                }
                size_t block_idx = p / bl;
                float scale_val = static_cast<float>(scale_b[j * ldsb + block_idx]);
                acc += a_val * static_cast<float>(b_raw) * scale_val;
            }
            if (bias) {
                acc += bias[j];
            }
            D[i * ldd + j] = acc;
        }
    }
}

void fill_matrix_random_fp16(size_t rows, size_t cols, float16_t *matrix, const float min,
                             const float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float rand_val =
            min + (max - min) * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        matrix[i] = static_cast<float16_t>(rand_val);
    }
}

void fill_matrix_random_fp32(size_t rows, size_t cols, float *matrix, const float min,
                             const float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float rand_val =
            min + (max - min) * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        matrix[i] = rand_val;
    }
}

void quantize_fp16_to_int4(size_t n, size_t k, size_t bl, const float16_t *src, uint8_t *dst,
                           float16_t *scale, size_t ld_scale)
{
    memset(dst, 0, n * (k / 2));

    for (size_t j = 0; j < n; ++j) {
        const size_t num_blocks = k / bl;
        for (size_t b = 0; b < num_blocks; ++b) {
            float max_value = 0.0f;
            float abs_max_value = 0.0f;
            for (size_t t = 0; t < bl; ++t) {
                float value = static_cast<float>(src[j * k + b * bl + t]);
                if (fabsf(value) > abs_max_value) {
                    abs_max_value = fabsf(value);
                    max_value = value;
                }
            }

            float fp32_scale = max_value / -8.0f;
            float inv_scale = (fp32_scale != 0.0f) ? (1.0f / fp32_scale) : 0.0f;
            scale[j * ld_scale + b] = static_cast<float16_t>(fp32_scale);

            for (size_t t = 0; t < bl; t += 2) {
                float value0 = static_cast<float>(src[j * k + b * bl + t]);
                float value1 = static_cast<float>(src[j * k + b * bl + t + 1]);
                uint8_t q0 = static_cast<uint8_t>(
                    fminf(static_cast<float>(static_cast<int8_t>(value0 * inv_scale + 8.5f)), 15));
                uint8_t q1 = static_cast<uint8_t>(
                    fminf(static_cast<float>(static_cast<int8_t>(value1 * inv_scale + 8.5f)), 15));
                dst[j * (k / 2) + (b * bl + t) / 2] = q0 | static_cast<uint8_t>(q1 << 4);
            }
        }
    }
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
        return static_cast<float>(dot_product / (ref_norm * act_norm));
    }

    return 0.0f;
}

}  // namespace

int main()
{
    // Matrix dimensions
    const size_t M = 60;
    const size_t N = 63;
    const size_t K = 128;
    const size_t BL = 16;
    const size_t lda = K;
    const size_t ldb = K;
    const size_t ldsb = K / BL;
    const size_t ldc = N;
    const size_t ldd = N;

    // Allocate memory
    float16_t *A = static_cast<float16_t *>(malloc(M * K * sizeof(float16_t)));
    float16_t *B_fp16 = static_cast<float16_t *>(malloc(N * K * sizeof(float16_t)));
    uint8_t *B = static_cast<uint8_t *>(malloc(N * (K / 2)));
    float16_t *scale_b = static_cast<float16_t *>(malloc(N * ldsb * sizeof(float16_t)));
    float *C = static_cast<float *>(malloc(M * N * sizeof(float)));
    float *bias = static_cast<float *>(malloc(N * sizeof(float)));
    float *D_ref = static_cast<float *>(malloc(M * N * sizeof(float)));
    float *D = static_cast<float *>(malloc(M * N * sizeof(float)));

    if (!A || !B_fp16 || !B || !scale_b || !C || !bias || !D_ref || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    srand(42);
    fill_matrix_random_fp16(M, K, A, -1.0f, 1.0f);
    fill_matrix_random_fp16(N, K, B_fp16, -1.0f, 1.0f);
    fill_matrix_random_fp32(M, N, C, -5.0f, 5.0f);
    fill_matrix_random_fp32(1, N, bias, -0.5f, 0.5f);

    quantize_fp16_to_int4(N, K, BL, B_fp16, B, scale_b, ldsb);

    const float clamp_min = -FLT_MAX;
    const float clamp_max = FLT_MAX;

    // Run reference (using original unpacked data)
    run_gemm_ref(M, N, K, BL, A, lda, B, ldb, scale_b, ldsb, C, ldc, bias, D_ref, ldd);

    // === Packed variant ===
    const size_t a_packed_size = ukernel.get_a_packed_size(M, K);
    const size_t b_packed_size = ukernel.get_b_packed_size(N, K);
    void *A_packed = malloc(a_packed_size);
    void *B_packed = malloc(b_packed_size);

    if (!A_packed || !B_packed) {
        fprintf(stderr, "Packed buffer allocation failed\n");
        return 1;
    }

    ukernel.run_a_pack(M, K, lda, K, 0, A, A_packed);
    ukernel.run_bt_pack(N, K, ldb, K, 0, B, B_packed);

    memset(D, 0, M * N * sizeof(float));
    const size_t m_step_val = ukernel.get_m_step();
    const size_t n_step_val = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step_val) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step_val) {
            const size_t actual_m = std::min(M - m_idx, m_step_val);
            const size_t actual_n = std::min(N - n_idx, n_step_val);

            const uint8_t *a_ptr =
                (const uint8_t *)A_packed + ukernel.get_a_packed_offset(m_idx, 0, K, actual_m);
            const uint8_t *b_ptr =
                (const uint8_t *)B_packed + ukernel.get_b_packed_offset(n_idx, 0, K, actual_n);
            const uint8_t *scale_ptr =
                (const uint8_t *)scale_b + ukernel.get_scale_b_offset(n_idx, 0, ldsb);
            const uint8_t *c_ptr = (const uint8_t *)C + ukernel.get_c_offset(m_idx, n_idx, N);
            const uint8_t *bias_ptr = (const uint8_t *)bias + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, N);

            ukernel.run_gemm(actual_m, actual_n, K, BL, a_ptr, K, 0, b_ptr, K, 0, scale_ptr, ldsb,
                             c_ptr, ldc, d_ptr, ldd, bias_ptr, clamp_min, clamp_max);
        }
    }

    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref, D);
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[gemm_1xnbias_clamp_f32_f16p_qsi4c2p]\n");
    printf("- ukernel: gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A);
    free(B_fp16);
    free(B);
    free(scale_b);
    free(C);
    free(bias);
    free(D_ref);
    free(D);
    free(A_packed);
    free(B_packed);

    return passed ? 0 : 1;
}
