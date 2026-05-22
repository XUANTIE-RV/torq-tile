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
#include "src/gemm/gemm_i8_i8/tqt_utils_i8_i8.h"

// Include micro-kernel variants
#include "tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.h"
#include "tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_ukernel ukernel{
    tqt_get_m_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_n_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_a_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_b_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_c_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_bias_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_d_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_get_d_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
    tqt_run_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
};

// Reference fp32 GEMM: D = A[M,K] * B[K,N] + bias[M,1]
void run_gemm_ref(size_t m, size_t n, size_t k, const float *A, size_t lda, const float *B,
                  size_t ldb, const float *bias, float *D, size_t ldd)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += A[i * lda + p] * B[p * ldb + j];
            }
            if (bias != nullptr) {
                acc += bias[i];
            }
            D[i * ldd + j] = acc;
        }
    }
}

void find_min_max(const float *data, size_t size, float *min_val, float *max_val)
{
    float fmin = FLT_MAX;
    float fmax = -FLT_MAX;
    for (size_t i = 0; i < size; ++i) {
        if (data[i] < fmin)
            fmin = data[i];
        if (data[i] > fmax)
            fmax = data[i];
    }
    *min_val = fmin;
    *max_val = fmax;
}

void get_quant_info_asym_i8(const float *data, size_t size, float *scale, int32_t *zero_point)
{
    float fmin, fmax;
    find_min_max(data, size, &fmin, &fmax);

    fmax = fmax > 0.0f ? fmax : 0.0f;
    fmin = fmin < 0.0f ? fmin : 0.0f;

    float scale_tmp = (fmax - fmin) / 255.0f;
    if (scale_tmp == 0.0f) {
        *scale = 1.0f;
        *zero_point = 0;
        return;
    }

    float zp_tmp = -128.0f - fmin / scale_tmp;
    int32_t zp = (int32_t)nearbyintf(zp_tmp);
    if (zp == -128)
        zp = -127;

    *scale = scale_tmp;
    *zero_point = zp;
}

void get_quant_info_sym_i8(const float *data, size_t size, float *scale)
{
    float fmin, fmax;
    find_min_max(data, size, &fmin, &fmax);

    float abs_max = (fabsf(fmax) >= fabsf(fmin)) ? fabsf(fmax) : fabsf(fmin);
    float scale_tmp = 2.0f * abs_max / 255.0f;
    if (scale_tmp == 0.0f)
        scale_tmp = 1.0f;

    *scale = scale_tmp;
}

void quantize_fp32_to_i8(const float *src, int8_t *dst, size_t size, float scale,
                         int32_t zero_point)
{
    for (size_t i = 0; i < size; ++i) {
        float val = nearbyintf(src[i] / scale) + (float)zero_point;
        if (val > 127.0f)
            val = 127.0f;
        if (val < -128.0f)
            val = -128.0f;
        dst[i] = (int8_t)val;
    }
}

void quantize_fp32_to_i32(const float *src, int32_t *dst, size_t size, float scale,
                          int32_t zero_point)
{
    for (size_t i = 0; i < size; ++i) {
        float val = nearbyintf(src[i] / scale) + (float)zero_point;
        if (val > 2147483647.0f)
            val = 2147483647.0f;
        if (val < -2147483648.0f)
            val = -2147483648.0f;
        dst[i] = (int32_t)val;
    }
}

void dequantize_i8_to_fp32(const int8_t *src, float *dst, size_t size, float scale,
                           int32_t zero_point)
{
    for (size_t i = 0; i < size; ++i) {
        dst[i] = ((float)src[i] - (float)zero_point) * scale;
    }
}

// Fuse zero_point of asymmetric-quantized B into bias
// bias_fused[m] = bias_q[m] - zp_b * sum(A_i8[m, 0..K-1])
// A_i8 is [M, K], each row is a channel
void fuse_zp_to_bias(int32_t *bias_fused, const int32_t *bias, const int8_t *input_i8,
                     int32_t zero_point, size_t num_channels, size_t channel_size)
{
    for (size_t ch = 0; ch < num_channels; ++ch) {
        int32_t input_sum = 0;
        for (size_t k = 0; k < channel_size; ++k) {
            input_sum += (int32_t)input_i8[ch * channel_size + k];
        }
        bias_fused[ch] = bias[ch] - zero_point * input_sum;
    }
}

void fill_matrix_random(size_t rows, size_t cols, float *matrix, const float min, const float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float rand_val = min + (max - min) * ((float)rand() / (float)RAND_MAX);
        matrix[i] = rand_val;
    }
}

void print_matrix_fp32(size_t rows, size_t cols, const char *name, const float *matrix)
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

    // Allocate fp32 data: A[M,K], B[K,N], bias[M,1]
    float *A_fp32 = (float *)malloc(M * K * sizeof(float));
    float *B_fp32 = (float *)malloc(K * N * sizeof(float));
    float *bias_fp32 = (float *)malloc(M * sizeof(float));
    float *D_ref_fp32 = (float *)malloc(M * N * sizeof(float));

    if (!A_fp32 || !B_fp32 || !bias_fp32 || !D_ref_fp32) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize fp32 matrices
    fill_matrix_random(M, K, A_fp32, -1.0f, 1.0f);
    fill_matrix_random(K, N, B_fp32, -1.0f, 1.0f);
    fill_matrix_random(M, 1, bias_fp32, -10.0f, 10.0f);

#ifdef TQT_DEBUG
    print_matrix_fp32(M, K, "A_fp32", A_fp32);
    print_matrix_fp32(K, N, "B_fp32", B_fp32);
    print_matrix_fp32(M, 1, "bias_fp32", bias_fp32);
#endif

    // Run fp32 reference: D_ref = A[M,K] * B[K,N] + bias[M,1]
    run_gemm_ref(M, N, K, A_fp32, K, B_fp32, N, bias_fp32, D_ref_fp32, N);

    // Quantization parameters
    // B: asymmetric quantization (global scale_b, zp_b)
    float scale_b;
    int32_t zp_b;
    get_quant_info_asym_i8(B_fp32, K * N, &scale_b, &zp_b);

    // A: symmetric per-channel quantization (A is [M,K], each row is a channel)
    float *scale_a = (float *)malloc(M * sizeof(float));
    for (size_t ch = 0; ch < M; ++ch) {
        get_quant_info_sym_i8(&A_fp32[ch * K], K, &scale_a[ch]);
    }

    // D: asymmetric quantization (global scale_d, zp_d)
    float scale_d;
    int32_t zp_d;
    get_quant_info_asym_i8(D_ref_fp32, M * N, &scale_d, &zp_d);

    // Allocate quantized buffers
    int8_t *A_i8 = (int8_t *)malloc(M * K * sizeof(int8_t));
    int8_t *B_i8 = (int8_t *)malloc(K * N * sizeof(int8_t));
    int32_t *bias_i32 = (int32_t *)malloc(M * sizeof(int32_t));
    int32_t *bias_fused = (int32_t *)malloc(M * sizeof(int32_t));
    int8_t *D_i8 = (int8_t *)malloc(M * N * sizeof(int8_t));
    float *D_deq = (float *)malloc(M * N * sizeof(float));

    if (!scale_a || !A_i8 || !B_i8 || !bias_i32 || !bias_fused || !D_i8 || !D_deq) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Quantize B (global asymmetric)
    quantize_fp32_to_i8(B_fp32, B_i8, K * N, scale_b, zp_b);

    // Quantize A (per-channel symmetric, each row is a channel)
    for (size_t ch = 0; ch < M; ++ch) {
        quantize_fp32_to_i8(&A_fp32[ch * K], &A_i8[ch * K], K, scale_a[ch], 0);
    }

    // Quantize bias (Mx1, scale = scale_a[ch] * scale_b, zp = 0)
    for (size_t ch = 0; ch < M; ++ch) {
        float bias_scale = scale_a[ch] * scale_b;
        quantize_fp32_to_i32(&bias_fp32[ch], &bias_i32[ch], 1, bias_scale, 0);
    }

    // Fuse zero_point_b into bias: bias_fused[m] = bias[m] - zp_b * sum(A_i8[m, 0..K-1])
    fuse_zp_to_bias(bias_fused, bias_i32, A_i8, zp_b, M, K);

#ifdef TQT_DEBUG
    print_matrix_int8(M, K, "A_i8", A_i8);
    print_matrix_int8(K, N, "B_i8", B_i8);
    print_matrix_int32(M, 1, "bias_fused", bias_fused);
#endif

    // Set up quantization params
    struct tqt_qai8_qsi8dx_qai8_params params;
    params.scale_a = scale_a;
    params.quant_channel_a = (int32_t)M;
    params.scale_b = scale_b;
    params.zero_point_b = zp_b;
    params.scale_d = scale_d;
    params.zero_point_d = zp_d;

    int8_t clamp_min = -128;
    int8_t clamp_max = 127;

    memset(D_i8, 0, M * N * sizeof(int8_t));

    // Run micro-kernel implementation (bias uses fused version)
    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_m = std::min(M - m_idx, m_step);
            const size_t actual_n = std::min(N - n_idx, n_step);

            const uint8_t *a_ptr = (const uint8_t *)A_i8 + ukernel.get_a_offset(m_idx, 0, K);
            const uint8_t *b_ptr = (const uint8_t *)B_i8 + ukernel.get_b_offset(n_idx, 0, N);
            const uint8_t *bias_ptr = (const uint8_t *)bias_fused + ukernel.get_bias_offset(m_idx);
            uint8_t *d_ptr = (uint8_t *)D_i8 + ukernel.get_d_offset(m_idx, n_idx, N);
#ifdef TQT_DEBUG
            printf("Processing a %zux%zu output block starting at (%zu, %zu)\n", actual_m, actual_n,
                   m_idx, n_idx);
#endif
            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, N, 0, nullptr, N, d_ptr, N,
                             bias_ptr, clamp_min, clamp_max, &params);
        }
    }

    // Dequantize kernel output and compare with fp32 reference
    dequantize_i8_to_fp32(D_i8, D_deq, M * N, scale_d, zp_d);

#ifdef TQT_DEBUG
    print_matrix_fp32(M, N, "D_ref_fp32", D_ref_fp32);
    print_matrix_fp32(M, N, "D_deq", D_deq);
#endif

    // Verify results
    const float tolerance = 1.0f;
    verify_results(M, N, tolerance, D_ref_fp32, D_deq);

    // Calculate cosine similarity
    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref_fp32, D_deq);
    const bool passed = cosine_sim > 0.999f;

    printf("TEST[gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_rvv]\n");
    printf("- ukernel: gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A_fp32);
    free(B_fp32);
    free(bias_fp32);
    free(D_ref_fp32);
    free(D_deq);
    free(A_i8);
    free(B_i8);
    free(bias_i32);
    free(bias_fused);
    free(D_i8);
    free(scale_a);

    return passed ? 0 : 1;
}
