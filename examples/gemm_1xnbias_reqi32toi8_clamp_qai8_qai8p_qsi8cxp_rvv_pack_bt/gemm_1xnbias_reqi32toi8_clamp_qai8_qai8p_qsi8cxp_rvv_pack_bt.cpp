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
#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv.h"
#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_get_a_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_run_b_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
    tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
};

// Reference fp32 gemm: D = A[M,K] * B[N,K]^T + bias[1,N]
// Note: B is [N,K] (transposed storage), bias is [1,N]
void run_gemm_ref(size_t m, size_t n, size_t k, const float *A, size_t lda, const float *B,
                  size_t ldb, const float *bias, float *D, size_t ldd)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += A[i * lda + p] * B[j * ldb + p];
            }
            if (bias != nullptr) {
                acc += bias[j];
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

// Fuse zero_point of asymmetric-quantized A into bias
// bias_fused[n] = bias_q[n] - zp_a * sum(BT_i8[n, 0..K-1])
// BT_i8 is [N, K] (transposed storage, each row is a channel)
void fuse_zp_to_bias(int32_t *bias_fused, const int32_t *bias, const int8_t *weight_i8,
                     int32_t zero_point, size_t num_channels, size_t channel_size)
{
    for (size_t ch = 0; ch < num_channels; ++ch) {
        int32_t weight_sum = 0;
        for (size_t k = 0; k < channel_size; ++k) {
            weight_sum += (int32_t)weight_i8[ch * channel_size + k];
        }
        bias_fused[ch] = bias[ch] - zero_point * weight_sum;
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
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f)\n", row, col, ref_val, act_val,
                   diff);
#endif
            passed = false;
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

    // Allocate fp32 data: A[M,K], B[N,K] (transposed), bias[1,N]
    float *A_fp32 = (float *)malloc(M * K * sizeof(float));
    float *B_fp32 = (float *)malloc(N * K * sizeof(float));
    float *bias_fp32 = (float *)malloc(N * sizeof(float));
    float *D_ref_fp32 = (float *)malloc(M * N * sizeof(float));

    if (!A_fp32 || !B_fp32 || !bias_fp32 || !D_ref_fp32) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize fp32 matrices
    fill_matrix_random(M, K, A_fp32, -1.0f, 1.0f);
    fill_matrix_random(N, K, B_fp32, -1.0f, 1.0f);
    fill_matrix_random(1, N, bias_fp32, -10.0f, 10.0f);

#ifdef TQT_DEBUG
    print_matrix_fp32(M, K, "A_fp32", A_fp32);
    print_matrix_fp32(N, K, "B_fp32", B_fp32);
    print_matrix_fp32(1, N, "bias_fp32", bias_fp32);
#endif

    // Run fp32 reference: D_ref = A[M,K] * B[N,K]^T + bias[1,N]
    run_gemm_ref(M, N, K, A_fp32, K, B_fp32, K, bias_fp32, D_ref_fp32, N);

    // Quantization parameters
    // A: asymmetric quantization (global scale_a, zp_a)
    float scale_a;
    int32_t zp_a;
    get_quant_info_asym_i8(A_fp32, M * K, &scale_a, &zp_a);

    // B: symmetric per-channel quantization (B is [N,K], each row is a channel)
    float *scale_b = (float *)malloc(N * sizeof(float));
    for (size_t ch = 0; ch < N; ++ch) {
        get_quant_info_sym_i8(&B_fp32[ch * K], K, &scale_b[ch]);
    }

    // D: asymmetric quantization (global scale_d, zp_d)
    float scale_d;
    int32_t zp_d;
    get_quant_info_asym_i8(D_ref_fp32, M * N, &scale_d, &zp_d);

    // Allocate quantized buffers
    int8_t *A_i8 = (int8_t *)malloc(M * K * sizeof(int8_t));
    int8_t *BT_i8 = (int8_t *)malloc(N * K * sizeof(int8_t));  // [N, K] layout
    int32_t *bias_i32 = (int32_t *)malloc(N * sizeof(int32_t));
    int32_t *bias_fused = (int32_t *)malloc(N * sizeof(int32_t));
    int8_t *D_i8 = (int8_t *)malloc(M * N * sizeof(int8_t));
    float *D_deq = (float *)malloc(M * N * sizeof(float));

    if (!scale_b || !A_i8 || !BT_i8 || !bias_i32 || !bias_fused || !D_i8 || !D_deq) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Quantize A (global asymmetric)
    quantize_fp32_to_i8(A_fp32, A_i8, M * K, scale_a, zp_a);

    // Quantize B (per-channel symmetric, each row of B[N,K] is a channel)
    for (size_t ch = 0; ch < N; ++ch) {
        quantize_fp32_to_i8(&B_fp32[ch * K], &BT_i8[ch * K], K, scale_b[ch], 0);
    }

    // Quantize bias (1xN, scale = scale_a * scale_b[ch], zp = 0)
    for (size_t ch = 0; ch < N; ++ch) {
        float bias_scale = scale_a * scale_b[ch];
        quantize_fp32_to_i32(&bias_fp32[ch], &bias_i32[ch], 1, bias_scale, 0);
    }

    // Fuse zero_point_a into bias: bias_fused[n] = bias[n] - zp_a * sum(BT_i8[n, 0..K-1])
    fuse_zp_to_bias(bias_fused, bias_i32, BT_i8, zp_a, N, K);

#ifdef TQT_DEBUG
    print_matrix_int8(M, K, "A_i8", A_i8);
    print_matrix_int8(N, K, "BT_i8", BT_i8);
    print_matrix_int32(1, N, "bias_fused", bias_fused);
#endif

    // Set up quantization params
    struct tqt_qai8_qai8_qsi8cx_params params;
    params.scale_a = scale_a;
    params.zero_point_a = zp_a;
    params.scale_b = scale_b;
    params.quant_channel_b = (int32_t)N;
    params.scale_d = scale_d;
    params.zero_point_d = zp_d;

    int8_t clamp_min = -128;
    int8_t clamp_max = 127;

    memset(D_i8, 0, M * N * sizeof(int8_t));

    // Pack A and B
    const size_t a_packed_size = ukernel.get_a_packed_size(M, K);
    const size_t b_packed_size = ukernel.get_b_packed_size(N, K);

    int8_t *A_packed = (int8_t *)malloc(a_packed_size);
    int8_t *B_packed = (int8_t *)malloc(b_packed_size);

    if (!A_packed || !B_packed) {
        fprintf(stderr, "Packed matrix memory allocation failed\n");
        return 1;
    }

    ukernel.run_a_pack(M, K, K, K, 0, A_i8, A_packed);
    ukernel.run_bt_pack(N, K, K, K, 0, BT_i8, B_packed);  // B from [N, K] (transposed)

    // Run micro-kernel implementation
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
            const uint8_t *bias_ptr = (const uint8_t *)bias_fused + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D_i8 + ukernel.get_d_offset(m_idx, n_idx, N);
#ifdef TQT_DEBUG
            printf("Processing a %zux%zu output block starting at (%zu, %zu)\n", actual_m, actual_n,
                   m_idx, n_idx);
#endif
            ukernel.run_gemm(actual_m, actual_n, K, a_packed_ptr, K, 0, b_packed_ptr, K, 0, nullptr,
                             N, d_ptr, N, bias_ptr, clamp_min, clamp_max, &params);
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

    printf("TEST[gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_rvv_pack_bt]\n");
    printf("- ukernel: gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A_fp32);
    free(B_fp32);
    free(bias_fp32);
    free(D_ref_fp32);
    free(D_deq);
    free(A_i8);
    free(BT_i8);
    free(bias_i32);
    free(bias_fused);
    free(D_i8);
    free(scale_b);
    free(A_packed);
    free(B_packed);

    return passed ? 0 : 1;
}
