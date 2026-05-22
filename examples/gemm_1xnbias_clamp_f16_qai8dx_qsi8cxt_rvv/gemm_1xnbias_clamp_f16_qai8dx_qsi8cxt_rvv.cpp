//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, fp16 && vec-fp16.
#endif  // Architectural features check.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "src/common/tqt_common.h"

// Include micro-kernel variant
#include "tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_interface.h"

namespace
{

const tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    tqt_get_a_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    tqt_get_b_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
    tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
};

// Reference fp32 gemm: D = A[M,K] * B[N,K] + bias[1,N]
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

// Asymmetric per-row int8 quantization: scale = (max - min) / 255, zp = round(-128 - min/scale)
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

// Symmetric int8 quantization: scale = 2 * abs_max / 255 (zp = 0)
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

void fill_matrix_random(size_t rows, size_t cols, float *matrix, const float min, const float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float rand_val = min + (max - min) * ((float)rand() / (float)RAND_MAX);
        matrix[i] = rand_val;
    }
}

bool verify_results(size_t rows, size_t cols, const float tolerance, const float *ref,
                    const float *act)
{
    bool passed = true;
    for (size_t i = 0; i < rows * cols; ++i) {
        float diff = fabsf(ref[i] - act[i]);
        if (diff > tolerance) {
            passed = false;
        }
    }
    return passed;
}

float calculate_cosine_similarity(size_t size, const float *ref, const float *act)
{
    double dot_product = 0.0;
    double ref_norm = 0.0;
    double act_norm = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double r = (double)ref[i];
        double a = (double)act[i];
        dot_product += r * a;
        ref_norm += r * r;
        act_norm += a * a;
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
    const size_t M = 60;
    const size_t N = 63;
    const size_t K = 127;

    float *A_fp32 = (float *)malloc(M * K * sizeof(float));
    float *B_fp32 = (float *)malloc(N * K * sizeof(float));
    float *bias_fp32 = (float *)malloc(N * sizeof(float));
    float *D_ref_fp32 = (float *)malloc(M * N * sizeof(float));

    if (!A_fp32 || !B_fp32 || !bias_fp32 || !D_ref_fp32) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    fill_matrix_random(M, K, A_fp32, -1.0f, 1.0f);
    fill_matrix_random(N, K, B_fp32, -1.0f, 1.0f);
    fill_matrix_random(1, N, bias_fp32, -10.0f, 10.0f);

    run_gemm_ref(M, N, K, A_fp32, K, B_fp32, K, bias_fp32, D_ref_fp32, N);

    // A: per-row asymmetric int8 (M scales + M zero_points)
    float *scale_a = (float *)malloc(M * sizeof(float));
    int32_t *zp_a = (int32_t *)malloc(M * sizeof(int32_t));
    for (size_t r = 0; r < M; ++r) {
        get_quant_info_asym_i8(&A_fp32[r * K], K, &scale_a[r], &zp_a[r]);
    }

    // B: per-channel symmetric int8 (N scales)
    float *scale_b = (float *)malloc(N * sizeof(float));
    for (size_t ch = 0; ch < N; ++ch) {
        get_quant_info_sym_i8(&B_fp32[ch * K], K, &scale_b[ch]);
    }

    int8_t *A_i8 = (int8_t *)malloc(M * K * sizeof(int8_t));
    int8_t *BT_i8 = (int8_t *)malloc(N * K * sizeof(int8_t));
    float16_t *bias_fp16 = (float16_t *)malloc(N * sizeof(float16_t));
    float16_t *D_act_fp16 = (float16_t *)malloc(M * N * sizeof(float16_t));
    float *D_act_fp32 = (float *)malloc(M * N * sizeof(float));

    if (!scale_a || !zp_a || !scale_b || !A_i8 || !BT_i8 || !bias_fp16 || !D_act_fp16 ||
        !D_act_fp32) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Quantize A row by row
    for (size_t r = 0; r < M; ++r) {
        quantize_fp32_to_i8(&A_fp32[r * K], &A_i8[r * K], K, scale_a[r], zp_a[r]);
    }
    // Quantize B per channel (symmetric, zp=0)
    for (size_t ch = 0; ch < N; ++ch) {
        quantize_fp32_to_i8(&B_fp32[ch * K], &BT_i8[ch * K], K, scale_b[ch], 0);
    }

    // Convert bias to fp16
    for (size_t j = 0; j < N; ++j) {
        bias_fp16[j] = (float16_t)bias_fp32[j];
    }

    // Set up kernel-internal quantization params
    struct tqt_qai8dxp_qsi8cxp_params params;
    params.scale_a = scale_a;
    params.zero_point_a = zp_a;
    params.scale_b = scale_b;

    float16_t clamp_min = (float16_t)-65504.0f;
    float16_t clamp_max = (float16_t)65504.0f;

    memset(D_act_fp16, 0, M * N * sizeof(float16_t));

    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_m = std::min(M - m_idx, m_step);
            const size_t actual_n = std::min(N - n_idx, n_step);

            const uint8_t *a_ptr = (const uint8_t *)A_i8 + ukernel.get_a_offset(m_idx, 0, K);
            const uint8_t *b_ptr = (const uint8_t *)BT_i8 + ukernel.get_b_offset(n_idx, 0, K);
            const uint8_t *bias_ptr = (const uint8_t *)bias_fp16 + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D_act_fp16 + ukernel.get_d_offset(m_idx, n_idx, N);

            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, K, 0, nullptr, N, d_ptr, N,
                             bias_ptr, clamp_min, clamp_max, &params);
        }
    }

    // Convert fp16 output to fp32 for comparison
    for (size_t i = 0; i < M * N; ++i) {
        D_act_fp32[i] = (float)D_act_fp16[i];
    }

    const float tolerance = 2.5f;
    bool passed_abs = verify_results(M, N, tolerance, D_ref_fp32, D_act_fp32);

    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref_fp32, D_act_fp32);
    const bool passed = passed_abs && (cosine_sim > 0.998f);

    printf("TEST[gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_rvv]\n");
    printf("- ukernel: gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv\n");
    printf("- Shape: M=%zu N=%zu K=%zu\n", M, N, K);
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A_fp32);
    free(B_fp32);
    free(bias_fp32);
    free(bias_fp16);
    free(D_ref_fp32);
    free(D_act_fp16);
    free(D_act_fp32);
    free(A_i8);
    free(BT_i8);
    free(scale_a);
    free(zp_a);
    free(scale_b);

    return passed ? 0 : 1;
}
