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

// Include micro-kernel variants (B-only packed)
#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv.h"
#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_interface.h"

namespace
{

/// Micro-kernel interface (B-only packed: A original, B packed)
const tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_run_b_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
    tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
};

void run_gemm_ref(size_t m, size_t n, size_t k, const float *A, size_t lda, const float *B,
                  size_t ldb, const float *bias, float *D, size_t ldd)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += A[i * lda + p] * B[p * ldb + j];
            }
            if (bias)
                acc += bias[j];
            D[i * ldd + j] = acc;
        }
    }
}

void find_min_max(const float *data, size_t size, float *min_val, float *max_val)
{
    float fmin = FLT_MAX, fmax = -FLT_MAX;
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
    float s = (fmax - fmin) / 255.0f;
    if (s == 0.0f) {
        *scale = 1.0f;
        *zero_point = 0;
        return;
    }
    int32_t zp = (int32_t)nearbyintf(-128.0f - fmin / s);
    if (zp == -128)
        zp = -127;
    *scale = s;
    *zero_point = zp;
}

void get_quant_info_sym_i8(const float *data, size_t size, float *scale)
{
    float fmin, fmax;
    find_min_max(data, size, &fmin, &fmax);
    float abs_max = (fabsf(fmax) >= fabsf(fmin)) ? fabsf(fmax) : fabsf(fmin);
    float s = 2.0f * abs_max / 255.0f;
    *scale = (s == 0.0f) ? 1.0f : s;
}

void quantize_fp32_to_i8(const float *src, int8_t *dst, size_t size, float scale, int32_t zp)
{
    for (size_t i = 0; i < size; ++i) {
        float val = nearbyintf(src[i] / scale) + (float)zp;
        val = std::max(-128.0f, std::min(127.0f, val));
        dst[i] = (int8_t)val;
    }
}

void quantize_fp32_to_i32(const float *src, int32_t *dst, size_t size, float scale, int32_t zp)
{
    for (size_t i = 0; i < size; ++i) {
        float val = nearbyintf(src[i] / scale) + (float)zp;
        val = std::max(-2147483648.0f, std::min(2147483647.0f, val));
        dst[i] = (int32_t)val;
    }
}

void dequantize_i8_to_fp32(const int8_t *src, float *dst, size_t size, float scale, int32_t zp)
{
    for (size_t i = 0; i < size; ++i) {
        dst[i] = ((float)src[i] - (float)zp) * scale;
    }
}

void fuse_zp_to_bias(int32_t *bias_fused, const int32_t *bias, const int8_t *B_i8,
                     int32_t zero_point, size_t n, size_t k, size_t ldb)
{
    for (size_t ch = 0; ch < n; ++ch) {
        int32_t weight_sum = 0;
        for (size_t ki = 0; ki < k; ++ki) {
            weight_sum += (int32_t)B_i8[ki * ldb + ch];
        }
        bias_fused[ch] = bias[ch] - zero_point * weight_sum;
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
        dot += (double)ref[i] * act[i];
        rn += (double)ref[i] * ref[i];
        an += (double)act[i] * act[i];
    }
    rn = sqrt(rn);
    an = sqrt(an);
    return (rn > 0 && an > 0) ? (float)(dot / (rn * an)) : 0.0f;
}

}  // namespace

int main()
{
    const size_t M = 60, N = 63, K = 127;

    float *A_fp32 = (float *)malloc(M * K * sizeof(float));
    float *B_fp32 = (float *)malloc(K * N * sizeof(float));
    float *bias_fp32 = (float *)malloc(N * sizeof(float));
    float *D_ref_fp32 = (float *)malloc(M * N * sizeof(float));

    if (!A_fp32 || !B_fp32 || !bias_fp32 || !D_ref_fp32) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    fill_matrix_random(M, K, A_fp32, -1.0f, 1.0f);
    fill_matrix_random(K, N, B_fp32, -1.0f, 1.0f);
    fill_matrix_random(1, N, bias_fp32, -10.0f, 10.0f);

    run_gemm_ref(M, N, K, A_fp32, K, B_fp32, N, bias_fp32, D_ref_fp32, N);

    // Quantization
    float scale_a;
    int32_t zp_a;
    get_quant_info_asym_i8(A_fp32, M * K, &scale_a, &zp_a);

    float *scale_b = (float *)malloc(N * sizeof(float));
    for (size_t ch = 0; ch < N; ++ch) {
        float *col = (float *)malloc(K * sizeof(float));
        for (size_t ki = 0; ki < K; ++ki)
            col[ki] = B_fp32[ki * N + ch];
        get_quant_info_sym_i8(col, K, &scale_b[ch]);
        free(col);
    }

    float scale_d;
    int32_t zp_d;
    get_quant_info_asym_i8(D_ref_fp32, M * N, &scale_d, &zp_d);

    int8_t *A_i8 = (int8_t *)malloc(M * K);
    int8_t *B_i8 = (int8_t *)malloc(K * N);
    int32_t *bias_i32 = (int32_t *)malloc(N * sizeof(int32_t));
    int32_t *bias_fused = (int32_t *)malloc(N * sizeof(int32_t));
    int8_t *D_i8 = (int8_t *)malloc(M * N);
    float *D_deq = (float *)malloc(M * N * sizeof(float));

    quantize_fp32_to_i8(A_fp32, A_i8, M * K, scale_a, zp_a);

    for (size_t ch = 0; ch < N; ++ch) {
        for (size_t ki = 0; ki < K; ++ki) {
            float val = nearbyintf(B_fp32[ki * N + ch] / scale_b[ch]);
            val = std::max(-128.0f, std::min(127.0f, val));
            B_i8[ki * N + ch] = (int8_t)val;
        }
    }

    for (size_t ch = 0; ch < N; ++ch) {
        quantize_fp32_to_i32(&bias_fp32[ch], &bias_i32[ch], 1, scale_a * scale_b[ch], 0);
    }

    fuse_zp_to_bias(bias_fused, bias_i32, B_i8, zp_a, N, K, N);

    struct tqt_qai8_qai8_qsi8cx_params params;
    params.scale_a = scale_a;
    params.zero_point_a = zp_a;
    params.scale_b = scale_b;
    params.quant_channel_b = (int32_t)N;
    params.scale_d = scale_d;
    params.zero_point_d = zp_d;

    int8_t clamp_min = -128, clamp_max = 127;
    memset(D_i8, 0, M * N);

    // B-only packed: only pack B, A stays in original layout
    const size_t b_packed_size = ukernel.get_b_packed_size(N, K);
    int8_t *B_packed = (int8_t *)malloc(b_packed_size);
    if (!B_packed) {
        fprintf(stderr, "Packed matrix memory allocation failed\n");
        return 1;
    }

    ukernel.run_b_pack(N, K, N, K, 0, B_i8, B_packed);

    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_m = std::min(M - m_idx, m_step);
            const size_t actual_n = std::min(N - n_idx, n_step);

            // A: original layout (not packed)
            const uint8_t *a_ptr = (const uint8_t *)A_i8 + ukernel.get_a_offset(m_idx, 0, K);
            // B: packed layout
            const uint8_t *b_ptr =
                (const uint8_t *)B_packed + ukernel.get_b_packed_offset(n_idx, 0, K, actual_n);
            const uint8_t *bias_ptr = (const uint8_t *)bias_fused + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D_i8 + ukernel.get_d_offset(m_idx, n_idx, N);

            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, K, 0, nullptr, N, d_ptr, N,
                             bias_ptr, clamp_min, clamp_max, &params);
        }
    }

    dequantize_i8_to_fp32(D_i8, D_deq, M * N, scale_d, zp_d);

    const float cosine_sim = calculate_cosine_similarity(M * N, D_ref_fp32, D_deq);
    const bool passed = cosine_sim > 0.999f;

    printf("TEST[gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp]\n");
    printf("- ukernel: gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv\n");
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
    free(scale_b);
    free(B_packed);

    return passed ? 0 : 1;
}
