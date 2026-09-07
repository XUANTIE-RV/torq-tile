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
#include "src/gemm/gemm_i8_i2/tqt_params_i8_i2.h"
#include "src/gemm/gemm_i8_i2/tqt_utils_i8_i2.h"

// Include micro-kernel variants
#include "tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_interface.h"

namespace
{

/// Micro-kernel interface
const tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
    tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
    tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
    tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv};

void fill_matrix_random_fp32(size_t rows, size_t cols, float *matrix, float min, float max)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        matrix[i] = min + (max - min) * r;
    }
}

// Quantize fp32 [M, K] -> int8 (per-row asymmetric, qai8dx).
//   q = clamp(round(x*scale) + nudged_zp, -128, 127)
//   recip_scale = 1/scale stored in scale_a
void quantize_a_qai8dx(size_t m, size_t k, const float *A_fp32, int8_t *A_q, float *recip_scale_a,
                       int32_t *zp_a)
{
    const float qmin = static_cast<float>(INT8_MIN);
    const float qmax = static_cast<float>(INT8_MAX);

    for (size_t i = 0; i < m; ++i) {
        float row_min = A_fp32[i * k];
        float row_max = row_min;
        for (size_t j = 1; j < k; ++j) {
            float v = A_fp32[i * k + j];
            row_min = std::fmin(row_min, v);
            row_max = std::fmax(row_max, v);
        }
        float rmin = std::fmin(0.0f, row_min);
        float rmax = std::fmax(0.0f, row_max);
        float scale = (rmax == rmin) ? 1.0f : (qmax - qmin) / (rmax - rmin);
        float recip = (scale == 0.0f) ? 0.0f : 1.0f / scale;

        float zp_min_err = qmin + rmin * scale;
        float zp_max_err = qmax + rmax * scale;
        float zp_f =
            (zp_min_err + zp_max_err > 0.0f) ? (qmin - rmin * scale) : (qmax - rmax * scale);
        zp_f = std::fmax(zp_f, qmin);
        zp_f = std::fmin(zp_f, qmax);
        int32_t nudged_zp = static_cast<int32_t>(std::nearbyint(zp_f));

        for (size_t j = 0; j < k; ++j) {
            float q = std::nearbyint(A_fp32[i * k + j] * scale) + static_cast<float>(nudged_zp);
            q = std::fmax(q, qmin);
            q = std::fmin(q, qmax);
            A_q[i * k + j] = static_cast<int8_t>(q);
        }
        recip_scale_a[i] = recip;
        zp_a[i] = nudged_zp;
    }
}

// Quantize fp32 [N, K] -> int2 (per-channel symmetric, qsi2cx). Stored as packed u2 bytes
// in [N, K/4] layout (s3 s2 s1 s0 per byte). Symmetric int2 range used: {-2, -1, 0, 1};
// scale_b = max_abs / 2.
void quantize_b_qsi2cx(size_t n, size_t k, const float *B_fp32, uint8_t *B_u2, float *scale_b,
                       int32_t *sum_b)
{
    const size_t k_bytes = k / 4;
    for (size_t i = 0; i < n; ++i) {
        float max_abs = 0.0f;
        for (size_t j = 0; j < k; ++j) {
            max_abs = std::fmax(max_abs, std::fabs(B_fp32[i * k + j]));
        }
        float scale = (max_abs == 0.0f) ? 1.0f : max_abs / 2.0f;
        scale_b[i] = scale;
        float recip = 1.0f / scale;

        int32_t row_sum = 0;
        for (size_t kb = 0; kb < k_bytes; ++kb) {
            uint8_t byte = 0;
            for (size_t s = 0; s < 4; ++s) {
                size_t kk = kb * 4 + s;
                float q_f = std::nearbyint(B_fp32[i * k + kk] * recip);
                q_f = std::fmax(q_f, -2.0f);
                q_f = std::fmin(q_f, 1.0f);
                int32_t q_i = static_cast<int32_t>(q_f);
                row_sum += q_i;
                uint8_t u2 = static_cast<uint8_t>(q_i + 2) & 0x3u;
                byte |= static_cast<uint8_t>(u2 << (s * 2));
            }
            B_u2[i * k_bytes + kb] = byte;
        }
        sum_b[i] = row_sum;
    }
}

// Reference implementation.
void run_gemm_ref(size_t m, size_t n, size_t k, const int8_t *A_q, const float *recip_scale_a,
                  const int32_t *zp_a, const uint8_t *B_u2, const float *scale_b, const int8_t *lut,
                  const float *bias, float *D, float clamp_min, float clamp_max)
{
    const size_t k_bytes = k / 4;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t iacc = 0;
            for (size_t kb = 0; kb < k_bytes; ++kb) {
                uint8_t byte = B_u2[j * k_bytes + kb];
                for (size_t s = 0; s < 4; ++s) {
                    size_t kk = kb * 4 + s;
                    uint8_t u2 = static_cast<uint8_t>((byte >> (s * 2)) & 0x3u);
                    int32_t qa = A_q[i * k + kk];
                    int32_t qb = lut[u2];
                    iacc += (qa - zp_a[i]) * qb;
                }
            }
            float acc = recip_scale_a[i] * scale_b[j] * static_cast<float>(iacc);
            if (bias)
                acc += bias[j];
            acc = std::fmax(acc, clamp_min);
            acc = std::fmin(acc, clamp_max);
            D[i * n + j] = acc;
        }
    }
}

float calculate_cosine_similarity(size_t size, const float *ref, const float *act)
{
    double dp = 0.0, rn = 0.0, an = 0.0;
    for (size_t i = 0; i < size; ++i) {
        dp += static_cast<double>(ref[i]) * static_cast<double>(act[i]);
        rn += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        an += static_cast<double>(act[i]) * static_cast<double>(act[i]);
    }
    rn = std::sqrt(rn);
    an = std::sqrt(an);
    if (rn > 0.0 && an > 0.0)
        return static_cast<float>(dp / (rn * an));
    return 0.0f;
}

}  // namespace

int main()
{
    const size_t M = 16;
    const size_t N = 64;
    const size_t K = 128;
    const size_t lda = K;
    const size_t ldb = K;
    const size_t ldd = N;

    float *A_fp32 = static_cast<float *>(malloc(M * K * sizeof(float)));
    float *B_fp32 = static_cast<float *>(malloc(N * K * sizeof(float)));
    int8_t *A_q = static_cast<int8_t *>(malloc(M * K));
    uint8_t *B_u2 = static_cast<uint8_t *>(malloc(N * (K / 4)));
    float *recip_scale_a = static_cast<float *>(malloc(M * sizeof(float)));
    int32_t *zp_a = static_cast<int32_t *>(malloc(M * sizeof(int32_t)));
    float *scale_b = static_cast<float *>(malloc(N * sizeof(float)));
    int32_t *sum_b = static_cast<int32_t *>(malloc(N * sizeof(int32_t)));
    float *bias = static_cast<float *>(malloc(N * sizeof(float)));
    float *D_ref = static_cast<float *>(malloc(M * N * sizeof(float)));
    float *D = static_cast<float *>(malloc(M * N * sizeof(float)));

    if (!A_fp32 || !B_fp32 || !A_q || !B_u2 || !recip_scale_a || !zp_a || !scale_b || !sum_b ||
        !bias || !D_ref || !D) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    srand(42);
    fill_matrix_random_fp32(M, K, A_fp32, -1.0f, 1.0f);
    fill_matrix_random_fp32(N, K, B_fp32, -1.0f, 1.0f);
    fill_matrix_random_fp32(1, N, bias, -0.5f, 0.5f);

    quantize_a_qai8dx(M, K, A_fp32, A_q, recip_scale_a, zp_a);
    quantize_b_qsi2cx(N, K, B_fp32, B_u2, scale_b, sum_b);

    const float clamp_min = -1e9f;
    const float clamp_max = 1e9f;

    run_gemm_ref(M, N, K, A_q, recip_scale_a, zp_a, B_u2, scale_b, tqt_qsi2_default_lut, bias,
                 D_ref, clamp_min, clamp_max);

    tqt_qai8dx_qsi2cx_f32_params params{};
    params.scale_a = recip_scale_a;
    params.zero_point_a = zp_a;
    params.scale_b = scale_b;
    params.sum_b = sum_b;

    memset(D, 0, M * N * sizeof(float));
    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_m = std::min(M - m_idx, m_step);
            const size_t actual_n = std::min(N - n_idx, n_step);

            const uint8_t *a_ptr = (const uint8_t *)A_q + ukernel.get_a_offset(m_idx, 0, lda);
            const uint8_t *b_ptr = (const uint8_t *)B_u2 + ukernel.get_b_offset(n_idx, 0, ldb);
            const uint8_t *bias_ptr = (const uint8_t *)bias + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D + ukernel.get_d_offset(m_idx, n_idx, ldd);

            tqt_qai8dx_qsi2cx_f32_params tile_params{};
            tile_params.scale_a = params.scale_a + m_idx;
            tile_params.zero_point_a = params.zero_point_a + m_idx;
            tile_params.scale_b = params.scale_b + n_idx;
            tile_params.sum_b = params.sum_b + n_idx;

            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, lda, 0, b_ptr, ldb, 0, d_ptr, ldd,
                             bias_ptr, clamp_min, clamp_max, /*lut=*/nullptr, &tile_params);
        }
    }

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
    const bool passed = cosine_sim > 0.9999f;

    printf("TEST[gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt]\n");
    printf("- ukernel: gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cosine_sim);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    free(A_fp32);
    free(B_fp32);
    free(A_q);
    free(B_u2);
    free(recip_scale_a);
    free(zp_a);
    free(scale_b);
    free(sum_b);
    free(bias);
    free(D_ref);
    free(D);

    return passed ? 0 : 1;
}
