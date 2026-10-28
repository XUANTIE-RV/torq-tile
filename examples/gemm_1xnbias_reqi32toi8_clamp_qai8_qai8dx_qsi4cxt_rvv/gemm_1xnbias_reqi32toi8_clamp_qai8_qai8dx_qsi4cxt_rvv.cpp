//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "src/common/tqt_common.h"
#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"
#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_interface.h"

namespace
{

const tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv,
    tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv,
    tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv,
    tqt_get_b_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv,
    tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv,
    tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv};

void fill_random(size_t count, float *dst, float lo, float hi)
{
    for (size_t i = 0; i < count; ++i) {
        float r = (float)rand() / (float)RAND_MAX;
        dst[i] = lo + (hi - lo) * r;
    }
}

void quantize_a_per_row(const float *A_fp32, size_t m, size_t k, int8_t *A_i8, float *scale_a,
                        int32_t *zp_a)
{
    for (size_t row = 0; row < m; ++row) {
        const float *row_ptr = A_fp32 + row * k;
        float fmin = FLT_MAX, fmax = -FLT_MAX;
        for (size_t i = 0; i < k; ++i) {
            if (row_ptr[i] < fmin)
                fmin = row_ptr[i];
            if (row_ptr[i] > fmax)
                fmax = row_ptr[i];
        }
        if (fmax < 0.0f)
            fmax = 0.0f;
        if (fmin > 0.0f)
            fmin = 0.0f;
        float s = (fmax - fmin) / 255.0f;
        if (s == 0.0f) {
            scale_a[row] = 1.0f;
            zp_a[row] = 0;
            for (size_t i = 0; i < k; ++i)
                A_i8[row * k + i] = 0;
            continue;
        }
        int32_t zp = (int32_t)nearbyintf(-128.0f - fmin / s);
        if (zp < -128)
            zp = -128;
        if (zp > 127)
            zp = 127;
        scale_a[row] = s;
        zp_a[row] = zp;
        for (size_t i = 0; i < k; ++i) {
            float q = nearbyintf(row_ptr[i] / s) + (float)zp;
            if (q > 127.0f)
                q = 127.0f;
            if (q < -128.0f)
                q = -128.0f;
            A_i8[row * k + i] = (int8_t)q;
        }
    }
}

void quantize_b_per_channel_int4(const float *B_fp32, size_t n, size_t k, uint8_t *B_packed,
                                 float *scale_b)
{
    for (size_t row = 0; row < n; ++row) {
        const float *row_ptr = B_fp32 + row * k;
        float abs_max = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            float v = fabsf(row_ptr[i]);
            if (v > abs_max)
                abs_max = v;
        }
        float s = abs_max / 8.0f;
        if (s == 0.0f)
            s = 1.0f;
        scale_b[row] = s;
        for (size_t kp = 0; kp < k / 2; ++kp) {
            int32_t q0 = (int32_t)nearbyintf(row_ptr[2 * kp + 0] / s);
            int32_t q1 = (int32_t)nearbyintf(row_ptr[2 * kp + 1] / s);
            if (q0 < -8)
                q0 = -8;
            if (q0 > 7)
                q0 = 7;
            if (q1 < -8)
                q1 = -8;
            if (q1 > 7)
                q1 = 7;
            B_packed[row * (k / 2) + kp] = (uint8_t)((uint8_t)(q0 + 8) | ((uint8_t)(q1 + 8) << 4));
        }
    }
}

// Real-domain reference: D_real = scale_a[m] * scale_b[n] * acc' (+ bias[n]).
void run_reference_real(size_t m, size_t n, size_t k, const int8_t *A, const uint8_t *B,
                        const float *scale_a, const int32_t *zp_a, const float *scale_b,
                        const float *bias, float *D)
{
    for (size_t mi = 0; mi < m; ++mi) {
        for (size_t ni = 0; ni < n; ++ni) {
            int32_t acc = 0, b_sum = 0;
            for (size_t kp = 0; kp < k / 2; ++kp) {
                uint8_t byte = B[ni * (k / 2) + kp];
                int32_t b_lo = (int32_t)(byte & 0xf) - 8;
                int32_t b_hi = (int32_t)(byte >> 4) - 8;
                acc +=
                    (int32_t)A[mi * k + 2 * kp + 0] * b_lo + (int32_t)A[mi * k + 2 * kp + 1] * b_hi;
                b_sum += b_lo + b_hi;
            }
            acc -= zp_a[mi] * b_sum;
            float r = (float)acc * scale_a[mi] * scale_b[ni];
            if (bias)
                r += bias[ni];
            D[mi * n + ni] = r;
        }
    }
}

// Asymmetric int8 quantization parameters: scale = (max - min) / 255.
void compute_asym_quant_i8(const float *data, size_t size, float &scale, int32_t &zp)
{
    float fmin = FLT_MAX, fmax = -FLT_MAX;
    for (size_t i = 0; i < size; ++i) {
        if (data[i] < fmin)
            fmin = data[i];
        if (data[i] > fmax)
            fmax = data[i];
    }
    if (fmax < 0.0f)
        fmax = 0.0f;
    if (fmin > 0.0f)
        fmin = 0.0f;
    float s = (fmax - fmin) / 255.0f;
    if (s == 0.0f) {
        scale = 1.0f;
        zp = 0;
        return;
    }
    int32_t z = (int32_t)nearbyintf(-128.0f - fmin / s);
    if (z == -128)
        z = -127;
    scale = s;
    zp = z;
}

float cosine_similarity(size_t size, const float *a, const float *b)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < size; ++i) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0)
        return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

}  // namespace

int main()
{
    const size_t M = 32, N = 32, K = 64;
    const size_t ldd = N;

    srand(42);

    std::vector<float> A_fp32(M * K), B_fp32(N * K);
    std::vector<int8_t> A(M * K);
    std::vector<uint8_t> B(N * (K / 2));
    std::vector<float> scale_a(M), scale_b(N);
    std::vector<int32_t> zp_a(M);
    std::vector<float> bias(N), D_real(M * N);
    std::vector<int8_t> D(M * N, 0);

    fill_random(M * K, A_fp32.data(), -1.0f, 1.0f);
    fill_random(N * K, B_fp32.data(), -1.0f, 1.0f);
    fill_random(N, bias.data(), -0.5f, 0.5f);

    quantize_a_per_row(A_fp32.data(), M, K, A.data(), scale_a.data(), zp_a.data());
    quantize_b_per_channel_int4(B_fp32.data(), N, K, B.data(), scale_b.data());

    // Real-domain reference (no clamp), then derive output int8 quant params.
    run_reference_real(M, N, K, A.data(), B.data(), scale_a.data(), zp_a.data(), scale_b.data(),
                       bias.data(), D_real.data());

    float scale_d;
    int32_t zp_d;
    compute_asym_quant_i8(D_real.data(), M * N, scale_d, zp_d);

    struct tqt_qai8_qai8dx_qsi4cx_params params
    {
    };
    params.scale_a = scale_a.data();
    params.zero_point_a = zp_a.data();
    params.scale_b = scale_b.data();
    params.scale_d = scale_d;
    params.zero_point_d = zp_d;

    const int8_t clamp_min = -128, clamp_max = 127;

    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_m = std::min(M - m_idx, m_step);
            const size_t actual_n = std::min(N - n_idx, n_step);

            const uint8_t *a_ptr = (const uint8_t *)A.data() + ukernel.get_a_offset(m_idx, 0, K);
            const uint8_t *b_ptr = (const uint8_t *)B.data() + ukernel.get_b_offset(n_idx, 0, K);
            const uint8_t *bias_ptr = (const uint8_t *)bias.data() + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D.data() + ukernel.get_d_offset(m_idx, n_idx, N);

            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, K, 0, nullptr, N, d_ptr,
                             ldd, bias_ptr, clamp_min, clamp_max, &params);
        }
    }

    // Dequantize the int8 kernel output back to fp32 for comparison.
    std::vector<float> D_act(M * N);
    for (size_t i = 0; i < M * N; ++i) {
        D_act[i] = ((float)D[i] - (float)zp_d) * scale_d;
    }

#ifdef TQT_DEBUG
    for (size_t dbg_i = 0; dbg_i < M * N; ++dbg_i) {
        size_t dbg_row = dbg_i / N;
        size_t dbg_col = dbg_i % N;
        float dbg_diff = fabsf(D_real[dbg_i] - D_act[dbg_i]);
        printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f)%s\n", dbg_row, dbg_col, D_real[dbg_i],
               D_act[dbg_i], dbg_diff, dbg_diff > 2.0f * scale_d ? " [FAIL]" : "");
    }
#endif

    const float cs = cosine_similarity(M * N, D_real.data(), D_act.data());
    // The int8-output requantize path adds an extra output-quantization step on
    // top of the W4A8 loss, so cosine stays in the high 0.99 range.
    const bool passed = cs > 0.99f;

    printf("TEST[gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt]\n");
    printf("- ukernel: gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cs);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
