//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zvfbfwma)
#error This file must be compiled for riscv, riscv_vector, zvfbfwma.
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
#include "tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_interface.h"

namespace
{

const tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_ukernel ukernel{
    tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_get_a_packed_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_get_b_packed_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_get_c_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_get_d_offset_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_get_d_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_run_a_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv,
    tqt_run_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv};

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

void run_reference(size_t m, size_t n, size_t k, const int8_t *A, const uint8_t *B,
                   const float *scale_a, const int32_t *zp_a, const float *scale_b,
                   const bfloat16_t *bias, float clamp_min, float clamp_max, bfloat16_t *D)
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
                r += (float)bias[ni];
            if (r < clamp_min)
                r = clamp_min;
            if (r > clamp_max)
                r = clamp_max;
            D[mi * n + ni] = (bfloat16_t)r;
        }
    }
}

float cosine_similarity(size_t size, const bfloat16_t *a, const bfloat16_t *b)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double av = (float)a[i], bv = (float)b[i];
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    if (na <= 0.0 || nb <= 0.0)
        return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

}  // namespace

int main()
{
    const size_t M = 32, N = 32, K = 64;
    const size_t lda = K, ldb = K, ldd = N;

    srand(42);

    std::vector<float> A_fp32(M * K), B_fp32(N * K);
    std::vector<int8_t> A(M * K);
    std::vector<uint8_t> B(N * (K / 2));
    std::vector<float> scale_a(M), scale_b(N);
    std::vector<int32_t> zp_a(M);
    std::vector<bfloat16_t> bias(N), D_ref(M * N), D(M * N, (bfloat16_t)0);

    fill_random(M * K, A_fp32.data(), -1.0f, 1.0f);
    fill_random(N * K, B_fp32.data(), -1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i)
        bias[i] = (bfloat16_t)(-0.5f + 1.0f * ((float)rand() / (float)RAND_MAX));

    quantize_a_per_row(A_fp32.data(), M, K, A.data(), scale_a.data(), zp_a.data());
    quantize_b_per_channel_int4(B_fp32.data(), N, K, B.data(), scale_b.data());

    // bf16 has fp32-equivalent exponent range; use a wide clamp.
    const float clamp_min_f = -1.0e30f, clamp_max_f = 1.0e30f;
    const bfloat16_t clamp_min = (bfloat16_t)clamp_min_f;
    const bfloat16_t clamp_max = (bfloat16_t)clamp_max_f;

    run_reference(M, N, K, A.data(), B.data(), scale_a.data(), zp_a.data(), scale_b.data(),
                  bias.data(), clamp_min_f, clamp_max_f, D_ref.data());

    struct tqt_qai8dx_qsi4cx_params params;
    params.scale_a = scale_a.data();
    params.zero_point_a = zp_a.data();
    params.scale_b = scale_b.data();

    std::vector<uint8_t> A_packed(ukernel.get_a_packed_size(M, K));
    std::vector<uint8_t> B_packed(ukernel.get_b_packed_size(N, K));
    ukernel.run_a_pack(M, K, lda, K, 0, A.data(), A_packed.data());
    ukernel.run_bt_pack(N, K, ldb, K, 0, B.data(), B_packed.data());

    const size_t m_step = ukernel.get_m_step();
    const size_t n_step = ukernel.get_n_step();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            const size_t actual_m = std::min(M - m_idx, m_step);
            const size_t actual_n = std::min(N - n_idx, n_step);

            const uint8_t *a_ptr = A_packed.data() + ukernel.get_a_packed_offset(m_idx, 0, K, M);
            const uint8_t *b_ptr = B_packed.data() + ukernel.get_b_packed_offset(n_idx, 0, K, N);
            const uint8_t *bias_ptr = (const uint8_t *)bias.data() + ukernel.get_bias_offset(n_idx);
            uint8_t *d_ptr = (uint8_t *)D.data() + ukernel.get_d_offset(m_idx, n_idx, N);

            ukernel.run_gemm(actual_m, actual_n, K, a_ptr, K, 0, b_ptr, K, 0, nullptr, N, d_ptr,
                             ldd, bias_ptr, clamp_min, clamp_max, &params);
        }
    }

    const float cs = cosine_similarity(M * N, D_ref.data(), D.data());
    // bf16 has 7-bit mantissa (vs 10-bit for fp16); W4A8 quantization noise
    // dominates either way.
    const bool passed = cs > 0.99f;

    printf("TEST[gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp]\n");
    printf("- ukernel: gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cs);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");

    return passed ? 0 : 1;
}
