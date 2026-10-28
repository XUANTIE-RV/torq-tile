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
#include "tqt_gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t_4x1vl_rvv.h"

namespace
{

// ---------------------------------------------------------------------------
// Random fill (fp32 in [lo, hi])
// ---------------------------------------------------------------------------
void fill_random_f32(size_t count, float *dst, float lo, float hi)
{
    for (size_t i = 0; i < count; ++i) {
        float r = (float)rand() / (float)RAND_MAX;
        dst[i] = lo + (hi - lo) * r;
    }
}

// ---------------------------------------------------------------------------
// A quantization: per-row asymmetric int8 (qai8dx).
// ---------------------------------------------------------------------------
void quantize_a_qai8dx(const float *A_fp32, size_t m, size_t k, int8_t *A_i8, float *scale_a,
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

// ---------------------------------------------------------------------------
// A quantization: per-row block-quantized symmetric int8 (qsi8d2).
// ---------------------------------------------------------------------------
void quantize_a_qsi8d2(const float *A_fp32, size_t m, size_t k, size_t bl, int8_t *A_i8,
                       float *scale_a, size_t ldsa)
{
    const size_t num_blocks = k / bl;
    for (size_t row = 0; row < m; ++row) {
        for (size_t blk = 0; blk < num_blocks; ++blk) {
            const float *blk_ptr = A_fp32 + row * k + blk * bl;
            float abs_max = 0.0f;
            for (size_t i = 0; i < bl; ++i) {
                float v = fabsf(blk_ptr[i]);
                if (v > abs_max)
                    abs_max = v;
            }
            float s = abs_max / 127.0f;
            if (s == 0.0f)
                s = 1.0f;
            scale_a[row * ldsa + blk] = s;
            for (size_t i = 0; i < bl; ++i) {
                float q = nearbyintf(blk_ptr[i] / s);
                if (q > 127.0f)
                    q = 127.0f;
                if (q < -127.0f)
                    q = -127.0f;
                A_i8[row * k + blk * bl + i] = (int8_t)q;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// B quantization: per-block symmetric int4 (qsi4c2). ZP=8 unsigned encoding.
// ---------------------------------------------------------------------------
void quantize_b_qsi4c2(const float *B_fp32, size_t n, size_t k, size_t bl, uint8_t *B_packed,
                       float *scale_b, size_t ldsb)
{
    memset(B_packed, 0, n * (k / 2));
    const size_t num_blocks = k / bl;
    for (size_t row = 0; row < n; ++row) {
        for (size_t blk = 0; blk < num_blocks; ++blk) {
            const float *blk_ptr = B_fp32 + row * k + blk * bl;
            float abs_max = 0.0f;
            for (size_t i = 0; i < bl; ++i) {
                float v = fabsf(blk_ptr[i]);
                if (v > abs_max)
                    abs_max = v;
            }
            float s = abs_max / 8.0f;
            if (s == 0.0f)
                s = 1.0f;
            scale_b[row * ldsb + blk] = s;
            for (size_t i = 0; i < bl; ++i) {
                float q = nearbyintf(blk_ptr[i] / s);
                if (q > 7.0f)
                    q = 7.0f;
                if (q < -8.0f)
                    q = -8.0f;
                int32_t qi = (int32_t)q;
                size_t kidx = blk * bl + i;
                size_t byte_idx = row * (k / 2) + kidx / 2;
                uint8_t nib = (uint8_t)(qi + 8);
                if ((kidx & 1) == 0)
                    B_packed[byte_idx] = (B_packed[byte_idx] & 0xf0) | (nib & 0x0f);
                else
                    B_packed[byte_idx] = (B_packed[byte_idx] & 0x0f) | ((nib & 0x0f) << 4);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// B quantization: per-block ASYMMETRIC int4 (qai4c2). ZP=0 encoding.
// Dequant: (q - zp_b) * scale_b.
// ---------------------------------------------------------------------------
void quantize_b_qai4c2(const float *B_fp32, size_t n, size_t k, size_t bl, uint8_t *B_packed,
                       float *scale_b, float *zp_b, size_t ldsb, size_t ldzpb)
{
    memset(B_packed, 0, n * (k / 2));
    const size_t num_blocks = k / bl;
    for (size_t row = 0; row < n; ++row) {
        for (size_t blk = 0; blk < num_blocks; ++blk) {
            const float *blk_ptr = B_fp32 + row * k + blk * bl;
            float fmin = FLT_MAX, fmax = -FLT_MAX;
            for (size_t i = 0; i < bl; ++i) {
                if (blk_ptr[i] < fmin)
                    fmin = blk_ptr[i];
                if (blk_ptr[i] > fmax)
                    fmax = blk_ptr[i];
            }
            float s = (fmax - fmin) / 15.0f;
            if (s == 0.0f) {
                scale_b[row * ldsb + blk] = 1.0f;
                zp_b[row * ldzpb + blk] = 0.0f;
                for (size_t i = 0; i < bl; ++i) {
                    size_t kidx = blk * bl + i;
                    size_t byte_idx = row * (k / 2) + kidx / 2;
                    if ((kidx & 1) == 0)
                        B_packed[byte_idx] &= 0xf0;
                    else
                        B_packed[byte_idx] &= 0x0f;
                }
                continue;
            }
            float zp_f = -fmin / s;
            if (zp_f < 0.0f)
                zp_f = 0.0f;
            if (zp_f > 15.0f)
                zp_f = 15.0f;
            scale_b[row * ldsb + blk] = s;
            zp_b[row * ldzpb + blk] = zp_f;
            for (size_t i = 0; i < bl; ++i) {
                float q = nearbyintf(blk_ptr[i] / s + zp_f);
                if (q > 15.0f)
                    q = 15.0f;
                if (q < 0.0f)
                    q = 0.0f;
                uint8_t qb = (uint8_t)q;
                size_t kidx = blk * bl + i;
                size_t byte_idx = row * (k / 2) + kidx / 2;
                if ((kidx & 1) == 0)
                    B_packed[byte_idx] = (B_packed[byte_idx] & 0xf0) | (qb & 0x0f);
                else
                    B_packed[byte_idx] = (B_packed[byte_idx] & 0x0f) | ((qb & 0x0f) << 4);
            }
        }
    }
}

float cosine_similarity_fp32(size_t size, const float *a, const float *b)
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

// Class 2: qsi8d2 x qsi4c2.
//   acc = sum_blk( scale_a[m, blk] * scale_b[n, blk]
//           * sum_k( qa[m, k] * qb_signed[n, k] ) )
void ref_qsi8d2_qsi4c2(size_t m, size_t n, size_t k, size_t bl, const int8_t *A_i8,
                       const uint8_t *B_packed, const float *scale_a, size_t ldsa,
                       const float *scale_b, size_t ldsb, const float *bias, float clamp_min,
                       float clamp_max, float *D)
{
    const size_t num_blocks = k / bl;
    for (size_t mi = 0; mi < m; ++mi) {
        for (size_t ni = 0; ni < n; ++ni) {
            float acc = 0.0f;
            for (size_t blk = 0; blk < num_blocks; ++blk) {
                int32_t blk_acc = 0;
                for (size_t kk = 0; kk < bl; ++kk) {
                    size_t kidx = blk * bl + kk;
                    uint8_t byte = B_packed[ni * (k / 2) + kidx / 2];
                    int32_t qb =
                        (kidx & 1) ? ((int32_t)(byte >> 4) - 8) : ((int32_t)(byte & 0xf) - 8);
                    int32_t qa = (int32_t)A_i8[mi * k + kidx];
                    blk_acc += qa * qb;
                }
                acc += scale_a[mi * ldsa + blk] * scale_b[ni * ldsb + blk] * (float)blk_acc;
            }
            if (bias)
                acc += bias[ni];
            if (acc < clamp_min)
                acc = clamp_min;
            if (acc > clamp_max)
                acc = clamp_max;
            D[mi * n + ni] = acc;
        }
    }
}
}  // namespace

int main()
{
    // Test shape (K must be a multiple of BL).
    const size_t M = 32;
    const size_t N = 32;
    const size_t K = 64;
    const size_t BL = 16;
    const size_t lda = K;
    const size_t ldb = K;
    const size_t ldsa = K / BL;
    const size_t ldsb = K / BL;
    const size_t ldzpb = K / BL;
    const size_t ldd = N;
    (void)lda;
    (void)ldb;
    (void)ldsa;
    (void)ldzpb;

    srand(42);

    std::vector<float> A_fp32(M * K);
    std::vector<float> B_fp32(N * K);
    fill_random_f32(M * K, A_fp32.data(), -1.0f, 1.0f);
    fill_random_f32(N * K, B_fp32.data(), -1.0f, 1.0f);

    std::vector<int8_t> A(M * K);
    std::vector<float> scale_a(M * ldsa);
    quantize_a_qsi8d2(A_fp32.data(), M, K, BL, A.data(), scale_a.data(), ldsa);

    std::vector<uint8_t> B(N * (K / 2));
    std::vector<float> scale_b(N * ldsb);
    quantize_b_qsi4c2(B_fp32.data(), N, K, BL, B.data(), scale_b.data(), ldsb);

    std::vector<float> bias_fp32(N);
    fill_random_f32(N, bias_fp32.data(), -0.5f, 0.5f);
    std::vector<bfloat16_t> bias_data(N);
    for (size_t i = 0; i < N; ++i)
        bias_data[i] = static_cast<bfloat16_t>(bias_fp32[i]);

    const float CLAMP_MIN = -1e30f;
    const float CLAMP_MAX = 1e30f;
    const bfloat16_t clamp_min_t = static_cast<bfloat16_t>(CLAMP_MIN);
    const bfloat16_t clamp_max_t = static_cast<bfloat16_t>(CLAMP_MAX);

    std::vector<float> D_ref_fp32(M * N);
    std::vector<bfloat16_t> D(M * N, static_cast<bfloat16_t>(0));

    ref_qsi8d2_qsi4c2(M, N, K, BL, A.data(), B.data(), scale_a.data(), ldsa, scale_b.data(), ldsb,
                      bias_fp32.data(), CLAMP_MIN, CLAMP_MAX, D_ref_fp32.data());

    struct tqt_qsi8d2_qsi4c2_params params
    {
    };
    params.scale_a = scale_a.data();
    params.ldsa = ldsa;
    params.scale_b = scale_b.data();
    params.ldsb = ldsb;

    // The kernel does its own M / N tiling internally based on get_m_step /
    // get_n_step, so we just invoke it once with the full shape.
    tqt_run_gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t_4x1vl_rvv(
        M, N, K, BL, A.data(), lda, 0, B.data(), ldb, 0, nullptr, N, D.data(), ldd,
        bias_data.data(), clamp_min_t, clamp_max_t, &params);

    std::vector<float> D_fp32(M * N);
    for (size_t i = 0; i < M * N; ++i)
        D_fp32[i] = static_cast<float>(D[i]);

#ifdef TQT_DEBUG
    for (size_t dbg_i = 0; dbg_i < M * N; ++dbg_i) {
        size_t dbg_row = dbg_i / N;
        size_t dbg_col = dbg_i % N;
        float dbg_ref = (float)D_ref_fp32[dbg_i];
        float dbg_act = (float)D_fp32[dbg_i];
        float dbg_diff = fabsf(dbg_ref - dbg_act);
        if (dbg_diff > 1.0f)
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f) [FAIL]\n", dbg_row, dbg_col,
                   dbg_ref, dbg_act, dbg_diff);
        else
            printf("[%zu][%zu]: ref=%.6f vs act=%.6f (diff=%.6f)\n", dbg_row, dbg_col, dbg_ref,
                   dbg_act, dbg_diff);
    }
#endif

    const float cs = cosine_similarity_fp32(M * N, D_ref_fp32.data(), D_fp32.data());
    const bool passed = cs > 0.99f;

    printf("TEST[gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t]\n");
    printf("- ukernel: gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t_4x1vl_rvv\n");
    printf("- Cosine Similarity: %f\n", cs);
    printf("- Status: %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
