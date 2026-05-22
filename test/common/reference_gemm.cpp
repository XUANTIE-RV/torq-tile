//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/reference_gemm.h"

#include <algorithm>
#include <cmath>

namespace torq_tile
{
namespace test
{

template <typename TA, typename TD>
void reference_gemm(size_t m, size_t n, size_t k, const TA *A, size_t lda, const TA *B, size_t ldb,
                    BLayout b_layout, const TD *C, size_t ldc, TD *D, size_t ldd, const TD *bias,
                    BiasMode bias_mode, TD clamp_min, TD clamp_max)
{
    const float clamp_min_f = static_cast<float>(clamp_min);
    const float clamp_max_f = static_cast<float>(clamp_max);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // FP32 accumulation for precision
            float acc = 0.0f;

            for (size_t p = 0; p < k; ++p) {
                float a_val = static_cast<float>(A[i * lda + p]);
                float b_val;

                if (b_layout == BLayout::kTransposed) {
                    // B is [N, K], access B[j * ldb + p]
                    b_val = static_cast<float>(B[j * ldb + p]);
                } else {
                    // B is [K, N], access B[p * ldb + j]
                    b_val = static_cast<float>(B[p * ldb + j]);
                }

                acc += a_val * b_val;
            }

            // Add residual C if provided
            if (C != nullptr) {
                acc += static_cast<float>(C[i * ldc + j]);
            }

            // Add bias if provided
            if (bias != nullptr) {
                if (bias_mode == BiasMode::k1xN) {
                    acc += static_cast<float>(bias[j]);
                } else if (bias_mode == BiasMode::kMx1) {
                    acc += static_cast<float>(bias[i]);
                }
            }

            // Apply clamp
            acc = std::fmax(acc, clamp_min_f);
            acc = std::fmin(acc, clamp_max_f);

            D[i * ldd + j] = static_cast<TD>(acc);
        }
    }
}

// Explicit instantiations
template void reference_gemm<float, float>(size_t, size_t, size_t, const float *, size_t,
                                           const float *, size_t, BLayout, const float *, size_t,
                                           float *, size_t, const float *, BiasMode, float, float);
template void reference_gemm<float16_t, float16_t>(size_t, size_t, size_t, const float16_t *,
                                                   size_t, const float16_t *, size_t, BLayout,
                                                   const float16_t *, size_t, float16_t *, size_t,
                                                   const float16_t *, BiasMode, float16_t,
                                                   float16_t);
template void reference_gemm<bfloat16_t, bfloat16_t>(size_t, size_t, size_t, const bfloat16_t *,
                                                     size_t, const bfloat16_t *, size_t, BLayout,
                                                     const bfloat16_t *, size_t, bfloat16_t *,
                                                     size_t, const bfloat16_t *, BiasMode,
                                                     bfloat16_t, bfloat16_t);
template void reference_gemm<bfloat16_t, float>(size_t, size_t, size_t, const bfloat16_t *, size_t,
                                                const bfloat16_t *, size_t, BLayout, const float *,
                                                size_t, float *, size_t, const float *, BiasMode,
                                                float, float);
template void reference_gemm<bfloat16_t, float16_t>(size_t, size_t, size_t, const bfloat16_t *,
                                                    size_t, const bfloat16_t *, size_t, BLayout,
                                                    const float16_t *, size_t, float16_t *, size_t,
                                                    const float16_t *, BiasMode, float16_t,
                                                    float16_t);

void reference_gemm_i32_i8(size_t m, size_t n, size_t k, const int8_t *A, size_t lda,
                           const int8_t *B, size_t ldb, BLayout b_layout, const int32_t *C,
                           size_t ldc, int32_t *D, size_t ldd, int32_t clamp_min, int32_t clamp_max)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = 0;

            for (size_t p = 0; p < k; ++p) {
                int32_t a_val = static_cast<int32_t>(A[i * lda + p]);
                int32_t b_val;

                if (b_layout == BLayout::kTransposed) {
                    // B is [N, K], access B[j * ldb + p]
                    b_val = static_cast<int32_t>(B[j * ldb + p]);
                } else {
                    // B is [K, N], access B[p * ldb + j]
                    b_val = static_cast<int32_t>(B[p * ldb + j]);
                }

                acc += a_val * b_val;
            }

            // Add residual C if provided
            if (C != nullptr) {
                acc += C[i * ldc + j];
            }

            // Apply clamp
            if (acc < clamp_min)
                acc = clamp_min;
            if (acc > clamp_max)
                acc = clamp_max;

            D[i * ldd + j] = acc;
        }
    }
}

template <typename TA, typename TD>
void reference_gemm_quantized_weight_mixed(size_t m, size_t n, size_t k, size_t bl, const TA *A,
                                           size_t lda, const uint8_t *B, size_t ldb,
                                           const TA *scale_b, size_t ldsb, const TD *C, size_t ldc,
                                           TD *D, size_t ldd, const TD *bias, BiasMode bias_mode,
                                           TD clamp_min, TD clamp_max)
{
    const size_t b_row_bytes = ldb / 2;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float acc = 0.0f;

            if (C) {
                acc = static_cast<float>(C[i * ldc + j]);
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

                float b_dequant = static_cast<float>(b_raw) * scale_val;
                acc += a_val * b_dequant;
            }

            if (bias) {
                if (bias_mode == BiasMode::k1xN) {
                    acc += static_cast<float>(bias[j]);
                } else if (bias_mode == BiasMode::kMx1) {
                    acc += static_cast<float>(bias[i]);
                }
            }

            acc = std::max(acc, static_cast<float>(clamp_min));
            acc = std::min(acc, static_cast<float>(clamp_max));
            D[i * ldd + j] = static_cast<TD>(acc);
        }
    }
}

template void reference_gemm_quantized_weight_mixed<float16_t, float16_t>(
    size_t, size_t, size_t, size_t, const float16_t *, size_t, const uint8_t *, size_t,
    const float16_t *, size_t, const float16_t *, size_t, float16_t *, size_t, const float16_t *,
    BiasMode, float16_t, float16_t);
template void reference_gemm_quantized_weight_mixed<float16_t, float>(
    size_t, size_t, size_t, size_t, const float16_t *, size_t, const uint8_t *, size_t,
    const float16_t *, size_t, const float *, size_t, float *, size_t, const float *, BiasMode,
    float, float);

}  // namespace test
}  // namespace torq_tile
