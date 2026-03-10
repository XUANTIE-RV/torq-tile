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

template <typename T>
void reference_gemm(size_t m, size_t n, size_t k, const T *A, size_t lda, const T *B, size_t ldb,
                    BLayout b_layout, const T *C, size_t ldc, T *D, size_t ldd, const T *bias,
                    BiasMode bias_mode, T clamp_min, T clamp_max)
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

            D[i * ldd + j] = static_cast<T>(acc);
        }
    }
}

// Explicit instantiations
template void reference_gemm<float>(size_t, size_t, size_t, const float *, size_t, const float *,
                                    size_t, BLayout, const float *, size_t, float *, size_t,
                                    const float *, BiasMode, float, float);
template void reference_gemm<float16_t>(size_t, size_t, size_t, const float16_t *, size_t,
                                        const float16_t *, size_t, BLayout, const float16_t *,
                                        size_t, float16_t *, size_t, const float16_t *, BiasMode,
                                        float16_t, float16_t);
template void reference_gemm<bfloat16_t>(size_t, size_t, size_t, const bfloat16_t *, size_t,
                                         const bfloat16_t *, size_t, BLayout, const bfloat16_t *,
                                         size_t, bfloat16_t *, size_t, const bfloat16_t *, BiasMode,
                                         bfloat16_t, bfloat16_t);

}  // namespace test
}  // namespace torq_tile
