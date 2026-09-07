//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/reference_sgemm.h"

namespace torq_tile
{
namespace test
{

template <typename TA, typename TD>
void reference_sgemm(size_t m, size_t n, size_t k, const TA *A, size_t lda, ALayout a_layout,
                     const TA *B, size_t ldb, BLayout b_layout, const TD *C, size_t ldc, TD *D,
                     size_t ldd, TD alpha, TD beta)
{
    const float alpha_f = static_cast<float>(alpha);
    const float beta_f = static_cast<float>(beta);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // FP32 accumulation for precision
            float acc = 0.0f;

            for (size_t p = 0; p < k; ++p) {
                float a_val;
                if (a_layout == ALayout::kTransposed) {
                    // A is [K, M], access A[p * lda + i]
                    a_val = static_cast<float>(A[p * lda + i]);
                } else {
                    // A is [M, K], access A[i * lda + p]
                    a_val = static_cast<float>(A[i * lda + p]);
                }

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

            // Apply alpha * A*B
            float result = alpha_f * acc;

            // Add beta * C if C is provided and beta != 0
            if (C != nullptr && beta_f != 0.0f) {
                result += beta_f * static_cast<float>(C[i * ldc + j]);
            }

            D[i * ldd + j] = static_cast<TD>(result);
        }
    }
}

// Explicit instantiations
template void reference_sgemm<float, float>(size_t, size_t, size_t, const float *, size_t, ALayout,
                                            const float *, size_t, BLayout, const float *, size_t,
                                            float *, size_t, float, float);

}  // namespace test
}  // namespace torq_tile
