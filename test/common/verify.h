//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace test
{

struct VerificationResult
{
    bool passed;
    std::string error_message;
    float max_abs_error;
    float avg_abs_error;
    float cosine_similarity;
    size_t num_errors;
};

/// Verify GEMM results using absolute error threshold and cosine similarity.
///
/// @param reference         Reference output data
/// @param actual            Actual output data from micro-kernel
/// @param size              Number of elements
/// @param max_abs_error_threshold  Maximum acceptable absolute error per element
/// @param min_cosine_similarity    Minimum acceptable cosine similarity
/// @return VerificationResult with pass/fail status and error statistics
template <typename T>
VerificationResult verify_gemm_result(const T *reference, const T *actual, size_t size,
                                      float max_abs_error_threshold, float min_cosine_similarity);

// Explicit instantiation declarations
extern template VerificationResult verify_gemm_result<float>(const float *, const float *, size_t,
                                                             float, float);
extern template VerificationResult verify_gemm_result<float16_t>(const float16_t *,
                                                                 const float16_t *, size_t, float,
                                                                 float);
extern template VerificationResult verify_gemm_result<bfloat16_t>(const bfloat16_t *,
                                                                  const bfloat16_t *, size_t, float,
                                                                  float);
extern template VerificationResult verify_gemm_result<int32_t>(const int32_t *, const int32_t *,
                                                               size_t, float, float);

}  // namespace test
}  // namespace torq_tile
