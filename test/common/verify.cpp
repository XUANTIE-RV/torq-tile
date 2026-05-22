//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/verify.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace torq_tile
{
namespace test
{

template <typename T>
VerificationResult verify_gemm_result(const T *reference, const T *actual, size_t size,
                                      float max_abs_error_threshold, float min_cosine_similarity)
{
    VerificationResult result;
    result.passed = true;
    result.max_abs_error = 0.0f;
    result.avg_abs_error = 0.0f;
    result.cosine_similarity = 0.0f;
    result.num_errors = 0;

    float total_abs_error = 0.0f;

    // Track max error location
    size_t max_error_index = 0;
    float max_error_ref_f = 0.0f;
    float max_error_act_f = 0.0f;

    // Calculate absolute errors
    for (size_t i = 0; i < size; ++i) {
        float ref_f = static_cast<float>(reference[i]);
        float act_f = static_cast<float>(actual[i]);
        float abs_error = std::fabs(ref_f - act_f);

        total_abs_error += abs_error;

        if (abs_error > result.max_abs_error) {
            result.max_abs_error = abs_error;
            max_error_index = i;
            max_error_ref_f = ref_f;
            max_error_act_f = act_f;
        }

        if (abs_error > max_abs_error_threshold) {
            result.num_errors++;
        }
    }

    result.avg_abs_error = (size > 0) ? (total_abs_error / static_cast<float>(size)) : 0.0f;

    // Calculate cosine similarity
    double dot_product = 0.0;
    double ref_norm_sq = 0.0;
    double act_norm_sq = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double ref_d = static_cast<double>(static_cast<float>(reference[i]));
        double act_d = static_cast<double>(static_cast<float>(actual[i]));

        dot_product += ref_d * act_d;
        ref_norm_sq += ref_d * ref_d;
        act_norm_sq += act_d * act_d;
    }

    double ref_norm = std::sqrt(ref_norm_sq);
    double act_norm = std::sqrt(act_norm_sq);

    if (ref_norm > 0.0 && act_norm > 0.0) {
        result.cosine_similarity = static_cast<float>(dot_product / (ref_norm * act_norm));
    } else if (ref_norm == 0.0 && act_norm == 0.0) {
        result.cosine_similarity = 1.0f;
    } else {
        result.cosine_similarity = 0.0f;
    }

    // Check pass/fail
    if (result.num_errors > 0 || result.cosine_similarity < min_cosine_similarity) {
        result.passed = false;

        std::ostringstream error_msg;
        error_msg << std::fixed << std::setprecision(6);
        error_msg << "Verification failed:\n"
                  << "  Absolute error threshold: " << max_abs_error_threshold << "\n"
                  << "  Cosine similarity threshold: " << min_cosine_similarity << "\n"
                  << "  Max absolute error: " << result.max_abs_error << " at index "
                  << max_error_index << "\n"
                  << "    Reference value: " << max_error_ref_f << "\n"
                  << "    Actual value: " << max_error_act_f << "\n"
                  << "  Average absolute error: " << result.avg_abs_error << "\n"
                  << "  Cosine similarity: " << result.cosine_similarity << "\n"
                  << "  Number of errors: " << result.num_errors << "/" << size;
        result.error_message = error_msg.str();
    }

    return result;
}

// Explicit instantiations
template VerificationResult verify_gemm_result<float>(const float *, const float *, size_t, float,
                                                      float);
template VerificationResult verify_gemm_result<float16_t>(const float16_t *, const float16_t *,
                                                          size_t, float, float);
template VerificationResult verify_gemm_result<bfloat16_t>(const bfloat16_t *, const bfloat16_t *,
                                                           size_t, float, float);
template VerificationResult verify_gemm_result<int32_t>(const int32_t *, const int32_t *, size_t,
                                                        float, float);

}  // namespace test
}  // namespace torq_tile
