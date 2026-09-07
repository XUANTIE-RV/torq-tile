//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif  // Architectural features check.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/common/tqt_common.h"
#include "tqt_gelu_f16_f16_1x2vl_rvv.h"
#include "tqt_gelu_f16_f16_interface.h"

namespace
{

const tqt_gelu_f16_f16_ukernel ukernel{tqt_get_m_step_gelu_f16_f16_1x2vl_rvv,
                                       tqt_get_n_step_gelu_f16_f16_1x2vl_rvv,
                                       tqt_run_gelu_f16_f16_1x2vl_rvv};

void fill_random(std::vector<float16_t> &values, float min_value, float max_value)
{
    for (float16_t &value : values) {
        const float unit = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        value = static_cast<float16_t>(min_value + (max_value - min_value) * unit);
    }
}

void run_gelu_ref(size_t m, size_t n, const float16_t *a, size_t lda, float16_t *c, size_t ldc)
{
    constexpr double sqrt_2_over_pi = 0.7978845608028654;
    constexpr double cubic_coefficient = 0.044715;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double x = static_cast<double>(static_cast<float>(a[i * lda + j]));
            const double u = sqrt_2_over_pi * (x + cubic_coefficient * x * x * x);
            c[i * ldc + j] = static_cast<float16_t>(0.5 * x * (1.0 + std::tanh(u)));
        }
    }
}

bool verify_results(size_t m, size_t n, size_t ldc, const float16_t *expected,
                    const float16_t *actual, float max_abs_error_threshold,
                    float min_cosine_similarity, float &max_diff, double &cosine)
{
    double dot = 0.0;
    double expected_norm = 0.0;
    double actual_norm = 0.0;
    max_diff = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float ref = static_cast<float>(expected[i * ldc + j]);
            const float got = static_cast<float>(actual[i * ldc + j]);
            const float diff = std::fabs(ref - got);
            max_diff = diff > max_diff ? diff : max_diff;
            dot += static_cast<double>(ref) * got;
            expected_norm += static_cast<double>(ref) * ref;
            actual_norm += static_cast<double>(got) * got;
        }
    }

    const double norm_product = expected_norm * actual_norm;
    cosine = norm_product > 0.0 ? dot / std::sqrt(norm_product) : 0.0;
    return max_diff <= max_abs_error_threshold && cosine >= min_cosine_similarity;
}

}  // namespace

int main()
{
    constexpr size_t m = 16;
    constexpr size_t n = 100;
    constexpr size_t lda = n + 4;
    constexpr size_t ldc = n + 3;
    constexpr float max_abs_error = 2e-2f;
    constexpr float min_cosine = 0.9999f;

    std::vector<float16_t> input(m * lda);
    std::vector<float16_t> expected(m * ldc, static_cast<float16_t>(0.0f));
    std::vector<float16_t> actual(m * ldc, static_cast<float16_t>(0.0f));
    fill_random(input, -8.0f, 8.0f);
    run_gelu_ref(m, n, input.data(), lda, expected.data(), ldc);
    ukernel.run_gelu(m, n, input.data(), lda, actual.data(), ldc);

    float max_diff;
    double cosine;
    const bool passed = verify_results(m, n, ldc, expected.data(), actual.data(), max_abs_error,
                                       min_cosine, max_diff, cosine);

    std::printf("TEST[gelu_f16_f16]\n");
    std::printf("- ukernel: gelu_f16_f16_1x2vl_rvv\n");
    std::printf("- m_step: %zu, n_step: %zu\n", ukernel.get_m_step(), ukernel.get_n_step());
    std::printf("- shape: M=%zu N=%zu lda=%zu ldc=%zu\n", m, n, lda, ldc);
    std::printf("- max absolute error: %.8f\n", max_diff);
    std::printf("- cosine similarity: %.8f\n", cosine);
    std::printf("- Status: %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
