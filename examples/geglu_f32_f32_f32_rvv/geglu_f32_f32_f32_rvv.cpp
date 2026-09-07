//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/common/tqt_common.h"
#include "tqt_geglu_f32_f32_f32_1x4vl_rvv.h"
#include "tqt_geglu_f32_f32_f32_interface.h"

namespace
{

const tqt_geglu_f32_f32_f32_ukernel ukernel{tqt_get_m_step_geglu_f32_f32_f32_1x4vl_rvv,
                                            tqt_get_n_step_geglu_f32_f32_f32_1x4vl_rvv,
                                            tqt_run_geglu_f32_f32_f32_1x4vl_rvv};

void fill_random(std::vector<float> &values, float min_value, float max_value)
{
    for (float &value : values) {
        const float unit = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        value = static_cast<float>(min_value + (max_value - min_value) * unit);
    }
}

void run_reference(size_t m, size_t n, const float *a, size_t lda, const float *b, size_t ldb,
                   float *c, size_t ldc)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float x = static_cast<float>(a[i * lda + j]);
            const float y = static_cast<float>(b[i * ldb + j]);
            constexpr double sqrt_2_over_pi = 0.7978845608028654;
            constexpr double cubic_coefficient = 0.044715;
            const double xd = static_cast<double>(x);
            const double u = sqrt_2_over_pi * (xd + cubic_coefficient * xd * xd * xd);
            c[i * ldc + j] = static_cast<float>((0.5 * xd * (1.0 + std::tanh(u))) * y);
        }
    }
}

bool verify_results(size_t m, size_t n, size_t ldc, const float *expected, const float *actual,
                    float &max_diff, double &cosine)
{
    double dot = 0.0;
    double expected_norm = 0.0;
    double actual_norm = 0.0;
    max_diff = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float ref = static_cast<float>(expected[i * ldc + j]);
            const float got = static_cast<float>(actual[i * ldc + j]);
            max_diff = std::fmax(max_diff, std::fabs(ref - got));
            dot += static_cast<double>(ref) * got;
            expected_norm += static_cast<double>(ref) * ref;
            actual_norm += static_cast<double>(got) * got;
        }
    }

    const double norm_product = expected_norm * actual_norm;
    cosine = norm_product > 0.0 ? dot / std::sqrt(norm_product) : 0.0;
    return max_diff <= 5e-4f && cosine >= 0.999999f;
}

}  // namespace

int main()
{
    constexpr size_t m = 16;
    constexpr size_t n = 100;
    constexpr size_t lda = n + 4;
    constexpr size_t ldb = n + 2;
    constexpr size_t ldc = n + 3;

    std::vector<float> a(m * lda);
    std::vector<float> b(m * ldb);
    std::vector<float> expected(m * ldc);
    std::vector<float> actual(m * ldc);
    fill_random(a, -8.0f, 8.0f);
    fill_random(b, -4.0f, 4.0f);

    run_reference(m, n, a.data(), lda, b.data(), ldb, expected.data(), ldc);
    ukernel.run_geglu(m, n, a.data(), lda, b.data(), ldb, actual.data(), ldc);

    float max_diff;
    double cosine;
    const bool passed = verify_results(m, n, ldc, expected.data(), actual.data(), max_diff, cosine);

    std::printf("TEST[geglu_f32_f32_f32]\n");
    std::printf("- ukernel: geglu_f32_f32_f32_1x4vl_rvv\n");
    std::printf("- operation: GeGLU\n");
    std::printf("- m_step: %zu, n_step: %zu\n", ukernel.get_m_step(), ukernel.get_n_step());
    std::printf("- shape: M=%zu N=%zu lda=%zu ldb=%zu ldc=%zu\n", m, n, lda, ldb, ldc);
    std::printf("- max absolute error: %.8f\n", max_diff);
    std::printf("- cosine similarity: %.8f\n", cosine);
    std::printf("- Status: %s\n", passed ? "PASSED" : "FAILED");
    return passed ? 0 : 1;
}
