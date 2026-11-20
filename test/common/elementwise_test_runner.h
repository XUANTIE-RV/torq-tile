//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "test/common/data_generator.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

struct ElementwiseTestParams
{
    size_t n;
    std::string name;

    ElementwiseTestParams(size_t n_, std::string name_ = "") : n(n_), name(std::move(name_)) {}
};

inline auto common_elementwise_test_values()
{
    return testing::Values(ElementwiseTestParams(1, "N1"),        //
                           ElementwiseTestParams(7, "N7"),        //
                           ElementwiseTestParams(16, "N16"),      //
                           ElementwiseTestParams(31, "N31"),      //
                           ElementwiseTestParams(64, "N64"),      //
                           ElementwiseTestParams(128, "N128"),    //
                           ElementwiseTestParams(255, "N255"),    //
                           ElementwiseTestParams(1023, "N1023"),  //
                           ElementwiseTestParams(4096, "N4096"));
}

struct ElementwiseTestParamNameGenerator
{
    std::string operator()(const testing::TestParamInfo<ElementwiseTestParams> &info) const
    {
        const auto &params = info.param;
        return params.name.empty() ? "N" + std::to_string(params.n) : params.name;
    }
};

struct ElementwiseRandomConfig
{
    float low;
    float high;
    uint32_t seed;
};

struct ElementwiseTolerance
{
    float max_abs_error;
    float min_cosine_similarity;
};

struct ElementwiseVVTestConfig
{
    ElementwiseRandomConfig a;
    std::optional<ElementwiseRandomConfig> independent_b;
    ElementwiseTolerance tolerance;
};

template <typename T>
struct ElementwiseVSTestConfig
{
    ElementwiseRandomConfig a;
    T scalar_b;
    ElementwiseTolerance tolerance;
};

template <typename T, typename ReferenceFn, typename KernelFn>
inline void run_elementwise_vv_test(size_t n, ReferenceFn reference, KernelFn kernel,
                                    const ElementwiseVVTestConfig &config)
{
    std::vector<T> a(n);
    std::vector<T> b(n);
    std::vector<T> c_ref(n);
    std::vector<T> c_actual(n);

    UniformRandomGenerator<T> rng_a(config.a.low, config.a.high, config.a.seed);
    rng_a.fill_matrix(a.data(), 1, n);
    if (config.independent_b.has_value()) {
        const auto &b_config = config.independent_b.value();
        UniformRandomGenerator<T> rng_b(b_config.low, b_config.high, b_config.seed);
        rng_b.fill_matrix(b.data(), 1, n);
    } else {
        rng_a.fill_matrix(b.data(), 1, n);
    }

    reference(n, a.data(), b.data(), c_ref.data());
    kernel(n, a.data(), b.data(), c_actual.data());

    const auto result =
        verify_gemm_result<T>(c_ref.data(), c_actual.data(), n, config.tolerance.max_abs_error,
                              config.tolerance.min_cosine_similarity);
    ASSERT_TRUE(result.passed) << result.error_message;
}

template <typename T, typename ReferenceFn, typename KernelFn>
inline void run_elementwise_vs_test(size_t n, ReferenceFn reference, KernelFn kernel,
                                    const ElementwiseVSTestConfig<T> &config)
{
    std::vector<T> a(n);
    std::vector<T> c_ref(n);
    std::vector<T> c_actual(n);

    UniformRandomGenerator<T> rng(config.a.low, config.a.high, config.a.seed);
    rng.fill_matrix(a.data(), 1, n);

    reference(n, a.data(), config.scalar_b, c_ref.data());
    kernel(n, a.data(), config.scalar_b, c_actual.data());

    const auto result =
        verify_gemm_result<T>(c_ref.data(), c_actual.data(), n, config.tolerance.max_abs_error,
                              config.tolerance.min_cosine_similarity);
    ASSERT_TRUE(result.passed) << result.error_message;
}

template <typename T, typename Bits>
inline T elementwise_test_value_from_bits(Bits bits)
{
    static_assert(sizeof(T) == sizeof(Bits), "Bit pattern must match the value size");
    T value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Exact finite comparisons, signed zeros and special-value classification must
// not be hidden by aggregate error or cosine-similarity metrics.
template <typename T>
inline void assert_elementwise_fp_exact(T actual, T expected)
{
    const float a = static_cast<float>(actual);
    const float e = static_cast<float>(expected);
    if (std::isnan(e)) {
        ASSERT_TRUE(std::isnan(a));
    } else if (std::isinf(e)) {
        ASSERT_TRUE(std::isinf(a));
        ASSERT_EQ(std::signbit(a), std::signbit(e));
    } else {
        ASSERT_TRUE(std::isfinite(a));
        ASSERT_EQ(a, e);
        if (e == 0.0f) {
            ASSERT_EQ(std::signbit(a), std::signbit(e));
        }
    }
}

template <typename T, typename KernelFn>
inline void run_elementwise_vs_exact_test(const std::vector<T> &input, T divisor,
                                          const std::vector<T> &expected, KernelFn kernel)
{
    ASSERT_EQ(input.size(), expected.size());
    const size_t n = input.size();
    const T sentinel = static_cast<T>(42.0f);
    for (bool in_place : {false, true}) {
        SCOPED_TRACE(testing::Message() << "n=" << n << " in_place=" << in_place);
        std::vector<T> a(n + 2, sentinel);
        std::copy(input.begin(), input.end(), a.begin() + 1);
        std::vector<T> c(n + 2, sentinel);
        T *dst = (in_place ? a.data() : c.data()) + 1;
        kernel(n, a.data() + 1, divisor, dst);
        ASSERT_NO_FATAL_FAILURE(assert_elementwise_fp_exact(dst[-1], sentinel));
        ASSERT_NO_FATAL_FAILURE(assert_elementwise_fp_exact(dst[n], sentinel));
        for (size_t i = 0; i < n; ++i) {
            SCOPED_TRACE(testing::Message() << "i=" << i);
            ASSERT_NO_FATAL_FAILURE(assert_elementwise_fp_exact(dst[i], expected[i]));
        }
    }
}

// The original reciprocal-overflow reproducer, with exact identities that do
// not depend on a floating-point reference implementation.
template <typename T, typename KernelFn>
inline void run_divvs_subnormal_identity_test(T divisor, KernelFn kernel)
{
    const T zero = static_cast<T>(0.0f);
    const T negative_zero = static_cast<T>(-0.0f);
    const bool negative = std::signbit(static_cast<float>(divisor));
    const std::vector<T> input{zero, divisor, divisor, negative_zero};
    const std::vector<T> expected{negative ? negative_zero : zero, static_cast<T>(1.0f),
                                  static_cast<T>(1.0f), negative ? zero : negative_zero};
    ASSERT_NO_FATAL_FAILURE(run_elementwise_vs_exact_test(input, divisor, expected, kernel));
}

template <typename T, typename KernelFn>
inline void run_divvs_boundary_test(const std::vector<T> &divisors, bool direct_division,
                                    size_t n_step, T min_subnormal, T max_finite, KernelFn kernel)
{
    ASSERT_GT(n_step, 0u);
    const T inf = static_cast<T>(std::numeric_limits<float>::infinity());
    const T nan = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
    const std::vector<size_t> sizes{0, 1, 4, n_step - 1, n_step, n_step + 1, 2 * n_step + 3};
    for (T divisor : divisors) {
        SCOPED_TRACE(testing::Message() << "divisor=" << static_cast<float>(divisor));
        const std::vector<T> pattern{static_cast<T>(0.0f),
                                     divisor,
                                     divisor,
                                     static_cast<T>(-0.0f),
                                     static_cast<T>(-divisor),
                                     min_subnormal,
                                     static_cast<T>(-min_subnormal),
                                     static_cast<T>(1.0f),
                                     static_cast<T>(-1.0f),
                                     max_finite,
                                     static_cast<T>(-max_finite),
                                     inf,
                                     static_cast<T>(-inf),
                                     nan};
        for (size_t n : sizes) {
            std::vector<T> input(n);
            std::vector<T> expected(n);
            for (size_t i = 0; i < n; ++i) {
                input[i] = pattern[i % pattern.size()];
                if (direct_division) {
                    // Independent scalar division, with precision wider than
                    // either tested datatype before the final conversion.
                    expected[i] = static_cast<T>(static_cast<double>(input[i]) /
                                                 static_cast<double>(divisor));
                } else {
                    // Lock down the legacy rounding sequence for divisors
                    // outside the fallback range, including zero/Inf/NaN.
                    volatile float reciprocal = 1.0f / static_cast<float>(divisor);
                    const T inv_b = static_cast<T>(reciprocal);
                    expected[i] =
                        static_cast<T>(static_cast<float>(input[i]) * static_cast<float>(inv_b));
                }
            }
            ASSERT_NO_FATAL_FAILURE(
                run_elementwise_vs_exact_test(input, divisor, expected, kernel));
        }
    }
}

}  // namespace test
}  // namespace torq_tile
