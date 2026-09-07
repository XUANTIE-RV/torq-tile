//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
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

}  // namespace test
}  // namespace torq_tile
