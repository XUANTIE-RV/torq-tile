//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "test/common/data_generator.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

struct ReductionTestParams
{
    size_t m;
    size_t n;
    std::string name;

    ReductionTestParams(size_t m_, size_t n_, std::string name_ = "")
        : m(m_), n(n_), name(std::move(name_))
    {
    }
};

inline auto common_reduction_test_values()
{
    return testing::Values(ReductionTestParams(1, 1, "M1_N1"),          //
                           ReductionTestParams(2, 7, "M2_N7"),          //
                           ReductionTestParams(3, 16, "M3_N16"),        //
                           ReductionTestParams(5, 127, "M5_N127"),      //
                           ReductionTestParams(8, 256, "M8_N256"),      //
                           ReductionTestParams(17, 1023, "M17_N1023"),  //
                           ReductionTestParams(1, 4096, "M1_N4096"),    //
                           ReductionTestParams(64, 64, "M64_N64"),      //
                           ReductionTestParams(100, 100, "M100_N100"));
}

struct ReductionTestParamNameGenerator
{
    std::string operator()(const testing::TestParamInfo<ReductionTestParams> &info) const
    {
        const auto &params = info.param;
        return params.name.empty()
                   ? "M" + std::to_string(params.m) + "_N" + std::to_string(params.n)
                   : params.name;
    }
};

enum class ReductionAxis
{
    kRow,
    kColumn,
};

enum class ReductionInit
{
    kNone,
    kWithInit,
};

struct ReductionRandomConfig
{
    float low;
    float high;
    uint32_t seed;
};

struct ReductionTolerance
{
    float max_abs_error;
    float min_cosine_similarity;
};

template <typename T>
struct ReductionTestConfig
{
    ReductionRandomConfig input;
    T init_value;
};

inline size_t reduction_output_length(const ReductionTestParams &params, ReductionAxis axis)
{
    return axis == ReductionAxis::kRow ? params.m : params.n;
}

inline size_t reduction_reduced_length(const ReductionTestParams &params, ReductionAxis axis)
{
    return axis == ReductionAxis::kRow ? params.n : params.m;
}

template <typename T, typename ReferenceFn, typename KernelFn, typename ToleranceFn>
inline void run_reduction_test(const ReductionTestParams &params, ReductionAxis axis,
                               ReductionInit init_mode, ReferenceFn reference, KernelFn kernel,
                               const ReductionTestConfig<T> &config,
                               ToleranceFn tolerance_for_reduced_length)
{
    const size_t lda = params.n;
    const size_t output_length = reduction_output_length(params, axis);
    const size_t reduced_length = reduction_reduced_length(params, axis);

    std::vector<T> input(params.m * params.n);
    std::vector<T> output_ref(output_length);
    std::vector<T> output_actual(output_length);
    std::vector<T> init;

    UniformRandomGenerator<T> rng(config.input.low, config.input.high, config.input.seed);
    rng.fill_matrix(input.data(), params.m, params.n);

    const T *init_ptr = nullptr;
    if (init_mode == ReductionInit::kWithInit) {
        init.assign(output_length, config.init_value);
        init_ptr = init.data();
    }

    reference(params.m, params.n, input.data(), lda, output_ref.data(), init_ptr);
    kernel(params.m, params.n, input.data(), lda, output_actual.data(), init_ptr);

    const ReductionTolerance tolerance = tolerance_for_reduced_length(reduced_length);
    const auto result =
        verify_gemm_result<T>(output_ref.data(), output_actual.data(), output_length,
                              tolerance.max_abs_error, tolerance.min_cosine_similarity);
    ASSERT_TRUE(result.passed) << result.error_message;
}

}  // namespace test
}  // namespace torq_tile
