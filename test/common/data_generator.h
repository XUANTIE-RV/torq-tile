//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>

#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace test
{

namespace detail
{

/// Selects the appropriate distribution and parameter type based on T:
/// - integral types  → uniform_int_distribution<T>, param type is T
/// - floating-point types → uniform_real_distribution<float>, param type is float
template <typename T, bool IsIntegral = std::is_integral<T>::value>
struct DistributionTraits;

template <typename T>
struct DistributionTraits<T, /*IsIntegral=*/true>
{
    using DistType = std::uniform_int_distribution<T>;
    using ParamType = T;
};

template <typename T>
struct DistributionTraits<T, /*IsIntegral=*/false>
{
    using DistType = std::uniform_real_distribution<float>;
    using ParamType = float;
};

}  // namespace detail

template <typename T>
class UniformRandomGenerator
{
    using Traits = detail::DistributionTraits<T>;
    using DistType = typename Traits::DistType;
    using ParamType = typename Traits::ParamType;

   public:
    explicit UniformRandomGenerator(ParamType low = ParamType(-1), ParamType high = ParamType(1),
                                    uint32_t seed = 42)
        : engine_(seed), dist_(low, high)
    {
    }

    void fill_matrix(T *data, size_t rows, size_t cols);

   private:
    std::mt19937 engine_;
    DistType dist_;
};

// Explicit instantiation declarations
extern template class UniformRandomGenerator<float>;
extern template class UniformRandomGenerator<float16_t>;
extern template class UniformRandomGenerator<bfloat16_t>;
extern template class UniformRandomGenerator<int8_t>;
extern template class UniformRandomGenerator<int32_t>;

}  // namespace test
}  // namespace torq_tile
