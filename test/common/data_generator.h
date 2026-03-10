//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace test
{

template <typename T>
class UniformRandomGenerator
{
   public:
    explicit UniformRandomGenerator(float low = -1.0f, float high = 1.0f, uint32_t seed = 42)
        : engine_(seed), dist_(low, high)
    {
    }

    void fill_matrix(T *data, size_t rows, size_t cols);

   private:
    std::mt19937 engine_;
    std::uniform_real_distribution<float> dist_;
};

// Explicit instantiation declarations
extern template class UniformRandomGenerator<float>;
extern template class UniformRandomGenerator<float16_t>;
extern template class UniformRandomGenerator<bfloat16_t>;

}  // namespace test
}  // namespace torq_tile
