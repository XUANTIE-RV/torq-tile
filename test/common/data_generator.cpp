//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/data_generator.h"

namespace torq_tile
{
namespace test
{

template <typename T>
void UniformRandomGenerator<T>::fill_matrix(T *data, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] = static_cast<T>(dist_(engine_));
    }
}

// Explicit instantiations
template class UniformRandomGenerator<float>;
template class UniformRandomGenerator<float16_t>;
template class UniformRandomGenerator<bfloat16_t>;

}  // namespace test
}  // namespace torq_tile
