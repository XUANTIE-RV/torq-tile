//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Global benchmark parameters (set by main.cpp CLI parsing)
// ============================================================================

struct GemmShape
{
    size_t m = 128;
    size_t n = 128;
    size_t k = 128;
};

inline GemmShape &global_gemm_shape()
{
    static GemmShape shape;
    return shape;
}

// ============================================================================
// GEMM FLOPS calculation: 2 * M * N * K (one multiply + one add per element)
// ============================================================================

inline double gemm_flops(size_t m, size_t n, size_t k)
{
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

// ============================================================================
// B layout enumeration
// ============================================================================

enum class BLayout
{
    kRowMajor,
    kTransposed,
};

// ============================================================================
// Bias mode enumeration
// ============================================================================

enum class BiasMode
{
    kNone,
    k1xN,
    kMx1,
};

// ============================================================================
// Simple random data filler (header-only, uses <random>)
// ============================================================================

template <typename T>
inline void fill_random(T *data, size_t count, float low = -1.0f, float high = 1.0f,
                        unsigned seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<T>(dist(gen));
    }
}

}  // namespace benchmark
}  // namespace torq_tile
