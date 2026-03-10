//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// Enumerations
// ============================================================================

enum class BiasMode
{
    kNone,  // No bias
    k1xN,   // bias shape [1, N], broadcast along rows, get_bias_offset(n_idx)
    kMx1,   // bias shape [M, 1], broadcast along columns, get_bias_offset(m_idx)
};

enum class BLayout
{
    kRowMajor,    // B stored as [K, N], access B[k_idx * ldb + n_idx]
    kTransposed,  // B stored as [N, K], access B[n_idx * ldb + k_idx]
};

// ============================================================================
// Default Thresholds
// ============================================================================

template <typename T>
struct DefaultThreshold;

template <>
struct DefaultThreshold<float>
{
    static constexpr float abs_error = 1e-5f;
    static constexpr float cosine_sim = 0.999999f;
};

template <>
struct DefaultThreshold<float16_t>
{
    static constexpr float abs_error = 1.0f;
    static constexpr float cosine_sim = 0.9999f;
};

template <>
struct DefaultThreshold<bfloat16_t>
{
    static constexpr float abs_error = 1.0f;
    static constexpr float cosine_sim = 0.9999f;
};

// ============================================================================
// Loop Dimension Enumeration (for configurable loop order)
// ============================================================================

enum class LoopDim
{
    M,
    N,
    K
};

// ============================================================================
// Chunk Configuration (for buffer tiling)
// ============================================================================

struct ChunkConfig
{
    size_t dim0 = 0;  // 0 means disabled
    size_t dim1 = 0;
};

// ============================================================================
// Unified GEMM Test Parameters (C++17 builder pattern)
// ============================================================================

struct GemmTestParams
{
    // Required: shape
    size_t m;
    size_t n;
    size_t k;

    // Optional: bias (default: provide bias matching the interface)
    bool has_bias = false;

    // Optional: C matrix accumulation (default: no C)
    bool has_c = false;

    // Optional: clamp (default: no clamp, i.e. -FLT_MAX / FLT_MAX)
    float clamp_min_f = -FLT_MAX;
    float clamp_max_f = FLT_MAX;

    // Optional: tiling (0 means use full dimension or m_step/n_step at runtime)
    size_t m_tile = 0;
    size_t n_tile = 0;
    size_t k_tile = 0;

    // Optional: loop order (default: M-N-K)
    LoopDim loop_order[3] = {LoopDim::M, LoopDim::N, LoopDim::K};

    // Optional: chunk buffers (default: disabled)
    ChunkConfig a_chunk;  // (chunk_m, chunk_k)
    ChunkConfig b_chunk;  // B non-transposed: (chunk_k, chunk_n); BT: (chunk_n, chunk_k)
    ChunkConfig c_chunk;  // (chunk_m, chunk_n)
    ChunkConfig d_chunk;  // (chunk_m, chunk_n)

    // Test case name (for GTest output)
    std::string name;

    // Constructor: only shape is required
    GemmTestParams(size_t m_, size_t n_, size_t k_, std::string name_ = "")
        : m(m_), n(n_), k(k_), name(std::move(name_))
    {
    }

    // Chainable setters for optional parameters
    GemmTestParams &set_bias(bool v)
    {
        has_bias = v;
        return *this;
    }
    GemmTestParams &set_c(bool v)
    {
        has_c = v;
        return *this;
    }
    GemmTestParams &set_clamp(float lo, float hi)
    {
        clamp_min_f = lo;
        clamp_max_f = hi;
        return *this;
    }
    GemmTestParams &set_tile(size_t mt, size_t nt, size_t kt)
    {
        m_tile = mt;
        n_tile = nt;
        k_tile = kt;
        return *this;
    }
    GemmTestParams &set_loop_order(LoopDim a, LoopDim b, LoopDim c)
    {
        loop_order[0] = a;
        loop_order[1] = b;
        loop_order[2] = c;
        return *this;
    }
    GemmTestParams &set_a_chunk(size_t d0, size_t d1)
    {
        a_chunk = {d0, d1};
        return *this;
    }
    GemmTestParams &set_b_chunk(size_t d0, size_t d1)
    {
        b_chunk = {d0, d1};
        return *this;
    }
    GemmTestParams &set_c_chunk(size_t d0, size_t d1)
    {
        c_chunk = {d0, d1};
        return *this;
    }
    GemmTestParams &set_d_chunk(size_t d0, size_t d1)
    {
        d_chunk = {d0, d1};
        return *this;
    }
    GemmTestParams &set_name(std::string n)
    {
        name = std::move(n);
        return *this;
    }
};

// ============================================================================
// Utility: Convert float to target type
// ============================================================================

template <typename T>
inline T to_target_type(float value)
{
    return static_cast<T>(value);
}

// ============================================================================
// Utility: Convert target type to float
// ============================================================================

template <typename T>
inline float to_float(T value)
{
    return static_cast<float>(value);
}

}  // namespace test
}  // namespace torq_tile
