//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include "src/common/tqt_common.h"
#include "test/common/data_generator.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/reference_gemm.h"
#include "test/common/test_common.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// Function pointer types for non-packed variants
// ============================================================================

template <typename ClampT, typename BiasT = void>
struct NonPackedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb);
    using GetCOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldc);
    using GetBiasOffsetFunc = size_t (*)(size_t idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 ClampT clamp_min, ClampT clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBOffsetFunc get_b_offset;
    GetCOffsetFunc get_c_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Function pointer types for packed variants
// ============================================================================

template <typename ClampT, typename BiasT = void>
struct PackedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda,
                                            size_t actual_m);
    using GetBPackedOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb,
                                            size_t actual_n);
    using GetCOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldc);
    using GetBiasOffsetFunc = size_t (*)(size_t idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                  const void *B, void *B_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 ClampT clamp_min, ClampT clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedOffsetFunc get_a_packed_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetCOffsetFunc get_c_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBPackFunc run_b_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Function pointer types for B-only-packed variants
// (A is in original layout, only B is packed)
// ============================================================================

template <typename ClampT>
struct BOnlyPackedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetBPackedOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb,
                                            size_t actual_n);
    using GetCOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldc);
    using GetBiasOffsetFunc = size_t (*)(size_t idx);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunBPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                  const void *B, void *B_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 ClampT clamp_min, ClampT clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBPackedOffsetFunc get_b_packed_offset;
    GetCOffsetFunc get_c_offset;
    GetBiasOffsetFunc get_bias_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunBPackFunc run_b_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

/// Verify the family-specific K=0 contract with null A/B and in-place C == D.
template <typename TD, typename ClampT, typename GemmFuncs>
void run_gemm_zero_k_test(size_t m, size_t n, const GemmFuncs &funcs)
{
    UniformRandomGenerator<TD> gen(-2.0f, 2.0f, 61);
    std::vector<TD> D(m * n);
    std::vector<TD> bias(n);
    gen.fill_matrix(D.data(), m, n);
    gen.fill_matrix(bias.data(), 1, n);
    std::vector<TD> expected = D;
    const TD clamp_min = to_target_type<TD>(-1.0f);
    const TD clamp_max = to_target_type<TD>(1.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float value = static_cast<float>(expected[i * n + j]) + static_cast<float>(bias[j]);
            value = std::max(value, static_cast<float>(clamp_min));
            value = std::min(value, static_cast<float>(clamp_max));
            expected[i * n + j] = static_cast<TD>(value);
        }
    }

    funcs.run_gemm(m, n, 0, nullptr, 0, 0, nullptr, 0, 0, D.data(), n, D.data(), n, bias.data(),
                   static_cast<ClampT>(clamp_min), static_cast<ClampT>(clamp_max));

    auto result = verify_gemm_result<TD>(expected.data(), D.data(), m * n, 0.0f, 0.999999f);
    EXPECT_TRUE(result.passed) << result.error_message;
}

// ============================================================================
// Helper: resolve tile sizes (0 means use m_step/n_step/full-K)
// ============================================================================

inline void resolve_tile_sizes(const GemmTestParams &params, size_t m_step, size_t n_step,
                               size_t &m_tile, size_t &n_tile, size_t &k_tile)
{
    // m_tile=0 means use full M (no external M-tiling, kernel handles it)
    // n_tile=0 means use full N
    // k_tile=0 means use full K (no K-tiling)
    m_tile = (params.m_tile == 0) ? params.m : params.m_tile;
    n_tile = (params.n_tile == 0) ? params.n : params.n_tile;
    k_tile = (params.k_tile == 0) ? params.k : params.k_tile;
    if (gemm_clamp_probe_active) {
        // Historical fixed tile sizes may be illegal at a larger runtime VLEN.
        // Keep their split-K work, but give the dedicated clamp probe legal M/N panels.
        if (m_tile != params.m && m_tile % m_step != 0) {
            m_tile = std::min(params.m, ((m_tile + m_step - 1) / m_step) * m_step);
        }
        if (n_tile != params.n && n_tile % n_step != 0) {
            n_tile = std::min(params.n, ((n_tile + n_step - 1) / n_step) * n_step);
        }
    }
}

// ============================================================================
// Helper: validate tile constraints
// Returns empty string if valid, error message otherwise
// ============================================================================

inline std::string validate_params(const GemmTestParams &params, size_t m_step, size_t n_step,
                                   size_t m_tile, size_t n_tile, size_t k_tile)
{
    // Only enforce step-alignment when doing external tiling (tile < total dim).
    // When tile == total dim, the kernel handles M/N iteration internally and
    // supports arbitrary (non-aligned) dimensions.
    if (m_tile != params.m && m_tile % m_step != 0) {
        return "m_tile (" + std::to_string(m_tile) + ") must be a multiple of m_step (" +
               std::to_string(m_step) + ")";
    }
    if (n_tile != params.n && n_tile % n_step != 0) {
        return "n_tile (" + std::to_string(n_tile) + ") must be a multiple of n_step (" +
               std::to_string(n_step) + ")";
    }

    // Validate chunk constraints
    auto check_chunk = [&](const ChunkConfig &chunk, const char *name, size_t tile_d0,
                           size_t tile_d1) -> std::string {
        if (chunk.dim0 == 0 && chunk.dim1 == 0)
            return "";
        if (chunk.dim0 != 0) {
            if (chunk.dim0 % tile_d0 != 0) {
                return std::string(name) + ".dim0 (" + std::to_string(chunk.dim0) +
                       ") must be a multiple of tile (" + std::to_string(tile_d0) + ")";
            }
        }
        if (chunk.dim1 != 0) {
            if (chunk.dim1 % tile_d1 != 0) {
                return std::string(name) + ".dim1 (" + std::to_string(chunk.dim1) +
                       ") must be a multiple of tile (" + std::to_string(tile_d1) + ")";
            }
        }
        return "";
    };

    std::string err;
    err = check_chunk(params.a_chunk, "a_chunk", m_tile, k_tile);
    if (!err.empty())
        return err;
    err = check_chunk(params.b_chunk, "b_chunk", k_tile, n_tile);
    if (!err.empty())
        return err;
    err = check_chunk(params.c_chunk, "c_chunk", m_tile, n_tile);
    if (!err.empty())
        return err;
    err = check_chunk(params.d_chunk, "d_chunk", m_tile, n_tile);
    if (!err.empty())
        return err;

    return "";
}

// ============================================================================
// Helper: execute the 3-level tiled loop with configurable order
// The callback receives (m_start, n_start, k_start, actual_m, actual_n, actual_k, is_first_k)
// ============================================================================

using TileCallback =
    std::function<void(size_t m_start, size_t n_start, size_t k_start, size_t actual_m,
                       size_t actual_n, size_t actual_k, bool is_first_k)>;

inline void run_tiled_loop(size_t total_m, size_t total_n, size_t total_k, size_t m_tile,
                           size_t n_tile, size_t k_tile, const LoopDim loop_order[3],
                           const TileCallback &callback)
{
    // Build iteration ranges for each dimension
    struct DimRange
    {
        LoopDim dim;
        size_t total;
        size_t tile;
    };

    DimRange ranges[3];
    for (int i = 0; i < 3; ++i) {
        ranges[i].dim = loop_order[i];
        switch (loop_order[i]) {
            case LoopDim::M:
                ranges[i].total = total_m;
                ranges[i].tile = m_tile;
                break;
            case LoopDim::N:
                ranges[i].total = total_n;
                ranges[i].tile = n_tile;
                break;
            case LoopDim::K:
                ranges[i].total = total_k;
                ranges[i].tile = k_tile;
                break;
        }
    }

    // Track first K visit for each (m_tile_idx, n_tile_idx) pair
    size_t num_m_tiles = (total_m + m_tile - 1) / m_tile;
    size_t num_n_tiles = (total_n + n_tile - 1) / n_tile;
    std::vector<bool> first_k_visit(num_m_tiles * num_n_tiles, true);

    for (size_t i0 = 0; i0 < ranges[0].total; i0 += ranges[0].tile) {
        for (size_t i1 = 0; i1 < ranges[1].total; i1 += ranges[1].tile) {
            for (size_t i2 = 0; i2 < ranges[2].total; i2 += ranges[2].tile) {
                // Map loop indices back to M, N, K
                size_t indices[3] = {i0, i1, i2};
                size_t m_start = 0, n_start = 0, k_start = 0;
                for (int d = 0; d < 3; ++d) {
                    switch (ranges[d].dim) {
                        case LoopDim::M:
                            m_start = indices[d];
                            break;
                        case LoopDim::N:
                            n_start = indices[d];
                            break;
                        case LoopDim::K:
                            k_start = indices[d];
                            break;
                    }
                }

                size_t actual_m = std::min(m_tile, total_m - m_start);
                size_t actual_n = std::min(n_tile, total_n - n_start);
                size_t actual_k = std::min(k_tile, total_k - k_start);

                size_t m_tile_idx = m_start / m_tile;
                size_t n_tile_idx = n_start / n_tile;
                size_t mn_idx = m_tile_idx * num_n_tiles + n_tile_idx;

                bool is_first_k = first_k_visit[mn_idx];
                first_k_visit[mn_idx] = false;

                callback(m_start, n_start, k_start, actual_m, actual_n, actual_k, is_first_k);
            }
        }
    }
}

// ============================================================================
// Non-packed GEMM test runner
// (TA: A type, TB: B type, TD: C/D/bias type; TB defaults to TA)
// ============================================================================

template <typename TA, typename TD, BLayout kBLayout, BiasMode kBiasMode, typename ClampT,
          typename TB = TA, typename BiasT = void>
void run_nonpacked_gemm_test(const GemmTestParams &params,
                             const NonPackedGemmFuncs<ClampT, BiasT> &funcs,
                             float abs_error_threshold = DefaultThreshold<TD>::abs_error,
                             float cosine_sim_threshold = DefaultThreshold<TD>::cosine_sim,
                             bool narrow_a_to_b_type = false)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    size_t m_tile, n_tile, k_tile;
    resolve_tile_sizes(params, m_step, n_step, m_tile, n_tile, k_tile);

    std::string validation_err = validate_params(params, m_step, n_step, m_tile, n_tile, k_tile);
    if (!validation_err.empty()) {
        GTEST_SKIP() << "Skipping: " << validation_err;
        return;
    }

    UniformRandomGenerator<TA> gen_a(-1.0f, 1.0f, 42);
    UniformRandomGenerator<TB> gen_b(-1.0f, 1.0f, 44);
    std::vector<TA> A(m * k);
    gen_a.fill_matrix(A.data(), m, k);
    std::vector<TA> A_ref = A;
    if (narrow_a_to_b_type) {
        for (TA &value : A_ref) {
            value = static_cast<TA>(static_cast<TB>(value));
        }
    }

    std::vector<TB> B_data;
    size_t lda = k;
    size_t ldb;
    if (kBLayout == BLayout::kTransposed) {
        B_data.resize(n * k);
        gen_b.fill_matrix(B_data.data(), n, k);
        ldb = k;
    } else {
        B_data.resize(k * n);
        gen_b.fill_matrix(B_data.data(), k, n);
        ldb = n;
    }

    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<TD> gen_d(-1.0f, 1.0f, 43);

    std::vector<TD> C_data;
    const TD *c_ptr_ref = nullptr;
    if (params.has_c) {
        C_data.resize(m * n);
        gen_d.fill_matrix(C_data.data(), m, n);
        c_ptr_ref = C_data.data();
    }

    using ActualBiasT = std::conditional_t<std::is_void_v<BiasT>, TD, BiasT>;
    UniformRandomGenerator<ActualBiasT> gen_bias(-10.0f, 10.0f, 123);
    std::vector<ActualBiasT> bias_data;
    const ActualBiasT *bias_ptr_ref = nullptr;
    BiasMode ref_bias_mode = BiasMode::kNone;
    if (params.has_bias) {
        if (kBiasMode == BiasMode::kMx1) {
            bias_data.resize(m);
            gen_bias.fill_matrix(bias_data.data(), m, 1);
        } else {
            bias_data.resize(n);
            gen_bias.fill_matrix(bias_data.data(), 1, n);
        }
        bias_ptr_ref = bias_data.data();
        ref_bias_mode = kBiasMode;
    }

    const TD clamp_min = to_target_type<TD>(params.clamp_min_f);
    const TD clamp_max = to_target_type<TD>(params.clamp_max_f);

    std::vector<TD> ref_D(m * n);
    reference_gemm<TA, TD, TB, ActualBiasT>(m, n, k, A_ref.data(), lda, B_data.data(), ldb,
                                            kBLayout, c_ptr_ref, ldc, ref_D.data(), ldd,
                                            bias_ptr_ref, ref_bias_mode, clamp_min, clamp_max);

    std::vector<TD> act_D(m * n, to_target_type<TD>(0.0f));

    run_tiled_loop(
        m, n, k, m_tile, n_tile, k_tile, params.loop_order,
        [&](size_t m_start, size_t n_start, size_t k_start, size_t actual_m, size_t actual_n,
            size_t actual_k, bool is_first_k) {
            const size_t a_offset = funcs.get_a_offset(m_start, k_start, lda);
            const size_t b_offset = funcs.get_b_offset(n_start, k_start, ldb);
            const size_t d_offset = funcs.get_d_offset(m_start, n_start, ldd);

            const void *a_ptr = reinterpret_cast<const uint8_t *>(A.data()) + a_offset;
            const void *b_ptr = reinterpret_cast<const uint8_t *>(B_data.data()) + b_offset;
            void *d_ptr = reinterpret_cast<uint8_t *>(act_D.data()) + d_offset;

            const void *c_ptr = nullptr;
            if (!is_first_k) {
                c_ptr = static_cast<const void *>(reinterpret_cast<const uint8_t *>(act_D.data()) +
                                                  d_offset);
            } else if (params.has_c) {
                const size_t c_offset = funcs.get_c_offset(m_start, n_start, ldc);
                c_ptr = reinterpret_cast<const uint8_t *>(C_data.data()) + c_offset;
            }

            bool is_last_k = (k_start + actual_k >= k);

            const BiasT *bias_ptr = nullptr;
            if (params.has_bias && is_last_k) {
                size_t bias_idx = (kBiasMode == BiasMode::kMx1) ? m_start : n_start;
                const size_t bias_offset = funcs.get_bias_offset(bias_idx);
                bias_ptr = reinterpret_cast<const BiasT *>(
                    reinterpret_cast<const uint8_t *>(bias_data.data()) + bias_offset);
            }

            ClampT effective_clamp_min = static_cast<ClampT>(to_target_type<TD>(-INFINITY));
            ClampT effective_clamp_max = static_cast<ClampT>(to_target_type<TD>(INFINITY));
            if (is_last_k) {
                effective_clamp_min = static_cast<ClampT>(clamp_min);
                effective_clamp_max = static_cast<ClampT>(clamp_max);
            }

            funcs.run_gemm(actual_m, actual_n, actual_k, a_ptr, lda, k_start, b_ptr, ldb, k_start,
                           c_ptr, ldc, d_ptr, ldd, bias_ptr, effective_clamp_min,
                           effective_clamp_max);
        });

    auto result = verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, abs_error_threshold,
                                         cosine_sim_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  m_tile=" << m_tile << " n_tile=" << n_tile
                               << " k_tile=" << k_tile
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Packed GEMM test runner
// (TA: A type, TB: B type, TD: C/D/bias type; TB defaults to TA)
// ============================================================================

template <typename TA, typename TD, BiasMode kBiasMode, typename ClampT, typename TB = TA,
          typename BiasT = void>
void run_packed_gemm_test(const GemmTestParams &params, const PackedGemmFuncs<ClampT, BiasT> &funcs,
                          float abs_error_threshold = DefaultThreshold<TD>::abs_error,
                          float cosine_sim_threshold = DefaultThreshold<TD>::cosine_sim,
                          bool narrow_a_to_b_type = false)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    size_t m_tile, n_tile, k_tile;
    resolve_tile_sizes(params, m_step, n_step, m_tile, n_tile, k_tile);

    std::string validation_err = validate_params(params, m_step, n_step, m_tile, n_tile, k_tile);
    if (!validation_err.empty()) {
        GTEST_SKIP() << "Skipping: " << validation_err;
        return;
    }

    UniformRandomGenerator<TA> gen_a(-1.0f, 1.0f, 42);
    UniformRandomGenerator<TB> gen_b(-1.0f, 1.0f, 44);
    std::vector<TA> A(m * k);
    std::vector<TB> B_orig(n * k);
    gen_a.fill_matrix(A.data(), m, k);
    gen_b.fill_matrix(B_orig.data(), n, k);
    std::vector<TA> A_ref = A;
    if (narrow_a_to_b_type) {
        for (TA &value : A_ref) {
            value = static_cast<TA>(static_cast<TB>(value));
        }
    }

    const size_t lda = k;
    const size_t ldb_orig = k;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<TD> gen_d(-1.0f, 1.0f, 43);

    std::vector<TD> C_data;
    const TD *c_ptr_ref = nullptr;
    if (params.has_c) {
        C_data.resize(m * n);
        gen_d.fill_matrix(C_data.data(), m, n);
        c_ptr_ref = C_data.data();
    }

    using ActualBiasT = std::conditional_t<std::is_void_v<BiasT>, TD, BiasT>;
    UniformRandomGenerator<ActualBiasT> gen_bias(-10.0f, 10.0f, 123);
    std::vector<ActualBiasT> bias_data;
    const ActualBiasT *bias_ptr_ref = nullptr;
    BiasMode ref_bias_mode = BiasMode::kNone;
    if (params.has_bias) {
        if (kBiasMode == BiasMode::kMx1) {
            bias_data.resize(m);
            gen_bias.fill_matrix(bias_data.data(), m, 1);
        } else {
            bias_data.resize(n);
            gen_bias.fill_matrix(bias_data.data(), 1, n);
        }
        bias_ptr_ref = bias_data.data();
        ref_bias_mode = kBiasMode;
    }

    const TD clamp_min = to_target_type<TD>(params.clamp_min_f);
    const TD clamp_max = to_target_type<TD>(params.clamp_max_f);

    std::vector<TD> ref_D(m * n);
    reference_gemm<TA, TD, TB, ActualBiasT>(m, n, k, A_ref.data(), lda, B_orig.data(), ldb_orig,
                                            BLayout::kTransposed, c_ptr_ref, ldc, ref_D.data(), ldd,
                                            bias_ptr_ref, ref_bias_mode, clamp_min, clamp_max);

    std::vector<TD> act_D(m * n, to_target_type<TD>(0.0f));

    if (k_tile == k) {
        const size_t a_packed_size = funcs.get_a_packed_size(m, k);
        const size_t b_packed_size = funcs.get_b_packed_size(n, k);
        std::vector<uint8_t> A_packed(a_packed_size);
        std::vector<uint8_t> B_packed(b_packed_size);

        funcs.run_a_pack(m, k, lda, k, 0, static_cast<const void *>(A.data()),
                         static_cast<void *>(A_packed.data()));
        funcs.run_bt_pack(n, k, ldb_orig, k, 0, static_cast<const void *>(B_orig.data()),
                          static_cast<void *>(B_packed.data()));

        run_tiled_loop(
            m, n, k, m_tile, n_tile, k_tile, params.loop_order,
            [&](size_t m_start, size_t n_start, size_t k_start, size_t actual_m, size_t actual_n,
                size_t actual_k, bool is_first_k) {
                const size_t a_packed_offset = funcs.get_a_packed_offset(m_start, 0, lda, m);
                const size_t b_packed_offset = funcs.get_b_packed_offset(n_start, 0, ldb_orig, n);
                const size_t d_offset = funcs.get_d_offset(m_start, n_start, ldd);

                const void *a_ptr = A_packed.data() + a_packed_offset;
                const void *b_ptr = B_packed.data() + b_packed_offset;
                void *d_ptr = reinterpret_cast<uint8_t *>(act_D.data()) + d_offset;

                const void *c_ptr = nullptr;
                if (params.has_c) {
                    const size_t c_offset = funcs.get_c_offset(m_start, n_start, ldc);
                    c_ptr = reinterpret_cast<const uint8_t *>(C_data.data()) + c_offset;
                }

                const BiasT *bias_ptr = nullptr;
                if (params.has_bias) {
                    size_t bias_idx = (kBiasMode == BiasMode::kMx1) ? m_start : n_start;
                    const size_t bias_offset = funcs.get_bias_offset(bias_idx);
                    bias_ptr = reinterpret_cast<const BiasT *>(
                        reinterpret_cast<const uint8_t *>(bias_data.data()) + bias_offset);
                }

                funcs.run_gemm(actual_m, actual_n, actual_k, a_ptr, lda, 0, b_ptr, ldb_orig, 0,
                               c_ptr, ldc, d_ptr, ldd, bias_ptr, static_cast<ClampT>(clamp_min),
                               static_cast<ClampT>(clamp_max));
            });
    } else {
        const size_t a_packed_size = funcs.get_a_packed_size(m, k);
        const size_t b_packed_size = funcs.get_b_packed_size(n, k);
        std::vector<uint8_t> A_packed(a_packed_size);
        std::vector<uint8_t> B_packed(b_packed_size);

        for (size_t k_start = 0; k_start < k; k_start += k_tile) {
            const size_t actual_k = TQT_MIN(k - k_start, k_tile);
            const size_t a_packed_offset = funcs.get_a_packed_offset(0, k_start, lda, m);
            const size_t b_packed_offset = funcs.get_b_packed_offset(0, k_start, ldb_orig, n);
            funcs.run_a_pack(m, actual_k, lda, lda, k_start,
                             static_cast<const void *>(A.data() + k_start),
                             static_cast<void *>(A_packed.data() + a_packed_offset));
            funcs.run_bt_pack(n, actual_k, ldb_orig, ldb_orig, k_start,
                              static_cast<const void *>(B_orig.data() + k_start),
                              static_cast<void *>(B_packed.data() + b_packed_offset));
        }
        run_tiled_loop(
            m, n, k, m_tile, n_tile, k_tile, params.loop_order,
            [&](size_t m_start, size_t n_start, size_t k_start, size_t actual_m, size_t actual_n,
                size_t actual_k, bool is_first_k) {
                const size_t a_packed_offset =
                    funcs.get_a_packed_offset(m_start, k_start, lda, actual_m);
                const size_t b_packed_offset =
                    funcs.get_b_packed_offset(n_start, k_start, ldb_orig, actual_n);
                const size_t d_offset = funcs.get_d_offset(m_start, n_start, ldd);
                const void *a_ptr = A_packed.data() + a_packed_offset;
                const void *b_ptr = B_packed.data() + b_packed_offset;
                void *d_ptr = reinterpret_cast<uint8_t *>(act_D.data()) + d_offset;

                const void *c_ptr = nullptr;
                if (!is_first_k) {
                    c_ptr = static_cast<const void *>(
                        reinterpret_cast<const uint8_t *>(act_D.data()) + d_offset);
                } else if (params.has_c) {
                    const size_t c_offset = funcs.get_c_offset(m_start, n_start, ldc);
                    c_ptr = reinterpret_cast<const uint8_t *>(C_data.data()) + c_offset;
                }

                bool is_last_k = (k_start + actual_k >= k);

                const BiasT *bias_ptr = nullptr;
                if (params.has_bias && is_last_k) {
                    size_t bias_idx = (kBiasMode == BiasMode::kMx1) ? m_start : n_start;
                    const size_t bias_offset = funcs.get_bias_offset(bias_idx);
                    bias_ptr = reinterpret_cast<const BiasT *>(
                        reinterpret_cast<const uint8_t *>(bias_data.data()) + bias_offset);
                }

                ClampT effective_clamp_min = static_cast<ClampT>(to_target_type<TD>(-INFINITY));
                ClampT effective_clamp_max = static_cast<ClampT>(to_target_type<TD>(INFINITY));
                if (is_last_k) {
                    effective_clamp_min = static_cast<ClampT>(clamp_min);
                    effective_clamp_max = static_cast<ClampT>(clamp_max);
                }

                funcs.run_gemm(actual_m, actual_n, actual_k, a_ptr, lda, k_start, b_ptr, ldb_orig,
                               k_start, c_ptr, ldc, d_ptr, ldd, bias_ptr, effective_clamp_min,
                               effective_clamp_max);
            });
    }

    auto result = verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, abs_error_threshold,
                                         cosine_sim_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  m_tile=" << m_tile << " n_tile=" << n_tile
                               << " k_tile=" << k_tile
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Packed GEMM full-buffer split-K harness
// ============================================================================

template <typename TA, typename TD, BiasMode kBiasMode, typename ClampT, typename TB = TA,
          typename BiasT = void>
void run_packed_gemm_full_buffer_split_k_test(
    size_t m, size_t n, size_t k, size_t k_split, const PackedGemmFuncs<ClampT, BiasT> &funcs,
    float abs_error_threshold = DefaultThreshold<TD>::abs_error,
    float cosine_sim_threshold = DefaultThreshold<TD>::cosine_sim, bool narrow_a_to_b_type = false)
{
    ASSERT_GT(k_split, 0U);
    ASSERT_LT(k_split, k);

    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldc = n;
    const size_t ldd = n;
    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    UniformRandomGenerator<TA> gen_a(-1.0f, 1.0f, 51);
    UniformRandomGenerator<TB> gen_b(-1.0f, 1.0f, 52);
    UniformRandomGenerator<TD> gen_d(-1.0f, 1.0f, 53);
    std::vector<TA> A(m * k);
    std::vector<TB> B(n * k);
    std::vector<TD> C(m * n);
    using ActualBiasT = std::conditional_t<std::is_void_v<BiasT>, TD, BiasT>;
    UniformRandomGenerator<ActualBiasT> gen_bias(-1.0f, 1.0f, 54);
    std::vector<ActualBiasT> bias(n);
    gen_a.fill_matrix(A.data(), m, k);
    gen_b.fill_matrix(B.data(), n, k);
    gen_d.fill_matrix(C.data(), m, n);
    gen_bias.fill_matrix(bias.data(), 1, n);

    std::vector<TA> A_ref = A;
    if (narrow_a_to_b_type) {
        for (TA &value : A_ref) {
            value = static_cast<TA>(static_cast<TB>(value));
        }
    }

    const TD clamp_min = to_target_type<TD>(-10.0f);
    const TD clamp_max = to_target_type<TD>(10.0f);
    std::vector<TD> ref_D(m * n);
    reference_gemm<TA, TD, TB, ActualBiasT>(m, n, k, A_ref.data(), lda, B.data(), ldb,
                                            BLayout::kTransposed, C.data(), ldc, ref_D.data(), ldd,
                                            bias.data(), kBiasMode, clamp_min, clamp_max);

    std::vector<uint8_t> A_packed(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> B_packed(funcs.get_b_packed_size(n, k));
    funcs.run_a_pack(m, k_split, lda, lda, 0, A.data(), A_packed.data());
    funcs.run_bt_pack(n, k_split, ldb, ldb, 0, B.data(), B_packed.data());

    const size_t a_split_offset = funcs.get_a_packed_offset(0, k_split, lda, m);
    const size_t b_split_offset = funcs.get_b_packed_offset(0, k_split, ldb, n);
    funcs.run_a_pack(m, k - k_split, lda, lda, k_split, A.data() + k_split,
                     A_packed.data() + a_split_offset);
    funcs.run_bt_pack(n, k - k_split, ldb, ldb, k_split, B.data() + k_split,
                      B_packed.data() + b_split_offset);

    std::vector<TD> act_D(m * n, to_target_type<TD>(0.0f));
    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        const size_t actual_m = TQT_MIN(m - m_idx, m_step);
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_n = TQT_MIN(n - n_idx, n_step);
            const size_t d_offset = funcs.get_d_offset(m_idx, n_idx, ldd);
            TD *d_ptr =
                reinterpret_cast<TD *>(reinterpret_cast<uint8_t *>(act_D.data()) + d_offset);
            const TD *c_ptr =
                reinterpret_cast<const TD *>(reinterpret_cast<const uint8_t *>(C.data()) +
                                             funcs.get_c_offset(m_idx, n_idx, ldc));
            const ActualBiasT *bias_ptr = reinterpret_cast<const ActualBiasT *>(
                reinterpret_cast<const uint8_t *>(bias.data()) + funcs.get_bias_offset(n_idx));

            const size_t a0_offset = funcs.get_a_packed_offset(m_idx, 0, lda, actual_m);
            const size_t b0_offset = funcs.get_b_packed_offset(n_idx, 0, ldb, actual_n);
            funcs.run_gemm(actual_m, actual_n, k_split, A_packed.data() + a0_offset, lda, 0,
                           B_packed.data() + b0_offset, ldb, 0, c_ptr, ldc, d_ptr, ldd, nullptr,
                           static_cast<ClampT>(-INFINITY), static_cast<ClampT>(INFINITY));

            const size_t a1_offset = funcs.get_a_packed_offset(m_idx, k_split, lda, actual_m);
            const size_t b1_offset = funcs.get_b_packed_offset(n_idx, k_split, ldb, actual_n);
            funcs.run_gemm(actual_m, actual_n, k - k_split, A_packed.data() + a1_offset, lda,
                           k_split, B_packed.data() + b1_offset, ldb, k_split, d_ptr, ldd, d_ptr,
                           ldd, bias_ptr, static_cast<ClampT>(clamp_min),
                           static_cast<ClampT>(clamp_max));
        }
    }

    auto result = verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, abs_error_threshold,
                                         cosine_sim_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << " split=" << k_split
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// B-only-packed GEMM test runner (A is in original layout, only B is packed)
// (TA: A/B element type, TD: C/D/bias element type; for same-type GEMM use TA == TD)
// ============================================================================

template <typename TA, typename TD, BiasMode kBiasMode, typename ClampT, typename BiasT = void>
void run_b_only_packed_gemm_test(const GemmTestParams &params,
                                 const BOnlyPackedGemmFuncs<ClampT> &funcs,
                                 float abs_error_threshold = DefaultThreshold<TD>::abs_error,
                                 float cosine_sim_threshold = DefaultThreshold<TD>::cosine_sim)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    size_t m_tile, n_tile, k_tile;
    resolve_tile_sizes(params, m_step, n_step, m_tile, n_tile, k_tile);

    std::string validation_err = validate_params(params, m_step, n_step, m_tile, n_tile, k_tile);
    if (!validation_err.empty()) {
        GTEST_SKIP() << "Skipping: " << validation_err;
        return;
    }

    UniformRandomGenerator<TA> gen_a(-1.0f, 1.0f, 42);
    std::vector<TA> A(m * k);
    std::vector<TA> B_orig(n * k);
    gen_a.fill_matrix(A.data(), m, k);
    gen_a.fill_matrix(B_orig.data(), n, k);

    const size_t lda = k;
    const size_t ldb_orig = k;
    const size_t ldc = n;
    const size_t ldd = n;

    UniformRandomGenerator<TD> gen_d(-1.0f, 1.0f, 43);

    std::vector<TD> C_data;
    const TD *c_ptr_ref = nullptr;
    if (params.has_c) {
        C_data.resize(m * n);
        gen_d.fill_matrix(C_data.data(), m, n);
        c_ptr_ref = C_data.data();
    }

    std::vector<TD> bias_data;
    const TD *bias_ptr_ref = nullptr;
    BiasMode ref_bias_mode = BiasMode::kNone;
    if (params.has_bias) {
        if (kBiasMode == BiasMode::kMx1) {
            bias_data.resize(m);
            gen_d.fill_matrix(bias_data.data(), m, 1);
        } else {
            bias_data.resize(n);
            gen_d.fill_matrix(bias_data.data(), 1, n);
        }
        bias_ptr_ref = bias_data.data();
        ref_bias_mode = kBiasMode;
    }

    const TD clamp_min = to_target_type<TD>(params.clamp_min_f);
    const TD clamp_max = to_target_type<TD>(params.clamp_max_f);

    std::vector<TD> ref_D(m * n);
    reference_gemm<TA, TD>(m, n, k, A.data(), lda, B_orig.data(), ldb_orig, BLayout::kTransposed,
                           c_ptr_ref, ldc, ref_D.data(), ldd, bias_ptr_ref, ref_bias_mode,
                           clamp_min, clamp_max);

    std::vector<TD> act_D(m * n, to_target_type<TD>(0.0f));

    if (k_tile == k) {
        const size_t b_packed_size = funcs.get_b_packed_size(n, k);
        std::vector<uint8_t> B_packed(b_packed_size);

        funcs.run_bt_pack(n, k, ldb_orig, k, 0, static_cast<const void *>(B_orig.data()),
                          static_cast<void *>(B_packed.data()));

        run_tiled_loop(
            m, n, k, m_tile, n_tile, k_tile, params.loop_order,
            [&](size_t m_start, size_t n_start, size_t k_start, size_t actual_m, size_t actual_n,
                size_t actual_k, bool is_first_k) {
                const size_t a_offset = funcs.get_a_offset(m_start, 0, lda);
                const size_t b_packed_offset = funcs.get_b_packed_offset(n_start, 0, ldb_orig, n);
                const size_t d_offset = funcs.get_d_offset(m_start, n_start, ldd);

                const void *a_ptr = reinterpret_cast<const uint8_t *>(A.data()) + a_offset;
                const void *b_ptr = B_packed.data() + b_packed_offset;
                void *d_ptr = reinterpret_cast<uint8_t *>(act_D.data()) + d_offset;

                const void *c_ptr = nullptr;
                if (params.has_c) {
                    const size_t c_offset = funcs.get_c_offset(m_start, n_start, ldc);
                    c_ptr = reinterpret_cast<const uint8_t *>(C_data.data()) + c_offset;
                }

                const void *bias_ptr = nullptr;
                if (params.has_bias) {
                    size_t bias_idx = (kBiasMode == BiasMode::kMx1) ? m_start : n_start;
                    const size_t bias_offset = funcs.get_bias_offset(bias_idx);
                    bias_ptr = reinterpret_cast<const uint8_t *>(bias_data.data()) + bias_offset;
                }

                funcs.run_gemm(actual_m, actual_n, actual_k, a_ptr, lda, 0, b_ptr, ldb_orig, 0,
                               c_ptr, ldc, d_ptr, ldd, bias_ptr, static_cast<ClampT>(clamp_min),
                               static_cast<ClampT>(clamp_max));
            });
    } else {
        const size_t b_packed_size = funcs.get_b_packed_size(n, k);
        std::vector<uint8_t> B_packed(b_packed_size);

        for (size_t k_start = 0; k_start < k; k_start += k_tile) {
            const size_t actual_k = TQT_MIN(k - k_start, k_tile);
            const size_t b_packed_offset = funcs.get_b_packed_offset(0, k_start, ldb_orig, n);
            funcs.run_bt_pack(n, actual_k, ldb_orig, ldb_orig, k_start,
                              static_cast<const void *>(B_orig.data() + k_start),
                              static_cast<void *>(B_packed.data() + b_packed_offset));
        }

        run_tiled_loop(
            m, n, k, m_tile, n_tile, k_tile, params.loop_order,
            [&](size_t m_start, size_t n_start, size_t k_start, size_t actual_m, size_t actual_n,
                size_t actual_k, bool is_first_k) {
                const size_t a_offset = funcs.get_a_offset(m_start, k_start, lda);
                const size_t b_packed_offset =
                    funcs.get_b_packed_offset(n_start, k_start, ldb_orig, actual_n);
                const size_t d_offset = funcs.get_d_offset(m_start, n_start, ldd);

                const void *a_ptr = reinterpret_cast<const uint8_t *>(A.data()) + a_offset;
                const void *b_ptr = B_packed.data() + b_packed_offset;
                void *d_ptr = reinterpret_cast<uint8_t *>(act_D.data()) + d_offset;

                const void *c_ptr = nullptr;
                if (!is_first_k) {
                    c_ptr = static_cast<const void *>(
                        reinterpret_cast<const uint8_t *>(act_D.data()) + d_offset);
                } else if (params.has_c) {
                    const size_t c_offset = funcs.get_c_offset(m_start, n_start, ldc);
                    c_ptr = reinterpret_cast<const uint8_t *>(C_data.data()) + c_offset;
                }

                bool is_last_k = (k_start + actual_k >= k);

                const void *bias_ptr = nullptr;
                if (params.has_bias && is_last_k) {
                    size_t bias_idx = (kBiasMode == BiasMode::kMx1) ? m_start : n_start;
                    const size_t bias_offset = funcs.get_bias_offset(bias_idx);
                    bias_ptr = reinterpret_cast<const uint8_t *>(bias_data.data()) + bias_offset;
                }

                ClampT effective_clamp_min = static_cast<ClampT>(to_target_type<TD>(-INFINITY));
                ClampT effective_clamp_max = static_cast<ClampT>(to_target_type<TD>(INFINITY));
                if (is_last_k) {
                    effective_clamp_min = static_cast<ClampT>(clamp_min);
                    effective_clamp_max = static_cast<ClampT>(clamp_max);
                }

                funcs.run_gemm(actual_m, actual_n, actual_k, a_ptr, lda, k_start, b_ptr, ldb_orig,
                               k_start, c_ptr, ldc, d_ptr, ldd, bias_ptr, effective_clamp_min,
                               effective_clamp_max);
            });
    }

    auto result = verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, abs_error_threshold,
                                         cosine_sim_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  m_tile=" << m_tile << " n_tile=" << n_tile
                               << " k_tile=" << k_tile
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// GTest parameter name generator
// ============================================================================

inline std::string sanitize_gtest_name(std::string name)
{
    for (auto &c : name) {
        if (!std::isalnum(c) && c != '_') {
            c = '_';
        }
    }
    return name;
}

inline std::string format_shape_string(size_t m, size_t n, size_t k)
{
    return "M" + std::to_string(m) + "_N" + std::to_string(n) + "_K" + std::to_string(k);
}

struct GemmTestParamNameGenerator
{
    std::string operator()(const testing::TestParamInfo<GemmTestParams> &info) const
    {
        const auto &p = info.param;
        std::string test_name = format_shape_string(p.m, p.n, p.k);
        if (p.bl > 0) {
            test_name += "_BL" + std::to_string(p.bl);
        }
        if (!p.name.empty()) {
            test_name += "_" + p.name;
        }
        return sanitize_gtest_name(test_name);
    }
};

// Square-packed GEMM contract: A/C/D use row-major blocks; B uses column-major
// blocks; each block is a row-major get_m_step() square. N/K padding uses
// get_n_step(). C and D are packed with public pack/unpack entry points.
// The reference computation and buffer checks are independent of the backend.
template <typename T>
struct GuardedGemmBuffer
{
    static constexpr size_t guard = 8;
    size_t count;
    std::vector<T> storage;
    explicit GuardedGemmBuffer(size_t n) : count(n), storage(n + 2 * guard, static_cast<T>(-7)) {}
    T *data() { return storage.data() + guard; }
    void check() const
    {
        for (size_t i = 0; i < guard; ++i) {
            EXPECT_EQ(static_cast<float>(storage[i]), -7.0f);
            EXPECT_EQ(static_cast<float>(storage[guard + count + i]), -7.0f);
        }
    }
};

static inline void expect_gemm_value(float actual, float expected)
{
    if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual));
    } else {
        EXPECT_EQ(actual, expected);
        if (expected == 0) {
            EXPECT_EQ(std::signbit(actual), std::signbit(expected));
        }
    }
}

template <typename T, bool Wide, typename Row, typename Bt, typename Packed>
void run_square_packed_gemm_test(const Row &row, const Bt &transposed, const Packed &packed,
                                 int layout, size_t m, size_t n, size_t k, bool residual, bool bias,
                                 bool alias, float lower = -INFINITY, float upper = INFINITY,
                                 bool special = false)
{
    SCOPED_TRACE(::testing::Message() << "layout=" << layout << " M/N/K=" << m << '/' << n << '/'
                                      << k << " wide=" << Wide << " alias=" << alias);
    const size_t lda = k + 3, ldb = n + 5, ldbt = k + 7, ldc = n + 9;
    const size_t ldd = alias ? ldc : n + 11;
    GuardedGemmBuffer<T> a(m * lda), b(k * ldb), bt(n * ldbt), c(m * ldc), d(m * ldd), biasv(n);
    const float special_values[] = {NAN, INFINITY, -INFINITY, -0.0f, 0.0f, -2, 2};
    for (size_t i = 0; i < m; ++i) {
        for (size_t q = 0; q < k; ++q) {
            a.data()[i * lda + q] =
                static_cast<T>((static_cast<int>((i * 17 + q * 3) % 13) - 6) / 13.0f);
        }
        for (size_t j = 0; j < n; ++j) {
            c.data()[i * ldc + j] =
                static_cast<T>(special ? special_values[(i * n + j) % 7]
                                       : (static_cast<int>((i * 3 + j) % 11) - 5) / 16.0f);
        }
    }
    for (size_t q = 0; q < k; ++q) {
        for (size_t j = 0; j < n; ++j) {
            b.data()[q * ldb + j] = bt.data()[j * ldbt + q] =
                static_cast<T>((static_cast<int>((q * 7 + j * 5) % 17) - 8) / 16.0f);
        }
    }
    for (size_t j = 0; j < n; ++j) {
        biasv.data()[j] = static_cast<T>((static_cast<int>(j % 7) - 3) / 16.0f);
    }
    const auto saved_a = a.storage, saved_b = b.storage, saved_bt = bt.storage;
    const auto saved_c = c.storage, saved_bias = biasv.storage;
    std::vector<float> expected(m * n);
    const T lo = static_cast<T>(lower), hi = static_cast<T>(upper);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = residual ? static_cast<float>(c.data()[i * ldc + j]) : 0;
            for (size_t q = 0; q < k; ++q) {
                const float av = static_cast<float>(a.data()[i * lda + q]);
                const float bv = static_cast<float>(b.data()[q * ldb + j]);
                if constexpr (Wide || sizeof(T) == sizeof(float)) {
                    sum = std::fma(av, bv, sum);
                } else {
                    // Double represents the product of two 16-bit inputs exactly.
                    sum = static_cast<float>(
                        static_cast<T>(std::fma(static_cast<double>(av), static_cast<double>(bv),
                                                static_cast<double>(sum))));
                }
            }
            if (bias) {
                sum += static_cast<float>(biasv.data()[j]);
            }
            sum = static_cast<float>(static_cast<T>(sum));
            if (!(lower == -INFINITY && upper == INFINITY)) {
                sum = fminf(fmaxf(sum, static_cast<float>(lo)), static_cast<float>(hi));
            }
            expected[i * n + j] = sum;
        }
    }
    T *output = alias ? c.data() : d.data();
    if (layout == 2) {
        GuardedGemmBuffer<T> ap(packed.get_a_packed_size(m, k) / sizeof(T));
        GuardedGemmBuffer<T> bp(packed.get_b_packed_size(n, k) / sizeof(T));
        GuardedGemmBuffer<T> btp(bp.count), cp(packed.get_d_packed_size(m, n) / sizeof(T)),
            dp(cp.count);
        GuardedGemmBuffer<T> biasp(packed.get_bias_packed_size(n) / sizeof(T));
        packed.run_a_pack(m, k, lda, a.data(), ap.data());
        packed.run_b_pack(n, k, ldb, b.data(), bp.data());
        packed.run_bt_pack(n, k, ldbt, bt.data(), btp.data());
        EXPECT_EQ(memcmp(bp.data(), btp.data(), bp.count * sizeof(T)), 0);
        packed.run_c_pack(m, n, ldc, c.data(), cp.data());
        packed.run_bias_pack(n, biasv.data(), biasp.data());
        T *outp = alias ? cp.data() : dp.data();
        packed.run_gemm(m, n, k, k ? ap.data() : nullptr, k ? bp.data() : nullptr,
                        residual ? cp.data() : nullptr, outp, bias ? biasp.data() : nullptr, lo,
                        hi);
        packed.run_d_unpack(m, n, ldd, outp, output);
        for (const auto *v : {&ap, &bp, &btp, &cp, &dp, &biasp}) {
            v->check();
        }
    } else {
        const auto fn = layout == 1 ? transposed.run_gemm : row.run_gemm;
        fn(m, n, k, k ? a.data() : nullptr, lda, 0,
           k ? (layout == 1 ? bt.data() : b.data()) : nullptr, layout == 1 ? ldbt : ldb, 0,
           residual ? c.data() : nullptr, ldc, output, ldd, bias ? biasv.data() : nullptr, lo, hi);
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            expect_gemm_value(static_cast<float>(output[i * ldd + j]), expected[i * n + j]);
        }
        for (size_t j = n; j < ldd; ++j) {
            EXPECT_EQ(static_cast<float>(output[i * ldd + j]), -7.0f);
        }
    }
    EXPECT_EQ(memcmp(a.storage.data(), saved_a.data(), saved_a.size() * sizeof(T)), 0);
    EXPECT_EQ(memcmp(b.storage.data(), saved_b.data(), saved_b.size() * sizeof(T)), 0);
    EXPECT_EQ(memcmp(bt.storage.data(), saved_bt.data(), saved_bt.size() * sizeof(T)), 0);
    EXPECT_EQ(memcmp(biasv.storage.data(), saved_bias.data(), saved_bias.size() * sizeof(T)), 0);
    if (!alias) {
        EXPECT_EQ(memcmp(c.storage.data(), saved_c.data(), saved_c.size() * sizeof(T)), 0);
    }
    for (const auto *v : {&a, &b, &bt, &c, &d, &biasv}) {
        v->check();
    }
}

template <typename T, typename Packed>
void run_square_packed_gemm_layout_test(const Packed &p)
{
    const size_t s = p.get_m_step(), step = p.get_n_step();
    const size_t rows = step + 1, cols = step + 3, ld = cols + 5;
    const size_t pr = 2 * step, pc = ((cols + step - 1) / step) * step;
    GuardedGemmBuffer<T> input(rows * ld), bp(p.get_b_packed_size(cols, rows) / sizeof(T));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            input.data()[i * ld + j] = static_cast<T>(1 + (i * 7 + j * 3) % 31);
        }
    }
    p.run_b_pack(cols, rows, ld, input.data(), bp.data());
    size_t offset = 0;
    for (size_t j = 0; j < pc; j += s) {
        for (size_t i = 0; i < pr; i += s) {
            for (size_t r = 0; r < s; ++r) {
                for (size_t c = 0; c < s; ++c) {
                    const float expected =
                        i + r < rows && j + c < cols
                            ? static_cast<float>(input.data()[(i + r) * ld + j + c])
                            : 0;
                    EXPECT_EQ(static_cast<float>(bp.data()[offset++]), expected);
                }
            }
        }
    }
    EXPECT_EQ(offset, bp.count);
    GuardedGemmBuffer<T> ap(p.get_a_packed_size(rows, cols) / sizeof(T));
    GuardedGemmBuffer<T> cp(p.get_d_packed_size(rows, cols) / sizeof(T));
    GuardedGemmBuffer<T> unpacked(rows * ld), bias(cols),
        biasp(p.get_bias_packed_size(cols) / sizeof(T));
    p.run_a_pack(rows, cols, ld, input.data(), ap.data());
    p.run_c_pack(rows, cols, ld, input.data(), cp.data());
    EXPECT_EQ(memcmp(ap.data(), cp.data(), ap.count * sizeof(T)), 0);
    offset = 0;
    for (size_t i = 0; i < rows; i += s) {
        for (size_t j = 0; j < pc; j += s) {
            for (size_t r = 0; r < s; ++r) {
                for (size_t c = 0; c < s; ++c) {
                    const float expected =
                        i + r < rows && j + c < cols
                            ? static_cast<float>(input.data()[(i + r) * ld + j + c])
                            : 0;
                    EXPECT_EQ(static_cast<float>(ap.data()[offset++]), expected);
                }
            }
        }
    }
    EXPECT_EQ(offset, ap.count);
    p.run_d_unpack(rows, cols, ld, cp.data(), unpacked.data());
    EXPECT_EQ(memcmp(input.data(), unpacked.data(), rows * ld * sizeof(T)), 0);
    for (size_t j = 0; j < cols; ++j)
        bias.data()[j] = static_cast<T>(1 + j % 17);
    p.run_bias_pack(cols, bias.data(), biasp.data());
    for (size_t j = 0; j < biasp.count; ++j) {
        EXPECT_EQ(static_cast<float>(biasp.data()[j]),
                  j < cols ? static_cast<float>(bias.data()[j]) : 0.0f);
    }
    for (const auto *v : {&input, &bp, &ap, &cp, &unpacked, &bias, &biasp})
        v->check();
}

// A unit contribution is exactly half an ULP at base. Native accumulation loses
// each tie; widening retains the sum. Bias adds another tie before final RNE.
template <typename T, typename Row, typename Bt, typename Packed>
void run_gemm_accumulation_precision_test(const Row &native, const Row &wide, const Bt &native_bt,
                                          const Bt &wide_bt, const Packed &native_p,
                                          const Packed &wide_p, float base)
{
    const size_t m = native.get_m_step(), n = native.get_n_step(), k = n;
    std::vector<T> a(m * k, static_cast<T>(1)), b(k * n, static_cast<T>(1));
    std::vector<T> c(m * n, static_cast<T>(base)), bias(n, static_cast<T>(1));
    std::vector<T> dn(m * n), dw(m * n);
    EXPECT_EQ(native_p.get_a_packed_size(m, k), wide_p.get_a_packed_size(m, k));
    EXPECT_EQ(native_p.get_b_packed_size(n, k), wide_p.get_b_packed_size(n, k));
    EXPECT_EQ(native_p.get_d_packed_size(m, n), wide_p.get_d_packed_size(m, n));
    for (int layout : {0, 1, 2}) {
        if (layout == 2) {
            std::vector<T> ap(native_p.get_a_packed_size(m, k) / sizeof(T));
            std::vector<T> bp(native_p.get_b_packed_size(n, k) / sizeof(T));
            std::vector<T> cp(native_p.get_d_packed_size(m, n) / sizeof(T));
            std::vector<T> biasp(native_p.get_bias_packed_size(n) / sizeof(T));
            std::vector<T> dnp(cp.size()), dwp(cp.size());
            native_p.run_a_pack(m, k, k, a.data(), ap.data());
            native_p.run_b_pack(n, k, n, b.data(), bp.data());
            native_p.run_c_pack(m, n, n, c.data(), cp.data());
            native_p.run_bias_pack(n, bias.data(), biasp.data());
            native_p.run_gemm(m, n, k, ap.data(), bp.data(), cp.data(), dnp.data(), biasp.data(),
                              (T)-INFINITY, (T)INFINITY);
            wide_p.run_gemm(m, n, k, ap.data(), bp.data(), cp.data(), dwp.data(), biasp.data(),
                            (T)-INFINITY, (T)INFINITY);
            native_p.run_d_unpack(m, n, n, dnp.data(), dn.data());
            wide_p.run_d_unpack(m, n, n, dwp.data(), dw.data());
        } else {
            const auto fn = layout == 0 ? native.run_gemm : native_bt.run_gemm;
            const auto fw = layout == 0 ? wide.run_gemm : wide_bt.run_gemm;
            fn(m, n, k, a.data(), k, 0, b.data(), n, 0, c.data(), n, dn.data(), n, bias.data(),
               (T)-INFINITY, (T)INFINITY);
            fw(m, n, k, a.data(), k, 0, b.data(), n, 0, c.data(), n, dw.data(), n, bias.data(),
               (T)-INFINITY, (T)INFINITY);
        }
        for (size_t i = 0; i < m * n; ++i) {
            EXPECT_EQ(static_cast<float>(dn[i]), base);
            EXPECT_EQ(static_cast<float>(dw[i]), static_cast<float>(static_cast<T>(base + k + 1)));
            EXPECT_NE(dn[i], dw[i]);
        }
    }
}

// Exact powers-of-two inputs produce the supplied dot value. A large F32 bias
// cancels it to small fractions that would disappear if bias or dot narrowed early.
template <typename TD, typename Run>
void check_f32_bias_epilogue(size_t m, size_t n, float dot, bool supports_c, Run run)
{
    const size_t stride = n + 3;
    std::vector<float> bias(n);
    std::vector<TD> residual(m * stride, static_cast<TD>(0));
    for (size_t j = 0; j < n; ++j) {
        bias[j] = -dot + static_cast<float>(static_cast<int>(j % 7) - 3) * 0.375f;
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            residual[i * stride + j] = static_cast<TD>(0.5f * (static_cast<int>(i % 3) - 1));
        }
    }
    for (bool has_bias : {false, true}) {
        for (bool has_c : {false, true}) {
            if (has_c && !supports_c)
                continue;
            for (bool clamp : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "bias=" << has_bias << " C=" << has_c << " clamp=" << clamp);
                std::vector<TD> output(m * stride + 8, static_cast<TD>(123));
                run(has_c ? residual.data() : nullptr, stride, has_bias ? bias.data() : nullptr,
                    output.data(), static_cast<TD>(clamp ? -1.0f : -INFINITY),
                    static_cast<TD>(clamp ? 1.0f : INFINITY));
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        float expected = dot;
                        if (has_c)
                            expected += static_cast<float>(residual[i * stride + j]);
                        if (has_bias)
                            expected += bias[j];
                        if (clamp)
                            expected = std::min(1.0f, std::max(-1.0f, expected));
                        EXPECT_EQ(static_cast<float>(output[i * stride + j]),
                                  static_cast<float>(static_cast<TD>(expected)))
                            << "row=" << i << " col=" << j;
                    }
                    for (size_t j = n; j < stride; ++j) {
                        EXPECT_EQ(static_cast<float>(output[i * stride + j]), 123.0f);
                    }
                }
                for (size_t i = m * stride; i < output.size(); ++i) {
                    EXPECT_EQ(static_cast<float>(output[i]), 123.0f);
                }
            }
        }
    }
    if constexpr (std::is_same_v<TD, float>) {
        // A finite F32 clamp above the F16 range must not become infinity.
        if (dot > 70000.0f) {
            for (bool negative : {false, true}) {
                std::vector<float> wide_bias(n, -2.0f * dot);
                std::vector<float> output(m * stride + 8, 123.0f);
                run(nullptr, stride, negative ? wide_bias.data() : nullptr, output.data(),
                    -70000.0f, 70000.0f);
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        EXPECT_EQ(output[i * stride + j], negative ? -70000.0f : 70000.0f);
                    }
                    for (size_t j = n; j < stride; ++j) {
                        EXPECT_EQ(output[i * stride + j], 123.0f);
                    }
                }
                for (size_t i = m * stride; i < output.size(); ++i) {
                    EXPECT_EQ(output[i], 123.0f);
                }
            }
        }
    }
}

}  // namespace test
}  // namespace torq_tile
