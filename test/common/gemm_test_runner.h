//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "src/common/tqt_common.h"
#include "test/common/data_generator.h"
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

template <typename ClampT>
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

template <typename ClampT>
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
// (TA: A/B element type, TD: C/D/bias element type; for same-type GEMM use TA == TD)
// ============================================================================

template <typename TA, typename TD, BLayout kBLayout, BiasMode kBiasMode, typename ClampT>
void run_nonpacked_gemm_test(const GemmTestParams &params, const NonPackedGemmFuncs<ClampT> &funcs)
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
    gen_a.fill_matrix(A.data(), m, k);

    std::vector<TA> B_data;
    size_t lda = k;
    size_t ldb;
    if (kBLayout == BLayout::kTransposed) {
        B_data.resize(n * k);
        gen_a.fill_matrix(B_data.data(), n, k);
        ldb = k;
    } else {
        B_data.resize(k * n);
        gen_a.fill_matrix(B_data.data(), k, n);
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
    reference_gemm<TA, TD>(m, n, k, A.data(), lda, B_data.data(), ldb, kBLayout, c_ptr_ref, ldc,
                           ref_D.data(), ldd, bias_ptr_ref, ref_bias_mode, clamp_min, clamp_max);

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

            const void *bias_ptr = nullptr;
            if (params.has_bias && is_last_k) {
                size_t bias_idx = (kBiasMode == BiasMode::kMx1) ? m_start : n_start;
                const size_t bias_offset = funcs.get_bias_offset(bias_idx);
                bias_ptr = reinterpret_cast<const uint8_t *>(bias_data.data()) + bias_offset;
            }

            ClampT effective_clamp_min = static_cast<ClampT>(to_target_type<TD>(-FLT_MAX));
            ClampT effective_clamp_max = static_cast<ClampT>(to_target_type<TD>(FLT_MAX));
            if (is_last_k) {
                effective_clamp_min = static_cast<ClampT>(clamp_min);
                effective_clamp_max = static_cast<ClampT>(clamp_max);
            }

            funcs.run_gemm(actual_m, actual_n, actual_k, a_ptr, lda, 0, b_ptr, ldb, 0, c_ptr, ldc,
                           d_ptr, ldd, bias_ptr, effective_clamp_min, effective_clamp_max);
        });

    auto result =
        verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, DefaultThreshold<TD>::abs_error,
                               DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  m_tile=" << m_tile << " n_tile=" << n_tile
                               << " k_tile=" << k_tile
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// Packed GEMM test runner
// (TA: A/B element type, TD: C/D/bias element type; for same-type GEMM use TA == TD)
// ============================================================================

template <typename TA, typename TD, BiasMode kBiasMode, typename ClampT>
void run_packed_gemm_test(const GemmTestParams &params, const PackedGemmFuncs<ClampT> &funcs)
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
        run_tiled_loop(
            m, n, k, m_tile, n_tile, k_tile, params.loop_order,
            [&](size_t m_start, size_t n_start, size_t k_start, size_t actual_m, size_t actual_n,
                size_t actual_k, bool is_first_k) {
                std::vector<TA> A_block(actual_m * actual_k);
                for (size_t i = 0; i < actual_m; ++i) {
                    for (size_t j = 0; j < actual_k; ++j) {
                        A_block[i * actual_k + j] = A[(m_start + i) * lda + k_start + j];
                    }
                }

                std::vector<TA> B_block(actual_n * actual_k);
                for (size_t i = 0; i < actual_n; ++i) {
                    for (size_t j = 0; j < actual_k; ++j) {
                        B_block[i * actual_k + j] = B_orig[(n_start + i) * ldb_orig + k_start + j];
                    }
                }

                const size_t a_packed_size = funcs.get_a_packed_size(actual_m, actual_k);
                const size_t b_packed_size = funcs.get_b_packed_size(actual_n, actual_k);
                std::vector<uint8_t> A_packed(a_packed_size);
                std::vector<uint8_t> B_packed(b_packed_size);

                funcs.run_a_pack(actual_m, actual_k, actual_k, actual_k, 0,
                                 static_cast<const void *>(A_block.data()),
                                 static_cast<void *>(A_packed.data()));
                funcs.run_bt_pack(actual_n, actual_k, actual_k, actual_k, 0,
                                  static_cast<const void *>(B_block.data()),
                                  static_cast<void *>(B_packed.data()));

                const size_t d_offset = funcs.get_d_offset(m_start, n_start, ldd);
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

                ClampT effective_clamp_min = static_cast<ClampT>(to_target_type<TD>(-FLT_MAX));
                ClampT effective_clamp_max = static_cast<ClampT>(to_target_type<TD>(FLT_MAX));
                if (is_last_k) {
                    effective_clamp_min = static_cast<ClampT>(clamp_min);
                    effective_clamp_max = static_cast<ClampT>(clamp_max);
                }

                funcs.run_gemm(actual_m, actual_n, actual_k,
                               static_cast<const void *>(A_packed.data()), actual_k, 0,
                               static_cast<const void *>(B_packed.data()), actual_k, 0, c_ptr, ldc,
                               d_ptr, ldd, bias_ptr, effective_clamp_min, effective_clamp_max);
            });
    }

    auto result =
        verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, DefaultThreshold<TD>::abs_error,
                               DefaultThreshold<TD>::cosine_sim);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  m_tile=" << m_tile << " n_tile=" << n_tile
                               << " k_tile=" << k_tile
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// B-only-packed GEMM test runner (A is in original layout, only B is packed)
// (TA: A/B element type, TD: C/D/bias element type; for same-type GEMM use TA == TD)
// ============================================================================

template <typename TA, typename TD, BiasMode kBiasMode, typename ClampT>
void run_b_only_packed_gemm_test(const GemmTestParams &params,
                                 const BOnlyPackedGemmFuncs<ClampT> &funcs)
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
        run_tiled_loop(
            m, n, k, m_tile, n_tile, k_tile, params.loop_order,
            [&](size_t m_start, size_t n_start, size_t k_start, size_t actual_m, size_t actual_n,
                size_t actual_k, bool is_first_k) {
                std::vector<TA> B_block(actual_n * actual_k);
                for (size_t i = 0; i < actual_n; ++i) {
                    for (size_t j = 0; j < actual_k; ++j) {
                        B_block[i * actual_k + j] = B_orig[(n_start + i) * ldb_orig + k_start + j];
                    }
                }

                const size_t b_packed_size = funcs.get_b_packed_size(actual_n, actual_k);
                std::vector<uint8_t> B_packed(b_packed_size);

                funcs.run_bt_pack(actual_n, actual_k, actual_k, actual_k, 0,
                                  static_cast<const void *>(B_block.data()),
                                  static_cast<void *>(B_packed.data()));

                const size_t a_offset = funcs.get_a_offset(m_start, k_start, lda);
                const size_t d_offset = funcs.get_d_offset(m_start, n_start, ldd);

                const void *a_ptr = reinterpret_cast<const uint8_t *>(A.data()) + a_offset;
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

                ClampT effective_clamp_min = static_cast<ClampT>(to_target_type<TD>(-FLT_MAX));
                ClampT effective_clamp_max = static_cast<ClampT>(to_target_type<TD>(FLT_MAX));
                if (is_last_k) {
                    effective_clamp_min = static_cast<ClampT>(clamp_min);
                    effective_clamp_max = static_cast<ClampT>(clamp_max);
                }

                funcs.run_gemm(actual_m, actual_n, actual_k, a_ptr, lda, 0,
                               static_cast<const void *>(B_packed.data()), actual_k, 0, c_ptr, ldc,
                               d_ptr, ldd, bias_ptr, effective_clamp_min, effective_clamp_max);
            });
    }

    auto result =
        verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, DefaultThreshold<TD>::abs_error,
                               DefaultThreshold<TD>::cosine_sim);
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

}  // namespace test
}  // namespace torq_tile
