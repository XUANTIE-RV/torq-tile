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
#include "test/common/gemm_test_runner.h"
#include "test/common/reference_sgemm.h"
#include "test/common/test_common.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// SGEMM Test Parameters (BLAS-style: D = alpha * op(A) * op(B) + beta * C)
// ============================================================================

struct SgemmTestParams
{
    // Required: shape
    size_t m;
    size_t n;
    size_t k;

    // BLAS-style scalars
    float alpha = 1.0f;
    float beta = 0.0f;

    // Optional: C matrix accumulation (default: no C)
    bool has_c = false;

    // Optional: tiling (0 means use full dimension or m_step/n_step at runtime)
    size_t m_tile = 0;
    size_t n_tile = 0;
    size_t k_tile = 0;

    // Optional: loop order (default: M-N-K)
    LoopDim loop_order[3] = {LoopDim::M, LoopDim::N, LoopDim::K};

    // Test case name (for GTest output)
    std::string name;

    // Constructor: only shape is required
    SgemmTestParams(size_t m_, size_t n_, size_t k_, std::string name_ = "")
        : m(m_), n(n_), k(k_), name(std::move(name_))
    {
    }

    // Chainable setters for optional parameters
    SgemmTestParams &set_alpha(float v)
    {
        alpha = v;
        return *this;
    }
    SgemmTestParams &set_beta(float v)
    {
        beta = v;
        return *this;
    }
    SgemmTestParams &set_c(bool v)
    {
        has_c = v;
        return *this;
    }
    SgemmTestParams &set_tile(size_t mt, size_t nt, size_t kt)
    {
        m_tile = mt;
        n_tile = nt;
        k_tile = kt;
        return *this;
    }
    SgemmTestParams &set_loop_order(LoopDim a, LoopDim b, LoopDim c)
    {
        loop_order[0] = a;
        loop_order[1] = b;
        loop_order[2] = c;
        return *this;
    }
};

// ============================================================================
// Function pointer types for non-packed SGEMM variants
// ============================================================================

template <typename DataT>
struct NonPackedSgemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb);
    using GetCOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldc);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetDSizeFunc = size_t (*)(size_t m, size_t n);
    using RunSgemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                  size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                  const void *C, size_t ldc, void *D, size_t ldd, DataT alpha,
                                  DataT beta);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBOffsetFunc get_b_offset;
    GetCOffsetFunc get_c_offset;
    GetDOffsetFunc get_d_offset;
    GetDSizeFunc get_d_size;
    RunSgemmFunc run_sgemm;
};

// ============================================================================
// Helper: resolve tile sizes for SGEMM (0 means use full dimension)
// ============================================================================

inline void resolve_sgemm_tile_sizes(const SgemmTestParams &params, size_t m_step, size_t n_step,
                                     size_t &m_tile, size_t &n_tile, size_t &k_tile)
{
    m_tile = (params.m_tile == 0) ? params.m : params.m_tile;
    n_tile = (params.n_tile == 0) ? params.n : params.n_tile;
    k_tile = (params.k_tile == 0) ? params.k : params.k_tile;
}

// ============================================================================
// Non-packed SGEMM test runner
// D = alpha * op(A) * op(B) + beta * C
//
// K-tiling semantics:
//   first  K-tile: alpha_eff = alpha, beta_eff = beta,  C_ptr = user C
//   later  K-tiles: alpha_eff = alpha, beta_eff = 1.0,   C_ptr = act_D (accumulate)
// ============================================================================

template <typename TA, typename TD, ALayout kALayout, BLayout kBLayout>
void run_nonpacked_sgemm_test(const SgemmTestParams &params, const NonPackedSgemmFuncs<TD> &funcs,
                              float abs_error_threshold = 1e-4f,
                              float cosine_sim_threshold = DefaultThreshold<TD>::cosine_sim)
{
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    const size_t m_step = funcs.get_m_step();
    const size_t n_step = funcs.get_n_step();

    size_t m_tile, n_tile, k_tile;
    resolve_sgemm_tile_sizes(params, m_step, n_step, m_tile, n_tile, k_tile);

    // Allocate and fill A
    UniformRandomGenerator<TA> gen_a(-1.0f, 1.0f, 42);
    std::vector<TA> A_data;
    size_t lda;
    if (kALayout == ALayout::kTransposed) {
        // A stored as [K, M]
        A_data.resize(k * m);
        gen_a.fill_matrix(A_data.data(), k, m);
        lda = m;
    } else {
        // A stored as [M, K]
        A_data.resize(m * k);
        gen_a.fill_matrix(A_data.data(), m, k);
        lda = k;
    }

    // Allocate and fill B
    std::vector<TA> B_data;
    size_t ldb;
    if (kBLayout == BLayout::kTransposed) {
        // B stored as [N, K]
        B_data.resize(n * k);
        gen_a.fill_matrix(B_data.data(), n, k);
        ldb = k;
    } else {
        // B stored as [K, N]
        B_data.resize(k * n);
        gen_a.fill_matrix(B_data.data(), k, n);
        ldb = n;
    }

    const size_t ldc = n;
    const size_t ldd = n;

    // Allocate C if needed
    UniformRandomGenerator<TD> gen_d(-1.0f, 1.0f, 43);
    std::vector<TD> C_data;
    const TD *c_ptr_ref = nullptr;
    if (params.has_c) {
        C_data.resize(m * n);
        gen_d.fill_matrix(C_data.data(), m, n);
        c_ptr_ref = C_data.data();
    }

    const TD alpha_val = static_cast<TD>(params.alpha);
    const TD beta_val = static_cast<TD>(params.beta);

    // Compute reference result
    // Forward-declare reference_sgemm (defined in reference_sgemm.h/cpp)
    std::vector<TD> ref_D(m * n);
    reference_sgemm<TA, TD>(m, n, k, A_data.data(), lda, kALayout, B_data.data(), ldb, kBLayout,
                            c_ptr_ref, ldc, ref_D.data(), ldd, alpha_val, beta_val);

    // Actual result
    std::vector<TD> act_D(m * n, static_cast<TD>(0));
    std::vector<TD> prev_tile_buf(m_tile * n_tile);

    // Run tiled loop
    run_tiled_loop(
        m, n, k, m_tile, n_tile, k_tile, params.loop_order,
        [&](size_t m_start, size_t n_start, size_t k_start, size_t actual_m, size_t actual_n,
            size_t actual_k, bool is_first_k) {
            const size_t a_offset = funcs.get_a_offset(m_start, k_start, lda);
            const size_t b_offset = funcs.get_b_offset(n_start, k_start, ldb);
            const size_t d_offset = funcs.get_d_offset(m_start, n_start, ldd);

            const void *a_ptr = reinterpret_cast<const uint8_t *>(A_data.data()) + a_offset;
            const void *b_ptr = reinterpret_cast<const uint8_t *>(B_data.data()) + b_offset;
            void *d_ptr = reinterpret_cast<uint8_t *>(act_D.data()) + d_offset;

            TD effective_alpha;
            TD effective_beta;
            const void *c_ptr = nullptr;
            size_t effective_ldc = ldc;

            if (is_first_k) {
                effective_alpha = alpha_val;
                effective_beta = beta_val;
                if (params.has_c) {
                    const size_t c_offset = funcs.get_c_offset(m_start, n_start, ldc);
                    c_ptr = reinterpret_cast<const uint8_t *>(C_data.data()) + c_offset;
                }
            } else {
                effective_alpha = alpha_val;
                effective_beta = static_cast<TD>(1.0f);
                // Copy D_prev tile to temp buffer for K-tiling accumulation.
                // The kernel's C==D alias bug is fixed, but for K-tiling we
                // still need a separate buffer since d_ptr is both source and dest.
                for (size_t i = 0; i < actual_m; ++i) {
                    memcpy(prev_tile_buf.data() + i * actual_n,
                           act_D.data() + (m_start + i) * ldd + n_start, actual_n * sizeof(TD));
                }
                c_ptr = static_cast<const void *>(prev_tile_buf.data());
                effective_ldc = actual_n;
            }

            funcs.run_sgemm(actual_m, actual_n, actual_k, a_ptr, lda, 0, b_ptr, ldb, 0, c_ptr,
                            effective_ldc, d_ptr, ldd, effective_alpha, effective_beta);
        });

    auto result = verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, abs_error_threshold,
                                         cosine_sim_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: " << params.name
                               << "\n  alpha=" << params.alpha << " beta=" << params.beta
                               << "\n  m_tile=" << m_tile << " n_tile=" << n_tile
                               << " k_tile=" << k_tile
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// C==D alias test: passes the same buffer as both C and D (no K-tiling).
// Verifies that alpha*A*B + beta*C is correct when C and D overlap.
// ============================================================================

template <typename TA, typename TD, ALayout kALayout, BLayout kBLayout>
void run_nonpacked_sgemm_alias_test(const SgemmTestParams &params,
                                    const NonPackedSgemmFuncs<TD> &funcs,
                                    float abs_error_threshold = 1e-4f,
                                    float cosine_sim_threshold = DefaultThreshold<TD>::cosine_sim)
{
    ASSERT_TRUE(params.has_c) << "Alias test requires has_c=true";
    ASSERT_NE(params.beta, 0.0f) << "Alias test requires beta != 0";

    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    UniformRandomGenerator<TA> gen_a(-1.0f, 1.0f, 42);
    std::vector<TA> A_data;
    size_t lda;
    if (kALayout == ALayout::kTransposed) {
        A_data.resize(k * m);
        gen_a.fill_matrix(A_data.data(), k, m);
        lda = m;
    } else {
        A_data.resize(m * k);
        gen_a.fill_matrix(A_data.data(), m, k);
        lda = k;
    }

    std::vector<TA> B_data;
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

    const size_t ld = n;
    const TD alpha_val = static_cast<TD>(params.alpha);
    const TD beta_val = static_cast<TD>(params.beta);

    UniformRandomGenerator<TD> gen_d(-1.0f, 1.0f, 43);
    std::vector<TD> C_data(m * n);
    gen_d.fill_matrix(C_data.data(), m, n);

    // Reference: D_ref = alpha * op(A) * op(B) + beta * C
    std::vector<TD> ref_D(m * n);
    reference_sgemm<TA, TD>(m, n, k, A_data.data(), lda, kALayout, B_data.data(), ldb, kBLayout,
                            C_data.data(), ld, ref_D.data(), ld, alpha_val, beta_val);

    // Actual: pass C == D (same buffer) — the kernel must read C before writing D
    std::vector<TD> act_D(C_data);  // start with C's values
    funcs.run_sgemm(m, n, k, A_data.data(), lda, 0, B_data.data(), ldb, 0, act_D.data(), ld,
                    act_D.data(), ld, alpha_val, beta_val);

    auto result = verify_gemm_result<TD>(ref_D.data(), act_D.data(), m * n, abs_error_threshold,
                                         cosine_sim_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Config: C==D alias test"
                               << "\n  alpha=" << params.alpha << " beta=" << params.beta
                               << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

// ============================================================================
// GTest parameter name generator for SGEMM
// ============================================================================

struct SgemmTestParamNameGenerator
{
    std::string operator()(const testing::TestParamInfo<SgemmTestParams> &info) const
    {
        const auto &p = info.param;
        std::string test_name = format_shape_string(p.m, p.n, p.k);
        if (!p.name.empty()) {
            test_name += "_" + p.name;
        }
        return sanitize_gtest_name(test_name);
    }
};

}  // namespace test
}  // namespace torq_tile
