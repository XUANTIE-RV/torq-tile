//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <vector>

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_u8_u8/gemm_mx1bias_1xnbias_clamp_f32_u8_u8t/tqt_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv.h"
#include "src/gemm/gemm_u8_u8/gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p/tqt_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv.h"
#include "src/gemm/gemm_u8_u8/gemm_mx1bias_1xnbias_i32_u8_u8t/tqt_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv.h"
#include "src/gemm/gemm_u8_u8/gemm_mx1bias_1xnbias_i32_u8p_u8p/tqt_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv.h"
#include "src/gemm/gemm_u8_u8/tqt_params_u8_u8.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// NonPacked: gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<int32_t> make_nonpacked_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv,
        tqt_get_a_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv,
        tqt_get_b_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv,
        // Wrap to match RunGemmFunc signature (dual-bias -> single bias)
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, int32_t clamp_min, int32_t clamp_max) {
            (void)bias;
            (void)clamp_min;
            (void)clamp_max;
            // Provide zero-filled bias_m and bias_n for benchmarking
            static std::vector<int32_t> bias_m_vec, bias_n_vec;
            const size_t need_m = std::max(m, static_cast<size_t>(4096));
            const size_t need_n = std::max(n, static_cast<size_t>(4096));
            if (bias_m_vec.size() < need_m) {
                bias_m_vec.assign(need_m, 0);
            }
            if (bias_n_vec.size() < need_n) {
                bias_n_vec.assign(need_n, 0);
            }
            tqt_run_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(m, n, k, A, lda, k_idx_a, B, ldb,
                                                              k_idx_b, C, ldc, D, ldd,
                                                              bias_m_vec.data(), bias_n_vec.data());
        },
    };
}

// ============================================================================
// Packed: gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv
// ============================================================================

static PackedGemmFuncs<int32_t> make_packed_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
        tqt_run_a_pack_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
        // Wrap to match RunGemmFunc signature (dual-bias -> single bias)
        [](size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
           const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
           size_t ldd, const void *bias, int32_t clamp_min, int32_t clamp_max) {
            (void)bias;
            (void)clamp_min;
            (void)clamp_max;
            // Provide zero-filled bias_m and bias_n for benchmarking
            static std::vector<int32_t> bias_m_vec, bias_n_vec;
            const size_t need_m = std::max(m, static_cast<size_t>(4096));
            const size_t need_n = std::max(n, static_cast<size_t>(4096));
            if (bias_m_vec.size() < need_m) {
                bias_m_vec.assign(need_m, 0);
            }
            if (bias_n_vec.size() < need_n) {
                bias_n_vec.assign(need_n, 0);
            }
            tqt_run_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(
                m, n, k, A_packed, lda, k_idx_a, B_packed, ldb, k_idx_b, C, ldc, D, ldd,
                bias_m_vec.data(), bias_n_vec.data());
        },
    };
}

// ============================================================================
// NonPacked: gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float> make_nonpacked_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv,
        tqt_get_a_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv,
        tqt_get_b_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv,
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, float clamp_min, float clamp_max) {
            (void)bias;
            // Provide zero-filled bias_m and bias_n for benchmarking
            static std::vector<int32_t> bias_m_vec, bias_n_vec;
            const size_t need_m = std::max(m, static_cast<size_t>(4096));
            const size_t need_n = std::max(n, static_cast<size_t>(4096));
            if (bias_m_vec.size() < need_m) {
                bias_m_vec.assign(need_m, 0);
            }
            if (bias_n_vec.size() < need_n) {
                bias_n_vec.assign(need_n, 0);
            }
            struct tqt_u8_u8_clamp_f32_params params;
            params.scale = 1.0f;
            tqt_run_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(
                m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C, ldc, D, ldd, bias_m_vec.data(),
                bias_n_vec.data(), nullptr, clamp_min, clamp_max, &params);
        },
    };
}

// ============================================================================
// Packed: gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv
// ============================================================================

static PackedGemmFuncs<float> make_packed_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
        tqt_run_a_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
        [](size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
           const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
           size_t ldd, const void *bias, float clamp_min, float clamp_max) {
            (void)bias;
            // Provide zero-filled bias_m and bias_n for benchmarking
            static std::vector<int32_t> bias_m_vec, bias_n_vec;
            const size_t need_m = std::max(m, static_cast<size_t>(4096));
            const size_t need_n = std::max(n, static_cast<size_t>(4096));
            if (bias_m_vec.size() < need_m) {
                bias_m_vec.assign(need_m, 0);
            }
            if (bias_n_vec.size() < need_n) {
                bias_n_vec.assign(need_n, 0);
            }
            struct tqt_u8_u8_clamp_f32_params params;
            params.scale = 1.0f;
            tqt_run_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(
                m, n, k, A_packed, lda, k_idx_a, B_packed, ldb, k_idx_b, C, ldc, D, ldd,
                bias_m_vec.data(), bias_n_vec.data(), nullptr, clamp_min, clamp_max, &params);
        },
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_gemm_u8_u8_rvv_benchmarks()
{
    register_nonpacked_gemm_benchmark<uint8_t, int32_t, BLayout::kTransposed, int32_t>(
        "tqt_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv",
        make_nonpacked_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv());

    register_packed_gemm_benchmark<uint8_t, int32_t, int32_t>(
        "tqt_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv",
        make_packed_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv());

    register_nonpacked_gemm_benchmark<uint8_t, float, BLayout::kTransposed, float>(
        "tqt_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv",
        make_nonpacked_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv());

    register_packed_gemm_benchmark<uint8_t, float, float>(
        "tqt_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv",
        make_packed_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
