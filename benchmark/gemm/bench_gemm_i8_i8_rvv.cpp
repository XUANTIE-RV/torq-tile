//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <vector>

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv.h"
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv.h"
#endif
#include "src/gemm/gemm_i8_i8/gemm_clamp_i32_i8_i8/tqt_gemm_clamp_i32_i8_i8_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_clamp_i32_i8p_i8p/tqt_gemm_clamp_i32_i8p_i8p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8/tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p/tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// NonPacked: gemm_clamp_i32_i8_i8_4x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<int32_t> make_nonpacked_clamp_i32_i8_i8_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        tqt_get_n_step_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        tqt_get_a_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        tqt_get_b_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        tqt_get_d_offset_gemm_clamp_i32_i8_i8_4x1vl_rvv,
        // Wrap to match RunGemmFunc signature (which includes bias parameter)
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, int32_t clamp_min, int32_t clamp_max) {
            (void)bias;
            tqt_run_gemm_clamp_i32_i8_i8_4x1vl_rvv(m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C,
                                                   ldc, D, ldd, clamp_min, clamp_max);
        },
    };
}

// ============================================================================
// NonPacked: gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<int8_t> make_nonpacked_1xnbias_reqi32toi8_qai8_qai8_qsi8cxt_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
        // Wrap the run function to match the expected signature
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, int8_t clamp_min, int8_t clamp_max) {
            // Create dummy quantization params for benchmarking
            // Use dynamic allocation with static cache to handle large N values
            static std::vector<float> scale_b_vec;
            if (scale_b_vec.size() < n) {
                scale_b_vec.resize(std::max(n, static_cast<size_t>(4096)));
                std::fill(scale_b_vec.begin(), scale_b_vec.end(), 0.5f);
            }
            struct tqt_qai8_qai8_qsi8cx_params params;
            params.scale_a = 0.5f;
            params.zero_point_a = 0;
            params.scale_b = scale_b_vec.data();
            params.quant_channel_b = (int32_t)n;
            params.scale_d = 0.8f;
            params.zero_point_d = 0;
            tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv(
                m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C, ldc, D, ldd, bias, clamp_min,
                clamp_max, &params);
        },
    };
}

// ============================================================================
// NonPacked: gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<int8_t> make_nonpacked_mx1bias_reqi32toi8_qai8_qsi8dx_qai8_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_a_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_b_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv,
        // Wrap the run function to match the expected signature
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, int8_t clamp_min, int8_t clamp_max) {
            // Create dummy quantization params for benchmarking
            // Use dynamic allocation with static cache to handle large M values
            static std::vector<float> scale_a_vec;
            if (scale_a_vec.size() < m) {
                scale_a_vec.resize(std::max(m, static_cast<size_t>(4096)));
                std::fill(scale_a_vec.begin(), scale_a_vec.end(), 0.5f);
            }
            struct tqt_qai8_qsi8dx_qai8_params params;
            params.scale_a = scale_a_vec.data();
            params.quant_channel_a = (int32_t)m;
            params.scale_b = 0.5f;
            params.zero_point_b = 0;
            params.scale_d = 0.8f;
            params.zero_point_d = 0;
            tqt_run_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv(
                m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C, ldc, D, ldd, bias, clamp_min,
                clamp_max, &params);
        },
    };
}

// ============================================================================
// Packed: gemm_clamp_i32_i8p_i8p_4x1vl_rvv
// ============================================================================

static PackedGemmFuncs<int32_t> make_packed_clamp_i32_i8p_i8p_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_n_step_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_d_offset_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_run_a_pack_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_clamp_i32_i8p_i8p_4x1vl_rvv,
        // Wrap to match RunGemmFunc signature (which includes bias parameter)
        [](size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
           const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
           size_t ldd, const void *bias, int32_t clamp_min, int32_t clamp_max) {
            (void)bias;
            tqt_run_gemm_clamp_i32_i8p_i8p_4x1vl_rvv(m, n, k, A_packed, lda, k_idx_a, B_packed, ldb,
                                                     k_idx_b, C, ldc, D, ldd, clamp_min, clamp_max);
        },
    };
}

// ============================================================================
// Packed: gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv
// ============================================================================

static PackedGemmFuncs<int8_t> make_packed_1xnbias_reqi32toi8_qai8_qai8p_qsi8cxp_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv,
        // Wrap the run function to match the expected signature
        [](size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
           const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
           size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max) {
            // Create dummy quantization params for benchmarking
            // Use dynamic allocation with static cache to handle large N values
            static std::vector<float> scale_b_vec;
            if (scale_b_vec.size() < n) {
                scale_b_vec.resize(std::max(n, static_cast<size_t>(4096)));
                std::fill(scale_b_vec.begin(), scale_b_vec.end(), 0.5f);
            }
            struct tqt_qai8_qai8_qsi8cx_params params;
            params.scale_a = 0.5f;
            params.zero_point_a = 0;
            params.scale_b = scale_b_vec.data();
            params.quant_channel_b = (int32_t)n;
            params.scale_d = 0.8f;
            params.zero_point_d = 0;
            tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
                m, n, k, A_packed, lda, k_idx_a, B_packed, ldb, k_idx_b, C, ldc, D, ldd, bias,
                clamp_min, clamp_max, &params);
        },
    };
}

// ============================================================================
// Packed: gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv
// ============================================================================

static PackedGemmFuncs<int8_t> make_packed_mx1bias_reqi32toi8_qai8_qsi8dxp_qai8p_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_n_step_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_d_offset_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_a_pack_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv,
        // Wrap the run function to match the expected signature
        [](size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
           const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
           size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max) {
            // Create dummy quantization params for benchmarking
            // Use dynamic allocation with static cache to handle large M values
            static std::vector<float> scale_a_vec;
            if (scale_a_vec.size() < m) {
                scale_a_vec.resize(std::max(m, static_cast<size_t>(4096)));
                std::fill(scale_a_vec.begin(), scale_a_vec.end(), 0.5f);
            }
            struct tqt_qai8_qsi8dx_qai8_params params;
            params.scale_a = scale_a_vec.data();
            params.quant_channel_a = (int32_t)m;
            params.scale_b = 0.5f;
            params.zero_point_b = 0;
            params.scale_d = 0.8f;
            params.zero_point_d = 0;
            tqt_run_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv(
                m, n, k, A_packed, lda, k_idx_a, B_packed, ldb, k_idx_b, C, ldc, D, ldd, bias,
                clamp_min, clamp_max, &params);
        },
    };
}

// ============================================================================
// B-only-packed: gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv
// ============================================================================

static BOnlyPackedGemmFuncs<int8_t>
make_b_only_packed_1xnbias_reqi32toi8_qai8_qai8_qsi8cxp_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv,
        // Wrap the run function to match the expected signature
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a,
           const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
           size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max) {
            // Create dummy quantization params for benchmarking
            static std::vector<float> scale_b_vec;
            if (scale_b_vec.size() < n) {
                scale_b_vec.resize(std::max(n, static_cast<size_t>(4096)));
                std::fill(scale_b_vec.begin(), scale_b_vec.end(), 0.5f);
            }
            struct tqt_qai8_qai8_qsi8cx_params params;
            params.scale_a = 0.5f;
            params.zero_point_a = 0;
            params.scale_b = scale_b_vec.data();
            params.quant_channel_b = (int32_t)n;
            params.scale_d = 0.8f;
            params.zero_point_d = 0;
            tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv(
                m, n, k, A, lda, k_idx_a, B_packed, ldb, k_idx_b, C, ldc, D, ldd, bias, clamp_min,
                clamp_max, &params);
        },
    };
}

// ============================================================================
// NonPacked: gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float> make_nonpacked_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv,
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, float clamp_min, float clamp_max) {
            static std::vector<float> scale_a_vec, scale_b_vec;
            static std::vector<int32_t> zp_a_vec;
            const size_t need_m = std::max(m, static_cast<size_t>(4096));
            const size_t need_n = std::max(n, static_cast<size_t>(4096));
            if (scale_a_vec.size() < need_m) {
                scale_a_vec.assign(need_m, 0.5f);
                zp_a_vec.assign(need_m, 0);
            }
            if (scale_b_vec.size() < need_n) {
                scale_b_vec.assign(need_n, 0.5f);
            }
            tqt_qai8dxp_qsi8cxp_params params;
            params.scale_a = scale_a_vec.data();
            params.zero_point_a = zp_a_vec.data();
            params.scale_b = scale_b_vec.data();
            tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv(
                m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C, ldc, D, ldd, bias, clamp_min,
                clamp_max, &params);
        },
    };
}

// ============================================================================
// Packed: gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv
// ============================================================================

static PackedGemmFuncs<float> make_packed_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv,
        [](size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
           const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
           size_t ldd, const void *bias, float clamp_min, float clamp_max) {
            static std::vector<float> scale_a_vec, scale_b_vec;
            static std::vector<int32_t> zp_a_vec;
            const size_t need_m = std::max(m, static_cast<size_t>(4096));
            const size_t need_n = std::max(n, static_cast<size_t>(4096));
            if (scale_a_vec.size() < need_m) {
                scale_a_vec.assign(need_m, 0.5f);
                zp_a_vec.assign(need_m, 0);
            }
            if (scale_b_vec.size() < need_n) {
                scale_b_vec.assign(need_n, 0.5f);
            }
            tqt_qai8dxp_qsi8cxp_params params;
            params.scale_a = scale_a_vec.data();
            params.zero_point_a = zp_a_vec.data();
            params.scale_b = scale_b_vec.data();
            tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv(
                m, n, k, A_packed, lda, k_idx_a, B_packed, ldb, k_idx_b, C, ldc, D, ldd, bias,
                clamp_min, clamp_max, &params);
        },
    };
}

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
// ============================================================================
// NonPacked: gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_nonpacked_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv,
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, float16_t clamp_min, float16_t clamp_max) {
            static std::vector<float> scale_a_vec, scale_b_vec;
            static std::vector<int32_t> zp_a_vec;
            const size_t need_m = std::max(m, static_cast<size_t>(4096));
            const size_t need_n = std::max(n, static_cast<size_t>(4096));
            if (scale_a_vec.size() < need_m) {
                scale_a_vec.assign(need_m, 0.5f);
                zp_a_vec.assign(need_m, 0);
            }
            if (scale_b_vec.size() < need_n) {
                scale_b_vec.assign(need_n, 0.5f);
            }
            tqt_qai8dxp_qsi8cxp_params params;
            params.scale_a = scale_a_vec.data();
            params.zero_point_a = zp_a_vec.data();
            params.scale_b = scale_b_vec.data();
            tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv(
                m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, C, ldc, D, ldd, bias, clamp_min,
                clamp_max, &params);
        },
    };
}

// ============================================================================
// Packed: gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv
// ============================================================================

static PackedGemmFuncs<float16_t> make_packed_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv,
        [](size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
           const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
           size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max) {
            static std::vector<float> scale_a_vec, scale_b_vec;
            static std::vector<int32_t> zp_a_vec;
            const size_t need_m = std::max(m, static_cast<size_t>(4096));
            const size_t need_n = std::max(n, static_cast<size_t>(4096));
            if (scale_a_vec.size() < need_m) {
                scale_a_vec.assign(need_m, 0.5f);
                zp_a_vec.assign(need_m, 0);
            }
            if (scale_b_vec.size() < need_n) {
                scale_b_vec.assign(need_n, 0.5f);
            }
            tqt_qai8dxp_qsi8cxp_params params;
            params.scale_a = scale_a_vec.data();
            params.zero_point_a = zp_a_vec.data();
            params.scale_b = scale_b_vec.data();
            tqt_run_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv(
                m, n, k, A_packed, lda, k_idx_a, B_packed, ldb, k_idx_b, C, ldc, D, ldd, bias,
                clamp_min, clamp_max, &params);
        },
    };
}
#endif  // __riscv_zfh && __riscv_zvfh

// ============================================================================
// Registration
// ============================================================================

void register_gemm_i8_i8_rvv_benchmarks()
{
    register_nonpacked_gemm_benchmark<int8_t, int32_t, BLayout::kRowMajor, int32_t>(
        "tqt_gemm_clamp_i32_i8_i8_4x1vl_rvv", make_nonpacked_clamp_i32_i8_i8_4x1vl_rvv());

    register_nonpacked_gemm_benchmark<int8_t, int8_t, BLayout::kTransposed, int8_t>(
        "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv",
        make_nonpacked_1xnbias_reqi32toi8_qai8_qai8_qsi8cxt_4x1vl_rvv());

    register_nonpacked_gemm_benchmark<int8_t, int8_t, BLayout::kRowMajor, int8_t>(
        "tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dx_qai8_4x1vl_rvv",
        make_nonpacked_mx1bias_reqi32toi8_qai8_qsi8dx_qai8_4x1vl_rvv());

    register_packed_gemm_benchmark<int8_t, int32_t, int32_t>(
        "tqt_gemm_clamp_i32_i8p_i8p_4x1vl_rvv", make_packed_clamp_i32_i8p_i8p_4x1vl_rvv());

    register_packed_gemm_benchmark<int8_t, int8_t>(
        "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv",
        make_packed_1xnbias_reqi32toi8_qai8_qai8p_qsi8cxp_4x1vl_rvv());

    register_packed_gemm_benchmark<int8_t, int8_t>(
        "tqt_gemm_mx1bias_reqi32toi8_clamp_qai8_qsi8dxp_qai8p_4x1vl_rvv",
        make_packed_mx1bias_reqi32toi8_qai8_qsi8dxp_qai8p_4x1vl_rvv());

    register_b_only_packed_gemm_benchmark<int8_t, int8_t>(
        "tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxp_4x1vl_rvv",
        make_b_only_packed_1xnbias_reqi32toi8_qai8_qai8_qsi8cxp_4x1vl_rvv());

    register_nonpacked_gemm_benchmark<int8_t, float, BLayout::kTransposed, float>(
        "tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv",
        make_nonpacked_1xnbias_clamp_f32_qai8dx_qsi8cxt_4x1vl_rvv());

    register_packed_gemm_benchmark<int8_t, float, float>(
        "tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv",
        make_packed_1xnbias_clamp_f32_qai8dxp_qsi8cxp_4x1vl_rvv());

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
    register_nonpacked_gemm_benchmark<int8_t, float16_t, BLayout::kTransposed, float16_t>(
        "tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv",
        make_nonpacked_1xnbias_clamp_f16_qai8dx_qsi8cxt_4x1vl_rvv());

    register_packed_gemm_benchmark<int8_t, float16_t, float16_t>(
        "tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv",
        make_packed_1xnbias_clamp_f16_qai8dxp_qsi8cxp_4x1vl_rvv());
#endif
}

}  // namespace benchmark
}  // namespace torq_tile
