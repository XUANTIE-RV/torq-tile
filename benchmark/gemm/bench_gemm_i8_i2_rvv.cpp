//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <vector>

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_i8_i2/gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i2/gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i2/tqt_params_i8_i2.h"

namespace torq_tile
{
namespace benchmark
{

namespace
{

// Reusable static caches sized to the largest (M, N) seen so far. Avoids re-allocating
// per call; values are not used for correctness, only to keep the kernel from short-circuiting
// on a nullptr `params`.
struct DummyGemmI8I2Params
{
    std::vector<float> scale_a;
    std::vector<int32_t> zp_a;
    std::vector<float> scale_b;
    std::vector<int32_t> sum_b;

    tqt_qai8dx_qsi2cx_f32_params build(size_t m, size_t n, size_t k)
    {
        if (scale_a.size() < m) {
            scale_a.assign(std::max(m, static_cast<size_t>(4096)), 1.0f / 32.0f);
            zp_a.assign(std::max(m, static_cast<size_t>(4096)), 0);
        }
        if (scale_b.size() < n) {
            const size_t cap = std::max(n, static_cast<size_t>(4096));
            scale_b.assign(cap, 0.5f);
            sum_b.assign(cap, static_cast<int32_t>(k / 4));
        }
        tqt_qai8dx_qsi2cx_f32_params p{};
        p.scale_a = scale_a.data();
        p.zero_point_a = zp_a.data();
        p.scale_b = scale_b.data();
        p.sum_b = sum_b.data();
        return p;
    }
};

inline DummyGemmI8I2Params &dummy_params()
{
    static DummyGemmI8I2Params p;
    return p;
}

}  // namespace

// ============================================================================
// NonPacked: gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv  (a_bt)
// ============================================================================

static NonPackedGemmFuncs<float> make_nonpacked_1xnbias_qai8dx_qsi2cxt_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv,
        // Wrap the run function: ignore C/ldc, pass dummy params and default LUT.
        [](size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
           size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
           const void *bias, float clamp_min, float clamp_max) {
            (void)C;
            (void)ldc;
            tqt_qai8dx_qsi2cx_f32_params params = dummy_params().build(m, n, k);
            tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv(
                m, n, k, A, lda, k_idx_a, B, ldb, k_idx_b, D, ldd, bias, clamp_min, clamp_max,
                /*lut=*/nullptr, &params);
        },
    };
}

// ============================================================================
// Packed: gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv  (ap_bp)
// ============================================================================

static PackedGemmFuncs<float> make_packed_1xnbias_qai8dxp_qsi2cxp_4x1vl_rvv()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv,
        // Wrap the run function: ignore C/ldc, pass dummy params and default LUT.
        [](size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
           const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
           size_t ldd, const void *bias, float clamp_min, float clamp_max) {
            (void)C;
            (void)ldc;
            tqt_qai8dx_qsi2cx_f32_params params = dummy_params().build(m, n, k);
            tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv(
                m, n, k, A_packed, lda, k_idx_a, B_packed, ldb, k_idx_b, D, ldd, bias, clamp_min,
                clamp_max, /*lut=*/nullptr, &params);
        },
    };
}

// ============================================================================
// Registration
// ============================================================================

void register_gemm_i8_i2_rvv_benchmarks()
{
    // Inputs are int8 (A) / packed int2 (B treated as int8 byte stream); output is fp32.
    register_nonpacked_gemm_benchmark<int8_t, float, BLayout::kTransposed, float>(
        "tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi2cxt_4x1vl_rvv",
        make_nonpacked_1xnbias_qai8dx_qsi2cxt_4x1vl_rvv());

    register_packed_gemm_benchmark<int8_t, float, float>(
        "tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi2cxp_4x1vl_rvv",
        make_packed_1xnbias_qai8dxp_qsi2cxp_4x1vl_rvv());
}

}  // namespace benchmark
}  // namespace torq_tile
