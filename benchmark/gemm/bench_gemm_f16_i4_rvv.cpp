//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f16_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16_qai4c2t/tqt_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qai4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv.h"
#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv.h"

using namespace torq_tile::benchmark;

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// QuantizedWeight: gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv
// ============================================================================

static QuantizedWeightGemmFuncs<float16_t, float16_t>
make_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv()
{
    QuantizedWeightGemmFuncs<float16_t, float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_a_offset = tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_b_offset = tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv;
    return funcs;
}

// ============================================================================
// PackedQuantizedWeight: gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv
// ============================================================================

static PackedQuantizedWeightGemmFuncs<float16_t, float16_t>
make_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv()
{
    PackedQuantizedWeightGemmFuncs<float16_t, float16_t> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv;
    return funcs;
}

// ============================================================================
// QuantizedWeight: gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv
// ============================================================================

static QuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv()
{
    QuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_a_offset = tqt_get_a_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_b_offset = tqt_get_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv;
    return funcs;
}

// ============================================================================
// PackedQuantizedWeight: gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv
// ============================================================================

static PackedQuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv()
{
    PackedQuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv;
    return funcs;
}

// ============================================================================
// Registration
// ============================================================================

static QuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv()
{
    QuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv;
    funcs.get_a_offset = tqt_get_a_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv;
    funcs.get_b_offset = tqt_get_b_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv;
    return funcs;
}

static void bench_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(::benchmark::State &state)
{
    const size_t m = state.range(0), n = state.range(1), k = state.range(2), bl = 32;
    const size_t lds = k / bl;
    std::vector<float16_t> a(m * k, (float16_t)0.25f);
    std::vector<uint8_t> b(n * k / 2, 0xe3);
    std::vector<float> scale(n * lds, 0.125f), offset(n * lds, -0.5f), d(m * n);
    for (auto _ : state) {
        tqt_run_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv(
            m, n, k, bl, a.data(), k, 0, b.data(), k, 0, scale.data(), lds, offset.data(), lds,
            nullptr, n, d.data(), n, nullptr, -INFINITY, INFINITY);
        ::benchmark::DoNotOptimize(d.data());
        ::benchmark::ClobberMemory();
    }
    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

static PackedQuantizedWeightGemmFuncs<float16_t, float>
make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv()
{
    PackedQuantizedWeightGemmFuncs<float16_t, float> funcs;
    funcs.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv;
    funcs.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv;
    funcs.get_d_offset = tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv;
    funcs.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv;
    funcs.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv;
    funcs.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv;
    funcs.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv;
    funcs.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv;
    return funcs;
}

static void bench_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(::benchmark::State &state)
{
    const size_t m = state.range(0), n = state.range(1), k = state.range(2), bl = 32;
    const size_t lds = k / bl;
    std::vector<float16_t> a(m * k, (float16_t)0.25f);
    std::vector<uint8_t> b(n * k / 2, 0xe3);
    std::vector<float> scale(n * lds, 0.125f), offset(n * lds, -0.5f), d(m * n);
    std::vector<uint8_t> ap(
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(m, k)),
        bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(n, k));
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(m, k, k, k, 0, a.data(),
                                                                 ap.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(n, k, k, k, 0, b.data(),
                                                                  bp.data());
    for (auto _ : state) {
        tqt_run_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(
            m, n, k, bl, ap.data(), k, 0, bp.data(), k, 0, scale.data(), lds, offset.data(), lds,
            nullptr, n, d.data(), n, nullptr, -INFINITY, INFINITY);
        ::benchmark::DoNotOptimize(d.data());
        ::benchmark::ClobberMemory();
    }
    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

void register_gemm_f16_i4_rvv_benchmarks()
{
    register_quantized_weight_gemm_benchmark<float16_t, float>(
        "gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv",
        make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x1vl_rvv());
    ::benchmark::RegisterBenchmark("gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv",
                                   bench_gemm_1xnbias_clamp_f32_f16_qai4c2t_8x1vl_rvv)
        ->Args({static_cast<int64_t>(global_gemm_shape().m),
                static_cast<int64_t>(global_gemm_shape().n),
                static_cast<int64_t>(global_gemm_shape().k)})
        ->ArgNames({"m", "n", "k"});
    register_packed_quantized_weight_gemm_benchmark<float16_t, float>(
        "gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv",
        make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv());
    ::benchmark::RegisterBenchmark("gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv",
                                   bench_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv)
        ->Args({static_cast<int64_t>(global_gemm_shape().m),
                static_cast<int64_t>(global_gemm_shape().n),
                static_cast<int64_t>(global_gemm_shape().k)})
        ->ArgNames({"m", "n", "k"});

    // D = fp16 variants
    auto funcs_f16 = make_gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv();
    register_quantized_weight_gemm_benchmark<float16_t, float16_t>(
        "gemm_1xnbias_clamp_f16_f16_qsi4c2t_8x2vl_rvv", funcs_f16);

    auto packed_funcs_f16 = make_gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv();
    register_packed_quantized_weight_gemm_benchmark<float16_t, float16_t>(
        "gemm_1xnbias_clamp_f16_f16p_qsi4c2p_8x2vl_rvv", packed_funcs_f16);

    // D = fp32 variants (A still fp16, scale still fp16)
    auto funcs_f32 = make_gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv();
    register_quantized_weight_gemm_benchmark<float16_t, float>(
        "gemm_1xnbias_clamp_f32_f16_qsi4c2t_8x2vl_rvv", funcs_f32);

    auto packed_funcs_f32 = make_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv();
    register_packed_quantized_weight_gemm_benchmark<float16_t, float>(
        "gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x2vl_rvv", packed_funcs_f32);
}

}  // namespace benchmark
}  // namespace torq_tile
