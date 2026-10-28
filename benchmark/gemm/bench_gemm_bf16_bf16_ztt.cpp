//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <benchmark/benchmark.h>

#include <vector>

#include "benchmark/common/benchmark_common.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16t/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16t/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16p_bf16p_bf16p/tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16p_bf16p_bf16p/tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt.h"

namespace torq_tile
{
namespace benchmark
{
namespace native
{
static const tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_ukernel rowmajor = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt,
};

static const tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_ukernel transposed = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
};

static const tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_ukernel packed = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_d_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_bias_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_a_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_b_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_bt_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_c_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_bias_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_d_unpack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
};

static void bench_gemm(::benchmark::State &state, int layout)
{
    const auto shape = global_gemm_shape();
    const size_t m = shape.m, n = shape.n, k = shape.k;
    const size_t nlen = packed.get_m_step(), nstep = packed.get_n_step();
    if (layout != 2 && (m % nlen != 0 || n % nstep != 0 || k % nstep != 0)) {
        state.SkipWithError("Nonpacked Ztt requires M/N/K divisible by the square size");
        return;
    }
    std::vector<bfloat16_t> a(m * k), b(k * n), d(m * n);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<bfloat16_t>(0.125f);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<bfloat16_t>(0.25f);
    std::vector<bfloat16_t> ap, bp, dp;
    if (layout == 2) {
        ap.resize(packed.get_a_packed_size(m, k) / sizeof(bfloat16_t));
        bp.resize(packed.get_b_packed_size(n, k) / sizeof(bfloat16_t));
        dp.resize(packed.get_d_packed_size(m, n) / sizeof(bfloat16_t));
        packed.run_a_pack(m, k, k, a.data(), ap.data());
        packed.run_b_pack(n, k, n, b.data(), bp.data());
    }
    for (auto _ : state) {
        if (layout == 2) {
            packed.run_gemm(m, n, k, ap.data(), bp.data(), nullptr, dp.data(), nullptr, -INFINITY,
                            INFINITY);
        } else if (layout == 1) {
            transposed.run_gemm(m, n, k, a.data(), k, 0, b.data(), k, 0, nullptr, n, d.data(), n,
                                nullptr, -INFINITY, INFINITY);
        } else {
            rowmajor.run_gemm(m, n, k, a.data(), k, 0, b.data(), n, 0, nullptr, n, d.data(), n,
                              nullptr, -INFINITY, INFINITY);
        }
        ::benchmark::ClobberMemory();
    }
    state.counters["FLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate);
    state.counters["M"] = static_cast<double>(m);
    state.counters["N"] = static_cast<double>(n);
    state.counters["K"] = static_cast<double>(k);
}

}  // namespace native
namespace f32acc
{
static const tqt_gemm_1xnbias_clamp_bf16_bf16_bf16_ukernel rowmajor = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt,
};

static const tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_ukernel transposed = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
};

static const tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_ukernel packed = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_bias_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_a_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_b_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_bt_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_c_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_bias_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_d_unpack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
};

static void bench_gemm(::benchmark::State &state, int layout)
{
    const auto shape = global_gemm_shape();
    const size_t m = shape.m, n = shape.n, k = shape.k;
    const size_t nlen = packed.get_m_step(), nstep = packed.get_n_step();
    if (layout != 2 && (m % nlen != 0 || n % nstep != 0 || k % nstep != 0)) {
        state.SkipWithError("Nonpacked Ztt requires M/N/K divisible by the square size");
        return;
    }
    std::vector<bfloat16_t> a(m * k), b(k * n), d(m * n);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<bfloat16_t>(0.125f);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<bfloat16_t>(0.25f);
    std::vector<bfloat16_t> ap, bp, dp;
    if (layout == 2) {
        ap.resize(packed.get_a_packed_size(m, k) / sizeof(bfloat16_t));
        bp.resize(packed.get_b_packed_size(n, k) / sizeof(bfloat16_t));
        dp.resize(packed.get_d_packed_size(m, n) / sizeof(bfloat16_t));
        packed.run_a_pack(m, k, k, a.data(), ap.data());
        packed.run_b_pack(n, k, n, b.data(), bp.data());
    }
    for (auto _ : state) {
        if (layout == 2) {
            packed.run_gemm(m, n, k, ap.data(), bp.data(), nullptr, dp.data(), nullptr, -INFINITY,
                            INFINITY);
        } else if (layout == 1) {
            transposed.run_gemm(m, n, k, a.data(), k, 0, b.data(), k, 0, nullptr, n, d.data(), n,
                                nullptr, -INFINITY, INFINITY);
        } else {
            rowmajor.run_gemm(m, n, k, a.data(), k, 0, b.data(), n, 0, nullptr, n, d.data(), n,
                              nullptr, -INFINITY, INFINITY);
        }
        ::benchmark::ClobberMemory();
    }
    state.counters["FLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate);
    state.counters["M"] = static_cast<double>(m);
    state.counters["N"] = static_cast<double>(n);
    state.counters["K"] = static_cast<double>(k);
}

}  // namespace f32acc
void register_gemm_bf16_bf16_ztt_benchmarks()
{
    ::benchmark::RegisterBenchmark("gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_ztt",
                                   native::bench_gemm, 0);
    ::benchmark::RegisterBenchmark("gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt",
                                   native::bench_gemm, 1);
    ::benchmark::RegisterBenchmark("gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt",
                                   native::bench_gemm, 2);
    ::benchmark::RegisterBenchmark("gemm_1xnbias_clamp_bf16_bf16_bf16_1nlenxpnlen_f32acc_ztt",
                                   f32acc::bench_gemm, 0);
    ::benchmark::RegisterBenchmark("gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt",
                                   f32acc::bench_gemm, 1);
    ::benchmark::RegisterBenchmark("gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt",
                                   f32acc::bench_gemm, 2);
}
}  // namespace benchmark
}  // namespace torq_tile
