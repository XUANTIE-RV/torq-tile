//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <benchmark/benchmark.h>

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv.h"
#endif  // __riscv_zfh && __riscv_zvfh
#if defined(__riscv_zvfbfwma)
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv.h"
#endif  // __riscv_zvfbfwma

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// W4A8 (qai8dx × qsi4cx) GEMM benchmark utilities
// ============================================================================

namespace
{

template <typename TD>
struct W4A8GemmBenchFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

template <typename TD>
struct W4A8PackedGemmBenchFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A_packed, size_t lda,
                                 size_t k_idx_a, const void *B_packed, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

template <typename TD>
void run_w4a8_gemm_benchmark(::benchmark::State &state, const W4A8GemmBenchFuncs<TD> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldd = n;

    std::vector<int8_t> A(m * k);
    std::vector<uint8_t> B(n * (k / 2));
    std::srand(123);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = static_cast<int8_t>(std::rand() % 256 - 128);
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = static_cast<uint8_t>(std::rand() % 256);

    std::vector<float> scale_a(m, 0.01f);
    std::vector<int32_t> zp_a(m, 0);
    std::vector<float> scale_b(n, 0.01f);

    std::vector<TD> D(m * n, static_cast<TD>(0));

    struct tqt_qai8dx_qsi4cx_params params;
    params.scale_a = scale_a.data();
    params.zero_point_a = zp_a.data();
    params.scale_b = scale_b.data();

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, A.data(), lda, 0, B.data(), ldb, 0, nullptr, n, D.data(), ldd,
                       nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

template <typename TD>
void run_w4a8_packed_gemm_benchmark(::benchmark::State &state,
                                    const W4A8PackedGemmBenchFuncs<TD> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldd = n;

    std::vector<int8_t> A(m * k);
    std::vector<uint8_t> B(n * (k / 2));
    std::srand(123);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = static_cast<int8_t>(std::rand() % 256 - 128);
    for (size_t i = 0; i < B.size(); ++i)
        B[i] = static_cast<uint8_t>(std::rand() % 256);

    std::vector<float> scale_a(m, 0.01f);
    std::vector<int32_t> zp_a(m, 0);
    std::vector<float> scale_b(n, 0.01f);

    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_a_pack(m, k, lda, k, 0, A.data(), A_packed.data());
    funcs.run_bt_pack(n, k, ldb, k, 0, B.data(), B_packed.data());

    std::vector<TD> D(m * n, static_cast<TD>(0));

    struct tqt_qai8dx_qsi4cx_params params;
    params.scale_a = scale_a.data();
    params.zero_point_a = zp_a.data();
    params.scale_b = scale_b.data();

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, A_packed.data(), lda, 0, B_packed.data(), ldb, 0, nullptr, n,
                       D.data(), ldd, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

template <typename TD>
inline void register_w4a8_gemm(const char *name, const W4A8GemmBenchFuncs<TD> &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(
        name, [funcs](::benchmark::State &state) { run_w4a8_gemm_benchmark<TD>(state, funcs); });
    const auto &shape = global_gemm_shape();
    bm->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
              static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

template <typename TD>
inline void register_w4a8_packed_gemm(const char *name, const W4A8PackedGemmBenchFuncs<TD> &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
        run_w4a8_packed_gemm_benchmark<TD>(state, funcs);
    });
    const auto &shape = global_gemm_shape();
    bm->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
              static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

}  // namespace

// ============================================================================
// Function-pointer factories per kernel variant
// ============================================================================

#if defined(__riscv_zfh) && defined(__riscv_zvfh)
static W4A8GemmBenchFuncs<float16_t> make_funcs_f16_qai8dx_qsi4cxt()
{
    W4A8GemmBenchFuncs<float16_t> f;
    f.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    f.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    f.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv;
    return f;
}

static W4A8PackedGemmBenchFuncs<float16_t> make_funcs_f16_qai8dxp_qsi4cxp()
{
    W4A8PackedGemmBenchFuncs<float16_t> f;
    f.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_gemm = tqt_run_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv;
    return f;
}
#endif  // __riscv_zfh && __riscv_zvfh

static W4A8GemmBenchFuncs<float> make_funcs_f32_qai8dx_qsi4cxt()
{
    W4A8GemmBenchFuncs<float> f;
    f.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    f.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    f.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv;
    return f;
}

static W4A8PackedGemmBenchFuncs<float> make_funcs_f32_qai8dxp_qsi4cxp()
{
    W4A8PackedGemmBenchFuncs<float> f;
    f.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_gemm = tqt_run_gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv;
    return f;
}

#if defined(__riscv_zvfbfwma)
static W4A8GemmBenchFuncs<bfloat16_t> make_funcs_bf16_qai8dx_qsi4cxt()
{
    W4A8GemmBenchFuncs<bfloat16_t> f;
    f.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    f.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    f.run_gemm = tqt_run_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv;
    return f;
}

static W4A8PackedGemmBenchFuncs<bfloat16_t> make_funcs_bf16_qai8dxp_qsi4cxp()
{
    W4A8PackedGemmBenchFuncs<bfloat16_t> f;
    f.get_m_step = tqt_get_m_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_n_step = tqt_get_n_step_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_a_packed_size = tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_b_packed_size = tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_a_pack = tqt_run_a_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_gemm = tqt_run_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv;
    return f;
}
#endif  // __riscv_zvfbfwma

// ============================================================================
// Registration
// ============================================================================

void register_gemm_i8_i4_rvv_benchmarks()
{
    register_w4a8_gemm<float>("gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv",
                              make_funcs_f32_qai8dx_qsi4cxt());
    register_w4a8_packed_gemm<float>("gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv",
                                     make_funcs_f32_qai8dxp_qsi4cxp());
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
    register_w4a8_gemm<float16_t>("gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv",
                                  make_funcs_f16_qai8dx_qsi4cxt());
    register_w4a8_packed_gemm<float16_t>("gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv",
                                         make_funcs_f16_qai8dxp_qsi4cxp());
#endif  // __riscv_zfh && __riscv_zvfh
#if defined(__riscv_zvfbfwma)
    register_w4a8_gemm<bfloat16_t>("gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv",
                                   make_funcs_bf16_qai8dx_qsi4cxt());
    register_w4a8_packed_gemm<bfloat16_t>("gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv",
                                          make_funcs_bf16_qai8dxp_qsi4cxp());
#endif  // __riscv_zvfbfwma
}

}  // namespace benchmark
}  // namespace torq_tile
