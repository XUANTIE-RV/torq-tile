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
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/tqt_params_i8_i4.h"
// W4A8 block-quantized (qai8dx × qsi4c2, qsi8d2 × qsi4c2, qsi8d2 × qai4c2)
// — F32 always compiled.
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t/tqt_gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t/tqt_gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p/tqt_gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv.h"
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi4cxp_4x1vl_rvv.h"
// W4A8 BQ F16
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qai8dxp_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_qai8dxp_qsi4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2_qai4c2t/tqt_gemm_1xnbias_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2_qsi4c2t/tqt_gemm_1xnbias_clamp_f16_qsi8d2_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2p_qai4c2p/tqt_gemm_1xnbias_clamp_f16_qsi8d2p_qai4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f16_qsi8d2p_qsi4c2p/tqt_gemm_1xnbias_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv.h"
#endif  // __riscv_zfh && __riscv_zvfh
#if defined(__riscv_zvfbfwma)
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_bf16_qai8dx_qsi4cxt_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp/tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4cxp_4x1vl_rvv.h"
// W4A8 BQ BF16
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dx_qsi4c2t/tqt_gemm_1xnbias_clamp_bf16_qai8dx_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qai8dxp_qsi4c2p/tqt_gemm_1xnbias_clamp_bf16_qai8dxp_qsi4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2_qai4c2t/tqt_gemm_1xnbias_clamp_bf16_qsi8d2_qai4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t/tqt_gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2p_qai4c2p/tqt_gemm_1xnbias_clamp_bf16_qsi8d2p_qai4c2p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p/tqt_gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv.h"
#endif  // __riscv_zvfbfwma

namespace torq_tile
{
namespace benchmark
{

static tqt_qsi8d32_qsi4c32_params make_qsi4c32_rvv_params(const float *scale_a, size_t ldsa,
                                                          const int16_t *, size_t,
                                                          const float *scale_b, const float *,
                                                          size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

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

// ----------------------------------------------------------------------------
// W4A8 requantized int8 output (qai8dx × qsi4cxt -> qai8): int8 D + int8 clamp +
// tqt_qai8_qai8dx_qsi4cx_params (scale_d / zero_point_d).
// ----------------------------------------------------------------------------

struct W4A8ReqI8GemmBenchFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 int8_t clamp_min, int8_t clamp_max,
                                 const struct tqt_qai8_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

struct W4A8ReqI8PackedGemmBenchFuncs
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
                                 int8_t clamp_min, int8_t clamp_max,
                                 const struct tqt_qai8_qai8dx_qsi4cx_params *params);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

inline void run_w4a8_reqi8_gemm_benchmark(::benchmark::State &state,
                                          const W4A8ReqI8GemmBenchFuncs &funcs)
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

    std::vector<int8_t> D(m * n, 0);

    struct tqt_qai8_qai8dx_qsi4cx_params params;
    params.scale_a = scale_a.data();
    params.zero_point_a = zp_a.data();
    params.scale_b = scale_b.data();
    params.scale_d = 0.05f;
    params.zero_point_d = 0;

    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, A.data(), lda, 0, B.data(), ldb, 0, nullptr, n, D.data(), ldd,
                       nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

inline void run_w4a8_reqi8_packed_gemm_benchmark(::benchmark::State &state,
                                                 const W4A8ReqI8PackedGemmBenchFuncs &funcs)
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

    std::vector<int8_t> D(m * n, 0);

    struct tqt_qai8_qai8dx_qsi4cx_params params;
    params.scale_a = scale_a.data();
    params.zero_point_a = zp_a.data();
    params.scale_b = scale_b.data();
    params.scale_d = 0.05f;
    params.zero_point_d = 0;

    const int8_t clamp_min = -128;
    const int8_t clamp_max = 127;

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, A_packed.data(), lda, 0, B_packed.data(), ldb, 0, nullptr, n,
                       D.data(), ldd, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

inline void register_w4a8_reqi8_gemm(const char *name, const W4A8ReqI8GemmBenchFuncs &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(
        name, [funcs](::benchmark::State &state) { run_w4a8_reqi8_gemm_benchmark(state, funcs); });
    const auto &shape = global_gemm_shape();
    bm->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
              static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

inline void register_w4a8_reqi8_packed_gemm(const char *name,
                                            const W4A8ReqI8PackedGemmBenchFuncs &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
        run_w4a8_reqi8_packed_gemm_benchmark(state, funcs);
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

static W4A8ReqI8GemmBenchFuncs make_funcs_reqi8_qai8dx_qsi4cxt()
{
    W4A8ReqI8GemmBenchFuncs f;
    f.get_m_step = tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv;
    f.get_n_step = tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv;
    f.run_gemm = tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv;
    return f;
}

static W4A8ReqI8PackedGemmBenchFuncs make_funcs_reqi8_qai8dxp_qsi4cxp()
{
    W4A8ReqI8PackedGemmBenchFuncs f;
    f.get_m_step = tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_n_step = tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_a_packed_size =
        tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.get_b_packed_size =
        tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_a_pack = tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_bt_pack = tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
    f.run_gemm = tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv;
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

// Forward declaration: defined later in this file (after the W4A8 BQ helpers).
void register_gemm_i8_i4_blockquant_rvv_benchmarks();

void register_gemm_i8_i4_rvv_benchmarks()
{
    register_w4a8_gemm<float>("gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv",
                              make_funcs_f32_qai8dx_qsi4cxt());
    register_w4a8_packed_gemm<float>("gemm_1xnbias_clamp_f32_qai8dxp_qsi4cxp_4x1vl_rvv",
                                     make_funcs_f32_qai8dxp_qsi4cxp());
    register_w4a8_reqi8_gemm("gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dx_qsi4cxt_4x1vl_rvv",
                             make_funcs_reqi8_qai8dx_qsi4cxt());
    register_w4a8_reqi8_packed_gemm("gemm_1xnbias_reqi32toi8_clamp_qai8_qai8dxp_qsi4cxp_4x1vl_rvv",
                                    make_funcs_reqi8_qai8dxp_qsi4cxp());
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

    // W4A8 block-quantized benchmarks are registered separately below.
    register_gemm_i8_i4_blockquant_rvv_benchmarks();
}

// ============================================================================
// W4A8 block-quantized (W4A8 BQ) GEMM benchmark utilities
// ============================================================================
//
// Three kernel classes share the same block-length parameter (bl) and the same
// 4x1vl tile but use different quantization parameter structs:
//
//   Class 1: qai8dx (A per-row asym) × qsi4c2 (B per-block sym)
//            -> tqt_qai8dx_qsi4c2_params
//   Class 2: qsi8d2 (A per-block sym) × qsi4c2 (B per-block sym)
//            -> tqt_qsi8d2_qsi4c2_params
//   Class 3: qsi8d2 (A per-block sym) × qai4c2 (B per-block asym)
//            -> tqt_qsi8d2_qai4c2_params
//
// Each class has an unpacked variant and a packed variant. F32 is always built;
// F16 is gated by zfh+zvfh and BF16 by zvfbfwma.

namespace
{

// ----------------------------------------------------------------------------
// Class 1: qai8dx (or qai8dxp) × qsi4c2t (or qsi4c2p)
// ----------------------------------------------------------------------------

template <typename TD>
struct W4A8BQ_Qai8dx_Qsi4c2_BenchFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

template <typename TD>
struct W4A8BQ_Qai8dx_Qsi4c2_PackedBenchFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A_packed,
                                 size_t lda, size_t k_idx_a, const void *B_packed, size_t ldb,
                                 size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                                 const void *bias, TD clamp_min, TD clamp_max,
                                 const struct tqt_qai8dx_qsi4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ----------------------------------------------------------------------------
// Class 2: qsi8d2 (or qsi8d2p) × qsi4c2t (or qsi4c2p)
// ----------------------------------------------------------------------------

template <typename TD>
struct W4A8BQ_Qsi8d2_Qsi4c2_BenchFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qsi8d2_qsi4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

template <typename TD>
struct W4A8BQ_Qsi8d2_Qsi4c2_PackedBenchFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A_packed,
                                 size_t lda, size_t k_idx_a, const void *B_packed, size_t ldb,
                                 size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                                 const void *bias, TD clamp_min, TD clamp_max,
                                 const struct tqt_qsi8d2_qsi4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ----------------------------------------------------------------------------
// Class 3: qsi8d2 (or qsi8d2p) × qai4c2t (or qai4c2p)
// ----------------------------------------------------------------------------

template <typename TD>
struct W4A8BQ_Qsi8d2_Qai4c2_BenchFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 TD clamp_min, TD clamp_max,
                                 const struct tqt_qsi8d2_qai4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    RunGemmFunc run_gemm;
};

template <typename TD>
struct W4A8BQ_Qsi8d2_Qai4c2_PackedBenchFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A_packed,
                                 size_t lda, size_t k_idx_a, const void *B_packed, size_t ldb,
                                 size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd,
                                 const void *bias, TD clamp_min, TD clamp_max,
                                 const struct tqt_qsi8d2_qai4c2_params *params);
    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ----------------------------------------------------------------------------
// Shared input setup for W4A8 BQ benchmarks
// ----------------------------------------------------------------------------

struct W4A8BQ_Inputs
{
    size_t m;
    size_t n;
    size_t k;
    size_t bl;
    size_t ldsa;
    size_t ldsb;
    size_t ldzpb;
    std::vector<int8_t> A;
    std::vector<uint8_t> B;
    std::vector<float> scale_a_per_row;    // length m   (class 1)
    std::vector<int32_t> zp_a_per_row;     // length m   (class 1)
    std::vector<float> scale_a_per_block;  // [m, k/bl]  (class 2/3)
    std::vector<float> scale_b_per_block;  // [n, k/bl]  (all classes)
    std::vector<float> zp_b_per_block;     // [n, k/bl]  (class 3)
};

inline W4A8BQ_Inputs make_w4a8_bq_inputs(::benchmark::State &state)
{
    W4A8BQ_Inputs in;
    in.m = static_cast<size_t>(state.range(0));
    in.n = static_cast<size_t>(state.range(1));
    in.k = static_cast<size_t>(state.range(2));
    in.bl = static_cast<size_t>(state.range(3));
    in.ldsa = in.k / in.bl;
    in.ldsb = in.k / in.bl;
    in.ldzpb = in.k / in.bl;

    in.A.resize(in.m * in.k);
    in.B.resize(in.n * (in.k / 2));
    std::srand(123);
    for (size_t i = 0; i < in.A.size(); ++i)
        in.A[i] = static_cast<int8_t>(std::rand() % 256 - 128);
    for (size_t i = 0; i < in.B.size(); ++i)
        in.B[i] = static_cast<uint8_t>(std::rand() % 256);

    in.scale_a_per_row.assign(in.m, 0.01f);
    in.zp_a_per_row.assign(in.m, 0);
    in.scale_a_per_block.assign(in.m * in.ldsa, 0.01f);
    in.scale_b_per_block.assign(in.n * in.ldsb, 0.01f);
    in.zp_b_per_block.assign(in.n * in.ldzpb, 8.0f);
    return in;
}

// ----------------------------------------------------------------------------
// Benchmark runners (Class 1)
// ----------------------------------------------------------------------------

template <typename TD>
void run_w4a8bq_qai8dx_qsi4c2_benchmark(::benchmark::State &state,
                                        const W4A8BQ_Qai8dx_Qsi4c2_BenchFuncs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs(state);
    const size_t lda = in.k, ldb = in.k, ldd = in.n;

    std::vector<TD> D(in.m * in.n, static_cast<TD>(0));

    struct tqt_qai8dx_qsi4c2_params params;
    params.scale_a = in.scale_a_per_row.data();
    params.zero_point_a = in.zp_a_per_row.data();
    params.scale_b = in.scale_b_per_block.data();
    params.ldsb = in.ldsb;

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(in.m, in.n, in.k, in.bl, in.A.data(), lda, 0, in.B.data(), ldb, 0, nullptr,
                       in.n, D.data(), ldd, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] = ::benchmark::Counter(gemm_flops(in.m, in.n, in.k),
                                                    ::benchmark::Counter::kIsIterationInvariantRate,
                                                    ::benchmark::Counter::OneK::kIs1000);
}

template <typename TD>
void run_w4a8bq_qai8dx_qsi4c2_packed_benchmark(
    ::benchmark::State &state, const W4A8BQ_Qai8dx_Qsi4c2_PackedBenchFuncs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs(state);
    const size_t lda = in.k, ldb = in.k, ldd = in.n;

    const size_t a_packed_size = funcs.get_a_packed_size(in.m, in.k);
    const size_t b_packed_size = funcs.get_b_packed_size(in.n, in.k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_a_pack(in.m, in.k, lda, in.k, 0, in.A.data(), A_packed.data());
    funcs.run_bt_pack(in.n, in.k, ldb, in.k, 0, in.B.data(), B_packed.data());

    std::vector<TD> D(in.m * in.n, static_cast<TD>(0));

    struct tqt_qai8dx_qsi4c2_params params;
    params.scale_a = in.scale_a_per_row.data();
    params.zero_point_a = in.zp_a_per_row.data();
    params.scale_b = in.scale_b_per_block.data();
    params.ldsb = in.ldsb;

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(in.m, in.n, in.k, in.bl, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                       nullptr, in.n, D.data(), ldd, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] = ::benchmark::Counter(gemm_flops(in.m, in.n, in.k),
                                                    ::benchmark::Counter::kIsIterationInvariantRate,
                                                    ::benchmark::Counter::OneK::kIs1000);
}

// ----------------------------------------------------------------------------
// Benchmark runners (Class 2)
// ----------------------------------------------------------------------------

template <typename TD>
void run_w4a8bq_qsi8d2_qsi4c2_benchmark(::benchmark::State &state,
                                        const W4A8BQ_Qsi8d2_Qsi4c2_BenchFuncs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs(state);
    const size_t lda = in.k, ldb = in.k, ldd = in.n;

    std::vector<TD> D(in.m * in.n, static_cast<TD>(0));

    struct tqt_qsi8d2_qsi4c2_params params;
    params.scale_a = in.scale_a_per_block.data();
    params.ldsa = in.ldsa;
    params.scale_b = in.scale_b_per_block.data();
    params.ldsb = in.ldsb;

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(in.m, in.n, in.k, in.bl, in.A.data(), lda, 0, in.B.data(), ldb, 0, nullptr,
                       in.n, D.data(), ldd, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] = ::benchmark::Counter(gemm_flops(in.m, in.n, in.k),
                                                    ::benchmark::Counter::kIsIterationInvariantRate,
                                                    ::benchmark::Counter::OneK::kIs1000);
}

template <typename TD>
void run_w4a8bq_qsi8d2_qsi4c2_packed_benchmark(
    ::benchmark::State &state, const W4A8BQ_Qsi8d2_Qsi4c2_PackedBenchFuncs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs(state);
    const size_t lda = in.k, ldb = in.k, ldd = in.n;

    const size_t a_packed_size = funcs.get_a_packed_size(in.m, in.k);
    const size_t b_packed_size = funcs.get_b_packed_size(in.n, in.k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_a_pack(in.m, in.k, lda, in.k, 0, in.A.data(), A_packed.data());
    funcs.run_bt_pack(in.n, in.k, ldb, in.k, 0, in.B.data(), B_packed.data());

    std::vector<TD> D(in.m * in.n, static_cast<TD>(0));

    struct tqt_qsi8d2_qsi4c2_params params;
    params.scale_a = in.scale_a_per_block.data();
    params.ldsa = in.ldsa;
    params.scale_b = in.scale_b_per_block.data();
    params.ldsb = in.ldsb;

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(in.m, in.n, in.k, in.bl, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                       nullptr, in.n, D.data(), ldd, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] = ::benchmark::Counter(gemm_flops(in.m, in.n, in.k),
                                                    ::benchmark::Counter::kIsIterationInvariantRate,
                                                    ::benchmark::Counter::OneK::kIs1000);
}

// ----------------------------------------------------------------------------
// Benchmark runners (Class 3)
// ----------------------------------------------------------------------------

template <typename TD>
void run_w4a8bq_qsi8d2_qai4c2_benchmark(::benchmark::State &state,
                                        const W4A8BQ_Qsi8d2_Qai4c2_BenchFuncs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs(state);
    const size_t lda = in.k, ldb = in.k, ldd = in.n;

    std::vector<TD> D(in.m * in.n, static_cast<TD>(0));

    struct tqt_qsi8d2_qai4c2_params params;
    params.scale_a = in.scale_a_per_block.data();
    params.ldsa = in.ldsa;
    params.scale_b = in.scale_b_per_block.data();
    params.ldsb = in.ldsb;
    params.zp_b = in.zp_b_per_block.data();
    params.ldzpb = in.ldzpb;

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(in.m, in.n, in.k, in.bl, in.A.data(), lda, 0, in.B.data(), ldb, 0, nullptr,
                       in.n, D.data(), ldd, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] = ::benchmark::Counter(gemm_flops(in.m, in.n, in.k),
                                                    ::benchmark::Counter::kIsIterationInvariantRate,
                                                    ::benchmark::Counter::OneK::kIs1000);
}

template <typename TD>
void run_w4a8bq_qsi8d2_qai4c2_packed_benchmark(
    ::benchmark::State &state, const W4A8BQ_Qsi8d2_Qai4c2_PackedBenchFuncs<TD> &funcs)
{
    auto in = make_w4a8_bq_inputs(state);
    const size_t lda = in.k, ldb = in.k, ldd = in.n;

    const size_t a_packed_size = funcs.get_a_packed_size(in.m, in.k);
    const size_t b_packed_size = funcs.get_b_packed_size(in.n, in.k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);
    funcs.run_a_pack(in.m, in.k, lda, in.k, 0, in.A.data(), A_packed.data());
    funcs.run_bt_pack(in.n, in.k, ldb, in.k, 0, in.B.data(), B_packed.data());

    std::vector<TD> D(in.m * in.n, static_cast<TD>(0));

    struct tqt_qsi8d2_qai4c2_params params;
    params.scale_a = in.scale_a_per_block.data();
    params.ldsa = in.ldsa;
    params.scale_b = in.scale_b_per_block.data();
    params.ldsb = in.ldsb;
    params.zp_b = in.zp_b_per_block.data();
    params.ldzpb = in.ldzpb;

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(in.m, in.n, in.k, in.bl, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                       nullptr, in.n, D.data(), ldd, nullptr, clamp_min, clamp_max, &params);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] = ::benchmark::Counter(gemm_flops(in.m, in.n, in.k),
                                                    ::benchmark::Counter::kIsIterationInvariantRate,
                                                    ::benchmark::Counter::OneK::kIs1000);
}

// ----------------------------------------------------------------------------
// Registration helpers (one per class × pack mode), all adding {m, n, k, bl}
// args with bl defaulting to 32 (a typical LLM group size).
// ----------------------------------------------------------------------------

template <typename TD>
inline void register_w4a8bq_qai8dx_qsi4c2(const char *name,
                                          const W4A8BQ_Qai8dx_Qsi4c2_BenchFuncs<TD> &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
        run_w4a8bq_qai8dx_qsi4c2_benchmark<TD>(state, funcs);
    });
    const auto &shape = global_gemm_shape();
    bm->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
              static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

template <typename TD>
inline void register_w4a8bq_qai8dx_qsi4c2_packed(
    const char *name, const W4A8BQ_Qai8dx_Qsi4c2_PackedBenchFuncs<TD> &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
        run_w4a8bq_qai8dx_qsi4c2_packed_benchmark<TD>(state, funcs);
    });
    const auto &shape = global_gemm_shape();
    bm->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
              static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

template <typename TD>
inline void register_w4a8bq_qsi8d2_qsi4c2(const char *name,
                                          const W4A8BQ_Qsi8d2_Qsi4c2_BenchFuncs<TD> &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
        run_w4a8bq_qsi8d2_qsi4c2_benchmark<TD>(state, funcs);
    });
    const auto &shape = global_gemm_shape();
    bm->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
              static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

template <typename TD>
inline void register_w4a8bq_qsi8d2_qsi4c2_packed(
    const char *name, const W4A8BQ_Qsi8d2_Qsi4c2_PackedBenchFuncs<TD> &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
        run_w4a8bq_qsi8d2_qsi4c2_packed_benchmark<TD>(state, funcs);
    });
    const auto &shape = global_gemm_shape();
    bm->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
              static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

template <typename TD>
inline void register_w4a8bq_qsi8d2_qai4c2(const char *name,
                                          const W4A8BQ_Qsi8d2_Qai4c2_BenchFuncs<TD> &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
        run_w4a8bq_qsi8d2_qai4c2_benchmark<TD>(state, funcs);
    });
    const auto &shape = global_gemm_shape();
    bm->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
              static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

template <typename TD>
inline void register_w4a8bq_qsi8d2_qai4c2_packed(
    const char *name, const W4A8BQ_Qsi8d2_Qai4c2_PackedBenchFuncs<TD> &funcs)
{
    auto *bm = ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
        run_w4a8bq_qsi8d2_qai4c2_packed_benchmark<TD>(state, funcs);
    });
    const auto &shape = global_gemm_shape();
    bm->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
              static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

}  // namespace

// ----------------------------------------------------------------------------
// Factory macros to keep make_funcs_* boilerplate small.
// ----------------------------------------------------------------------------

#define MAKE_W4A8BQ_UNPACKED_FUNCS(KBASE, TD, FUNCS_TYPE)       \
    static FUNCS_TYPE<TD> make_funcs_##KBASE()                  \
    {                                                           \
        FUNCS_TYPE<TD> f;                                       \
        f.get_m_step = tqt_get_m_step_gemm_##KBASE##_4x1vl_rvv; \
        f.get_n_step = tqt_get_n_step_gemm_##KBASE##_4x1vl_rvv; \
        f.run_gemm = tqt_run_gemm_##KBASE##_4x1vl_rvv;          \
        return f;                                               \
    }

#define MAKE_W4A8BQ_PACKED_FUNCS(KBASE, TD, FUNCS_TYPE)                       \
    static FUNCS_TYPE<TD> make_funcs_##KBASE()                                \
    {                                                                         \
        FUNCS_TYPE<TD> f;                                                     \
        f.get_m_step = tqt_get_m_step_gemm_##KBASE##_4x1vl_rvv;               \
        f.get_n_step = tqt_get_n_step_gemm_##KBASE##_4x1vl_rvv;               \
        f.get_a_packed_size = tqt_get_a_packed_size_gemm_##KBASE##_4x1vl_rvv; \
        f.get_b_packed_size = tqt_get_b_packed_size_gemm_##KBASE##_4x1vl_rvv; \
        f.run_a_pack = tqt_run_a_pack_gemm_##KBASE##_4x1vl_rvv;               \
        f.run_bt_pack = tqt_run_bt_pack_gemm_##KBASE##_4x1vl_rvv;             \
        f.run_gemm = tqt_run_gemm_##KBASE##_4x1vl_rvv;                        \
        return f;                                                             \
    }

// Class 1 (F32 always; F16/BF16 gated)
MAKE_W4A8BQ_UNPACKED_FUNCS(1xnbias_clamp_f32_qai8dx_qsi4c2t, float, W4A8BQ_Qai8dx_Qsi4c2_BenchFuncs)
MAKE_W4A8BQ_PACKED_FUNCS(1xnbias_clamp_f32_qai8dxp_qsi4c2p, float,
                         W4A8BQ_Qai8dx_Qsi4c2_PackedBenchFuncs)
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
MAKE_W4A8BQ_UNPACKED_FUNCS(1xnbias_clamp_f16_qai8dx_qsi4c2t, float16_t,
                           W4A8BQ_Qai8dx_Qsi4c2_BenchFuncs)
MAKE_W4A8BQ_PACKED_FUNCS(1xnbias_clamp_f16_qai8dxp_qsi4c2p, float16_t,
                         W4A8BQ_Qai8dx_Qsi4c2_PackedBenchFuncs)
#endif  // __riscv_zfh && __riscv_zvfh
#if defined(__riscv_zvfbfwma)
MAKE_W4A8BQ_UNPACKED_FUNCS(1xnbias_clamp_bf16_qai8dx_qsi4c2t, bfloat16_t,
                           W4A8BQ_Qai8dx_Qsi4c2_BenchFuncs)
MAKE_W4A8BQ_PACKED_FUNCS(1xnbias_clamp_bf16_qai8dxp_qsi4c2p, bfloat16_t,
                         W4A8BQ_Qai8dx_Qsi4c2_PackedBenchFuncs)
#endif  // __riscv_zvfbfwma

// Class 2
MAKE_W4A8BQ_UNPACKED_FUNCS(1xnbias_clamp_f32_qsi8d2_qsi4c2t, float, W4A8BQ_Qsi8d2_Qsi4c2_BenchFuncs)
MAKE_W4A8BQ_PACKED_FUNCS(1xnbias_clamp_f32_qsi8d2p_qsi4c2p, float,
                         W4A8BQ_Qsi8d2_Qsi4c2_PackedBenchFuncs)
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
MAKE_W4A8BQ_UNPACKED_FUNCS(1xnbias_clamp_f16_qsi8d2_qsi4c2t, float16_t,
                           W4A8BQ_Qsi8d2_Qsi4c2_BenchFuncs)
MAKE_W4A8BQ_PACKED_FUNCS(1xnbias_clamp_f16_qsi8d2p_qsi4c2p, float16_t,
                         W4A8BQ_Qsi8d2_Qsi4c2_PackedBenchFuncs)
#endif  // __riscv_zfh && __riscv_zvfh
#if defined(__riscv_zvfbfwma)
MAKE_W4A8BQ_UNPACKED_FUNCS(1xnbias_clamp_bf16_qsi8d2_qsi4c2t, bfloat16_t,
                           W4A8BQ_Qsi8d2_Qsi4c2_BenchFuncs)
MAKE_W4A8BQ_PACKED_FUNCS(1xnbias_clamp_bf16_qsi8d2p_qsi4c2p, bfloat16_t,
                         W4A8BQ_Qsi8d2_Qsi4c2_PackedBenchFuncs)
#endif  // __riscv_zvfbfwma

// Class 3
MAKE_W4A8BQ_UNPACKED_FUNCS(1xnbias_clamp_f32_qsi8d2_qai4c2t, float, W4A8BQ_Qsi8d2_Qai4c2_BenchFuncs)
MAKE_W4A8BQ_PACKED_FUNCS(1xnbias_clamp_f32_qsi8d2p_qai4c2p, float,
                         W4A8BQ_Qsi8d2_Qai4c2_PackedBenchFuncs)
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
MAKE_W4A8BQ_UNPACKED_FUNCS(1xnbias_clamp_f16_qsi8d2_qai4c2t, float16_t,
                           W4A8BQ_Qsi8d2_Qai4c2_BenchFuncs)
MAKE_W4A8BQ_PACKED_FUNCS(1xnbias_clamp_f16_qsi8d2p_qai4c2p, float16_t,
                         W4A8BQ_Qsi8d2_Qai4c2_PackedBenchFuncs)
#endif  // __riscv_zfh && __riscv_zvfh
#if defined(__riscv_zvfbfwma)
MAKE_W4A8BQ_UNPACKED_FUNCS(1xnbias_clamp_bf16_qsi8d2_qai4c2t, bfloat16_t,
                           W4A8BQ_Qsi8d2_Qai4c2_BenchFuncs)
MAKE_W4A8BQ_PACKED_FUNCS(1xnbias_clamp_bf16_qsi8d2p_qai4c2p, bfloat16_t,
                         W4A8BQ_Qsi8d2_Qai4c2_PackedBenchFuncs)
#endif  // __riscv_zvfbfwma

#undef MAKE_W4A8BQ_UNPACKED_FUNCS
#undef MAKE_W4A8BQ_PACKED_FUNCS

// ----------------------------------------------------------------------------
// Public registration entry point (called from register_gemm_i8_i4_rvv_benchmarks).
// ----------------------------------------------------------------------------

void register_gemm_i8_i4_blockquant_rvv_benchmarks()
{
    const PackedQsi8BlockQuantizedGemmBenchmarkFuncs<tqt_qsi8d32_qsi4c32_params> q4_0 = {
        32,
        32,
        Qsi8BlockWeightEncoding::Qau4,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv,
        make_qsi4c32_rvv_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv};
    register_packed_qsi8_block_quantized_gemm_benchmark(
        "tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi4c32p_4x1vl_rvv", q4_0);

    // Class 1: qai8dx × qsi4c2
    register_w4a8bq_qai8dx_qsi4c2<float>("gemm_1xnbias_clamp_f32_qai8dx_qsi4c2t_4x1vl_rvv",
                                         make_funcs_1xnbias_clamp_f32_qai8dx_qsi4c2t());
    register_w4a8bq_qai8dx_qsi4c2_packed<float>("gemm_1xnbias_clamp_f32_qai8dxp_qsi4c2p_4x1vl_rvv",
                                                make_funcs_1xnbias_clamp_f32_qai8dxp_qsi4c2p());
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
    register_w4a8bq_qai8dx_qsi4c2<float16_t>("gemm_1xnbias_clamp_f16_qai8dx_qsi4c2t_4x1vl_rvv",
                                             make_funcs_1xnbias_clamp_f16_qai8dx_qsi4c2t());
    register_w4a8bq_qai8dx_qsi4c2_packed<float16_t>(
        "gemm_1xnbias_clamp_f16_qai8dxp_qsi4c2p_4x1vl_rvv",
        make_funcs_1xnbias_clamp_f16_qai8dxp_qsi4c2p());
#endif  // __riscv_zfh && __riscv_zvfh
#if defined(__riscv_zvfbfwma)
    register_w4a8bq_qai8dx_qsi4c2<bfloat16_t>("gemm_1xnbias_clamp_bf16_qai8dx_qsi4c2t_4x1vl_rvv",
                                              make_funcs_1xnbias_clamp_bf16_qai8dx_qsi4c2t());
    register_w4a8bq_qai8dx_qsi4c2_packed<bfloat16_t>(
        "gemm_1xnbias_clamp_bf16_qai8dxp_qsi4c2p_4x1vl_rvv",
        make_funcs_1xnbias_clamp_bf16_qai8dxp_qsi4c2p());
#endif  // __riscv_zvfbfwma

    // Class 2: qsi8d2 × qsi4c2
    register_w4a8bq_qsi8d2_qsi4c2<float>("gemm_1xnbias_clamp_f32_qsi8d2_qsi4c2t_4x1vl_rvv",
                                         make_funcs_1xnbias_clamp_f32_qsi8d2_qsi4c2t());
    register_w4a8bq_qsi8d2_qsi4c2_packed<float>("gemm_1xnbias_clamp_f32_qsi8d2p_qsi4c2p_4x1vl_rvv",
                                                make_funcs_1xnbias_clamp_f32_qsi8d2p_qsi4c2p());
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
    register_w4a8bq_qsi8d2_qsi4c2<float16_t>("gemm_1xnbias_clamp_f16_qsi8d2_qsi4c2t_4x1vl_rvv",
                                             make_funcs_1xnbias_clamp_f16_qsi8d2_qsi4c2t());
    register_w4a8bq_qsi8d2_qsi4c2_packed<float16_t>(
        "gemm_1xnbias_clamp_f16_qsi8d2p_qsi4c2p_4x1vl_rvv",
        make_funcs_1xnbias_clamp_f16_qsi8d2p_qsi4c2p());
#endif  // __riscv_zfh && __riscv_zvfh
#if defined(__riscv_zvfbfwma)
    register_w4a8bq_qsi8d2_qsi4c2<bfloat16_t>("gemm_1xnbias_clamp_bf16_qsi8d2_qsi4c2t_4x1vl_rvv",
                                              make_funcs_1xnbias_clamp_bf16_qsi8d2_qsi4c2t());
    register_w4a8bq_qsi8d2_qsi4c2_packed<bfloat16_t>(
        "gemm_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p_4x1vl_rvv",
        make_funcs_1xnbias_clamp_bf16_qsi8d2p_qsi4c2p());
#endif  // __riscv_zvfbfwma

    // Class 3: qsi8d2 × qai4c2
    register_w4a8bq_qsi8d2_qai4c2<float>("gemm_1xnbias_clamp_f32_qsi8d2_qai4c2t_4x1vl_rvv",
                                         make_funcs_1xnbias_clamp_f32_qsi8d2_qai4c2t());
    register_w4a8bq_qsi8d2_qai4c2_packed<float>("gemm_1xnbias_clamp_f32_qsi8d2p_qai4c2p_4x1vl_rvv",
                                                make_funcs_1xnbias_clamp_f32_qsi8d2p_qai4c2p());
#if defined(__riscv_zfh) && defined(__riscv_zvfh)
    register_w4a8bq_qsi8d2_qai4c2<float16_t>("gemm_1xnbias_clamp_f16_qsi8d2_qai4c2t_4x1vl_rvv",
                                             make_funcs_1xnbias_clamp_f16_qsi8d2_qai4c2t());
    register_w4a8bq_qsi8d2_qai4c2_packed<float16_t>(
        "gemm_1xnbias_clamp_f16_qsi8d2p_qai4c2p_4x1vl_rvv",
        make_funcs_1xnbias_clamp_f16_qsi8d2p_qai4c2p());
#endif  // __riscv_zfh && __riscv_zvfh
#if defined(__riscv_zvfbfwma)
    register_w4a8bq_qsi8d2_qai4c2<bfloat16_t>("gemm_1xnbias_clamp_bf16_qsi8d2_qai4c2t_4x1vl_rvv",
                                              make_funcs_1xnbias_clamp_bf16_qsi8d2_qai4c2t());
    register_w4a8bq_qsi8d2_qai4c2_packed<bfloat16_t>(
        "gemm_1xnbias_clamp_bf16_qsi8d2p_qai4c2p_4x1vl_rvv",
        make_funcs_1xnbias_clamp_bf16_qsi8d2p_qai4c2p());
#endif  // __riscv_zvfbfwma
}

}  // namespace benchmark
}  // namespace torq_tile
