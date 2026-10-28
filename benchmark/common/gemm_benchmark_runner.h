//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "benchmark/common/benchmark_common.h"
#include "src/common/tqt_common.h"
#include "src/gemm/gemm_f4_f4/tqt_params_f4_f4.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Function pointer types for non-packed GEMM variants
// (mirrors test/common/gemm_test_runner.h but without gtest dependency)
// ============================================================================

template <typename ClampT, typename BiasT = void>
struct NonPackedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
                                 ClampT clamp_min, ClampT clamp_max);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBOffsetFunc get_b_offset;
    GetDOffsetFunc get_d_offset;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Function pointer types for packed GEMM variants
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
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
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
    GetDOffsetFunc get_d_offset;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Function pointer types for B-only-packed GEMM variants
// (A stays in original layout, only B is pre-packed)
// ============================================================================

template <typename ClampT>
struct BOnlyPackedGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda);
    using GetBPackedOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb,
                                            size_t actual_n);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
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
    GetDOffsetFunc get_d_offset;
    GetBPackedSizeFunc get_b_packed_size;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// B-only-packed GEMM benchmark function
// (A stays in original layout, only B is pre-packed)
// (InputT: A/B element type, OutputT: D element type; for same-type GEMM use InputT == OutputT)
// ============================================================================

template <typename InputT, typename OutputT, typename ClampT, typename BiasT = void>
void run_b_only_packed_gemm_benchmark(::benchmark::State &state,
                                      const BOnlyPackedGemmFuncs<ClampT> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));

    const size_t lda = k;
    const size_t ldb_orig = k;
    const size_t ldd = n;

    // Allocate and fill input buffers (A in row-major, B in transposed layout for pack input)
    std::vector<InputT> A(m * k);
    std::vector<InputT> B_orig(n * k);
    fill_random(A.data(), m * k);
    fill_random(B_orig.data(), n * k);

    // Pack only B outside the timing loop
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_bt_pack(n, k, ldb_orig, k, 0, static_cast<const void *>(B_orig.data()),
                      static_cast<void *>(B_packed.data()));

    // Output buffer D uses OutputT
    std::vector<OutputT> D(m * n, static_cast<OutputT>(0));

    const ClampT clamp_min = static_cast<ClampT>(static_cast<OutputT>(-FLT_MAX));
    const ClampT clamp_max = static_cast<ClampT>(static_cast<OutputT>(FLT_MAX));

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, A.data(), lda, 0, B_packed.data(), ldb_orig, 0, nullptr, ldd,
                       D.data(), ldd, nullptr, clamp_min, clamp_max);
        ::benchmark::DoNotOptimize(D.data());
    }

    // Report GFLOPS
    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Helper: register a B-only-packed GEMM benchmark
// ============================================================================

template <typename InputT, typename OutputT, typename ClampT, typename BiasT = void>
inline void register_b_only_packed_gemm_benchmark(const char *name,
                                                  const BOnlyPackedGemmFuncs<ClampT> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_b_only_packed_gemm_benchmark<InputT, OutputT, ClampT>(state, funcs);
        });

    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

// ============================================================================
// Non-packed GEMM benchmark function
// (InputT: A type, WeightT: B type, OutputT: D type; WeightT defaults to InputT)
// ============================================================================

template <typename InputT, typename OutputT, BLayout kBLayout, typename ClampT,
          typename WeightT = InputT, typename BiasT = void>
void run_nonpacked_gemm_benchmark(::benchmark::State &state,
                                  const NonPackedGemmFuncs<ClampT, BiasT> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));

    // Allocate and fill input buffers.
    std::vector<InputT> A(m * k);
    fill_random(A.data(), m * k);

    const size_t lda = k;
    size_t ldb;
    std::vector<WeightT> B;
    if (kBLayout == BLayout::kTransposed) {
        B.resize(n * k);
        fill_random(B.data(), n * k);
        ldb = k;
    } else {
        B.resize(k * n);
        fill_random(B.data(), k * n);
        ldb = n;
    }

    // Output buffer D uses OutputT
    const size_t ldd = n;
    std::vector<OutputT> D(m * n, static_cast<OutputT>(0));

    const ClampT clamp_min = static_cast<ClampT>(static_cast<OutputT>(-FLT_MAX));
    const ClampT clamp_max = static_cast<ClampT>(static_cast<OutputT>(FLT_MAX));

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, A.data(), lda, 0, B.data(), ldb, 0, nullptr, ldd, D.data(), ldd,
                       nullptr, clamp_min, clamp_max);
        ::benchmark::DoNotOptimize(D.data());
    }

    // Report GFLOPS
    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Helper: register a non-packed GEMM benchmark
// ============================================================================

template <typename InputT, typename OutputT, BLayout kBLayout, typename ClampT,
          typename WeightT = InputT, typename BiasT = void>
inline void register_nonpacked_gemm_benchmark(const char *name,
                                              const NonPackedGemmFuncs<ClampT, BiasT> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_nonpacked_gemm_benchmark<InputT, OutputT, kBLayout, ClampT, WeightT>(state, funcs);
        });

    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

// ============================================================================
// Packed GEMM benchmark function
// (InputT: A type, WeightT: B type, OutputT: D type; WeightT defaults to InputT)
// ============================================================================

template <typename InputT, typename OutputT, typename ClampT, typename WeightT = InputT,
          typename BiasT = void>
void run_packed_gemm_benchmark(::benchmark::State &state,
                               const PackedGemmFuncs<ClampT, BiasT> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));

    const size_t lda = k;
    const size_t ldb_orig = k;
    const size_t ldd = n;

    // Allocate and fill input buffers (B in transposed layout for pack input)
    std::vector<InputT> A(m * k);
    std::vector<WeightT> B_orig(n * k);
    fill_random(A.data(), m * k);
    fill_random(B_orig.data(), n * k);

    // Pack A and B outside the timing loop
    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_a_pack(m, k, lda, k, 0, static_cast<const void *>(A.data()),
                     static_cast<void *>(A_packed.data()));
    funcs.run_bt_pack(n, k, ldb_orig, k, 0, static_cast<const void *>(B_orig.data()),
                      static_cast<void *>(B_packed.data()));

    // Output buffer D uses OutputT
    std::vector<OutputT> D(m * n, static_cast<OutputT>(0));

    const ClampT clamp_min = static_cast<ClampT>(static_cast<OutputT>(-FLT_MAX));
    const ClampT clamp_max = static_cast<ClampT>(static_cast<OutputT>(FLT_MAX));

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, A_packed.data(), lda, 0, B_packed.data(), ldb_orig, 0, nullptr, ldd,
                       D.data(), ldd, nullptr, clamp_min, clamp_max);
        ::benchmark::DoNotOptimize(D.data());
    }

    // Report GFLOPS
    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Helper: register a packed GEMM benchmark
// ============================================================================

template <typename InputT, typename OutputT, typename ClampT, typename WeightT = InputT,
          typename BiasT = void>
inline void register_packed_gemm_benchmark(const char *name,
                                           const PackedGemmFuncs<ClampT, BiasT> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_packed_gemm_benchmark<InputT, OutputT, ClampT, WeightT>(state, funcs);
        });

    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

// ============================================================================
// Packed GEMM component benchmarks
// ============================================================================

template <typename InputT, typename ClampT, typename BiasT = void>
void run_gemm_a_pack_benchmark(::benchmark::State &state,
                               const PackedGemmFuncs<ClampT, BiasT> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t k = static_cast<size_t>(state.range(2));
    std::vector<InputT> A(m * k);
    fill_random(A.data(), A.size());
    std::vector<uint8_t> A_packed(funcs.get_a_packed_size(m, k));

    for (auto _ : state) {
        funcs.run_a_pack(m, k, k, k, 0, A.data(), A_packed.data());
        ::benchmark::DoNotOptimize(A_packed.data());
        ::benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * m * k * sizeof(InputT)));
}

template <typename WeightT, typename ClampT, typename BiasT = void>
void run_gemm_b_pack_benchmark(::benchmark::State &state,
                               const PackedGemmFuncs<ClampT, BiasT> &funcs)
{
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));
    std::vector<WeightT> B(n * k);
    fill_random(B.data(), B.size());
    std::vector<uint8_t> B_packed(funcs.get_b_packed_size(n, k));

    for (auto _ : state) {
        funcs.run_bt_pack(n, k, k, k, 0, B.data(), B_packed.data());
        ::benchmark::DoNotOptimize(B_packed.data());
        ::benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * k * sizeof(WeightT)));
}

template <typename InputT, typename OutputT, typename ClampT, typename WeightT = InputT,
          typename BiasT = void>
void run_gemm_a_pack_and_gemm_benchmark(::benchmark::State &state,
                                        const PackedGemmFuncs<ClampT, BiasT> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));
    std::vector<InputT> A(m * k);
    std::vector<WeightT> B(n * k);
    fill_random(A.data(), A.size());
    fill_random(B.data(), B.size());

    std::vector<uint8_t> A_packed(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> B_packed(funcs.get_b_packed_size(n, k));
    funcs.run_bt_pack(n, k, k, k, 0, B.data(), B_packed.data());

    std::vector<OutputT> D(m * n, static_cast<OutputT>(0));
    const ClampT clamp_min = static_cast<ClampT>(static_cast<OutputT>(-FLT_MAX));
    const ClampT clamp_max = static_cast<ClampT>(static_cast<OutputT>(FLT_MAX));

    for (auto _ : state) {
        funcs.run_a_pack(m, k, k, k, 0, A.data(), A_packed.data());
        funcs.run_gemm(m, n, k, A_packed.data(), k, 0, B_packed.data(), k, 0, nullptr, n, D.data(),
                       n, nullptr, clamp_min, clamp_max);
        ::benchmark::DoNotOptimize(D.data());
        ::benchmark::ClobberMemory();
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

template <typename InputT, typename ClampT, typename BiasT = void>
inline void register_gemm_a_pack_benchmark(const char *name,
                                           const PackedGemmFuncs<ClampT, BiasT> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_gemm_a_pack_benchmark<InputT, ClampT>(state, funcs);
        });
    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

template <typename WeightT, typename ClampT, typename BiasT = void>
inline void register_gemm_b_pack_benchmark(const char *name,
                                           const PackedGemmFuncs<ClampT, BiasT> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_gemm_b_pack_benchmark<WeightT, ClampT>(state, funcs);
        });
    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

template <typename InputT, typename OutputT, typename ClampT, typename WeightT = InputT,
          typename BiasT = void>
inline void register_gemm_a_pack_and_gemm_benchmark(const char *name,
                                                    const PackedGemmFuncs<ClampT, BiasT> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_gemm_a_pack_and_gemm_benchmark<InputT, OutputT, ClampT, WeightT>(state, funcs);
        });
    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

// ============================================================================
// Function pointer types for quantized-weight GEMM (e.g., int4 weight + fp16
// activation). Templated on activation type TA and output/clamp type TD so
// that both same-precision (TA=TD, e.g. fp16->fp16) and mixed-precision
// (TA=fp16 -> TD=fp32) kernels share one struct.
// ============================================================================

template <typename TA, typename TD>
struct QuantizedWeightGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda, size_t bl);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb, size_t bl);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A, size_t lda,
                                 size_t k_idx_a, const void *B, size_t ldb, size_t k_idx_b,
                                 const void *scale_b, size_t ldsb, const void *C, size_t ldc,
                                 void *D, size_t ldd, const void *bias, TD clamp_min, TD clamp_max,
                                 const TA *lut);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBOffsetFunc get_b_offset;
    GetDOffsetFunc get_d_offset;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Quantized-weight GEMM benchmark runner
// ============================================================================

template <typename TA, typename TD>
void run_quantized_weight_gemm_benchmark(::benchmark::State &state,
                                         const QuantizedWeightGemmFuncs<TA, TD> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));
    const size_t bl = static_cast<size_t>(state.range(3));
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldsb = k / bl;
    const size_t ldd = n;

    std::vector<TA> A(m * k);
    fill_random(A.data(), m * k);

    std::vector<uint8_t> B(n * (k / 2));
    std::srand(123);
    for (size_t i = 0; i < B.size(); i++) {
        B[i] = static_cast<uint8_t>(std::rand() % 256);
    }

    std::vector<TA> scale_b(n * ldsb);
    fill_random(scale_b.data(), n * ldsb);

    std::vector<TD> D(m * n, static_cast<TD>(0));

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, bl, A.data(), lda, 0, B.data(), ldb, 0, scale_b.data(), ldsb,
                       nullptr, ldd, D.data(), ldd, nullptr, clamp_min, clamp_max, nullptr);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Register helper for quantized-weight GEMM benchmark
// ============================================================================

template <typename TA, typename TD>
inline void register_quantized_weight_gemm_benchmark(const char *name,
                                                     const QuantizedWeightGemmFuncs<TA, TD> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_quantized_weight_gemm_benchmark<TA, TD>(state, funcs);
        });

    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

// ============================================================================
// Function pointer types for packed quantized-weight GEMM variants
// (W4A16 packed: A pre-packed, B pre-packed int4)
// ============================================================================

template <typename TA, typename TD>
struct PackedQuantizedWeightGemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetAPackedSizeFunc = size_t (*)(size_t m, size_t k);
    using GetBPackedSizeFunc = size_t (*)(size_t n, size_t k);
    using RunAPackFunc = void (*)(size_t m, size_t k, size_t lda, size_t lda_packed, size_t k_idx,
                                  const void *A, void *A_packed);
    using RunBtPackFunc = void (*)(size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx,
                                   const void *B, void *B_packed);
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl, const void *A_packed,
                                 size_t lda, size_t k_idx_a, const void *B_packed, size_t ldb,
                                 size_t k_idx_b, const void *scale_b, size_t ldsb, const void *C,
                                 size_t ldc, void *D, size_t ldd, const void *bias, TD clamp_min,
                                 TD clamp_max, const TA *lut);

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetDOffsetFunc get_d_offset;
    GetAPackedSizeFunc get_a_packed_size;
    GetBPackedSizeFunc get_b_packed_size;
    RunAPackFunc run_a_pack;
    RunBtPackFunc run_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Packed quantized-weight GEMM benchmark runner
// ============================================================================

template <typename TA, typename TD>
void run_packed_quantized_weight_gemm_benchmark(::benchmark::State &state,
                                                const PackedQuantizedWeightGemmFuncs<TA, TD> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));
    const size_t bl = static_cast<size_t>(state.range(3));
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldsb = k / bl;
    const size_t ldd = n;

    // Allocate and fill A
    std::vector<TA> A(m * k);
    fill_random(A.data(), m * k);

    // Allocate and fill B (random uint8 representing packed int4 pairs)
    std::vector<uint8_t> B(n * (k / 2));
    std::srand(123);
    for (size_t i = 0; i < B.size(); i++) {
        B[i] = static_cast<uint8_t>(std::rand() % 256);
    }

    // Scale buffer
    std::vector<TA> scale_b(n * ldsb);
    fill_random(scale_b.data(), n * ldsb);

    // Pack A and B outside the timing loop
    const size_t a_packed_size = funcs.get_a_packed_size(m, k);
    const size_t b_packed_size = funcs.get_b_packed_size(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    funcs.run_a_pack(m, k, lda, k, 0, static_cast<const void *>(A.data()),
                     static_cast<void *>(A_packed.data()));
    funcs.run_bt_pack(n, k, ldb, k, 0, static_cast<const void *>(B.data()),
                      static_cast<void *>(B_packed.data()));

    std::vector<TD> D(m * n, static_cast<TD>(0));

    const TD clamp_min = static_cast<TD>(-FLT_MAX);
    const TD clamp_max = static_cast<TD>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, bl, A_packed.data(), lda, 0, B_packed.data(), ldb, 0,
                       scale_b.data(), ldsb, nullptr, ldd, D.data(), ldd, nullptr, clamp_min,
                       clamp_max, nullptr);
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Register helper for packed quantized-weight GEMM benchmark
// ============================================================================

template <typename TA, typename TD>
inline void register_packed_quantized_weight_gemm_benchmark(
    const char *name, const PackedQuantizedWeightGemmFuncs<TA, TD> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_packed_quantized_weight_gemm_benchmark<TA, TD>(state, funcs);
        });

    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

// ============================================================================
// Packed block-quantized byte-weight GEMM benchmark runner
// ============================================================================

template <typename InputType, typename OutputType, typename QParamType, typename QParamsPack,
          typename RunGemm>
void run_packed_block_quantized_byte_gemm_benchmark(
    ::benchmark::State &state, size_t block_size, bool asymmetric,
    size_t (*get_a_packed_size)(size_t, size_t), size_t (*get_b_packed_size)(size_t, size_t),
    size_t (*get_qparams_packed_size)(size_t, size_t),
    void (*run_a_pack)(size_t, size_t, size_t, size_t, size_t, const void *, void *),
    void (*run_bt_pack)(size_t, size_t, size_t, size_t, size_t, const void *, void *),
    QParamsPack run_qparams_pack, RunGemm run_gemm)
{
    const size_t m = (size_t)state.range(0);
    const size_t n = (size_t)state.range(1);
    const size_t k = (size_t)state.range(2);
    const size_t blocks = (k + block_size - 1) / block_size;
    std::vector<InputType> A(m * k, (InputType)0.125f);
    std::vector<int8_t> B(n * k);
    std::vector<QParamType> scale(n * blocks, (QParamType)0.03125f);
    std::vector<QParamType> real_zero(n * blocks, (QParamType)0.25f);
    for (size_t i = 0; i < B.size(); ++i) {
        if (asymmetric) {
            B[i] = (int8_t)((i * 5 + 3) & 31);
        } else if (block_size == 16) {
            B[i] = (int8_t)((int)(i * 5 + 3) % 64 - 32);
        } else {
            B[i] = (int8_t)((int)(i * 7 + 5) % 256 - 128);
        }
    }

    std::vector<uint8_t> A_packed(get_a_packed_size(m, k));
    std::vector<uint8_t> B_packed(get_b_packed_size(n, k));
    std::vector<uint8_t> qparams_packed(get_qparams_packed_size(n, k));
    std::vector<OutputType> D(m * n);
    run_a_pack(m, k, k, k, 0, A.data(), A_packed.data());
    run_bt_pack(n, k, k, k, 0, B.data(), B_packed.data());
    run_qparams_pack(n, k, blocks, k, 0, scale.data(), asymmetric ? real_zero.data() : nullptr,
                     qparams_packed.data());

    for (auto _ : state) {
        run_gemm(m, n, k, A_packed.data(), B_packed.data(), D.data(), qparams_packed.data());
        ::benchmark::DoNotOptimize(D.data());
    }
    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed QSI8 block-quantized GEMM benchmark runner
// ============================================================================

enum class Qsi8BlockWeightEncoding
{
    Qsb1,
    Qau2,
    Qau4,
    Qau5,
    Qau8,
    Qsi6,
    Qsi8,
};

inline size_t get_qsi8_block_weight_row_bytes(Qsi8BlockWeightEncoding encoding, size_t k)
{
    switch (encoding) {
        case Qsi8BlockWeightEncoding::Qsb1:
            return k / 8;
        case Qsi8BlockWeightEncoding::Qau2:
            return k / 4;
        case Qsi8BlockWeightEncoding::Qau4:
            return k / 2;
        case Qsi8BlockWeightEncoding::Qau5:
            return (k / 32) * 20;
        case Qsi8BlockWeightEncoding::Qsi6:
            return (k / 16) * 12;
        case Qsi8BlockWeightEncoding::Qau8:
        case Qsi8BlockWeightEncoding::Qsi8:
            return k;
    }
    return 0;
}

inline std::vector<uint8_t> make_qsi8_block_weight_benchmark_source(
    size_t n, size_t k, Qsi8BlockWeightEncoding encoding)
{
    const size_t row_bytes = get_qsi8_block_weight_row_bytes(encoding, k);
    std::vector<uint8_t> source(n * row_bytes, 1);
    if (encoding == Qsi8BlockWeightEncoding::Qsb1) {
        std::fill(source.begin(), source.end(), 0xff);
    } else if (encoding == Qsi8BlockWeightEncoding::Qau2) {
        std::fill(source.begin(), source.end(), 0xaa);
    } else if (encoding == Qsi8BlockWeightEncoding::Qau4) {
        std::fill(source.begin(), source.end(), 0x11);
    } else if (encoding == Qsi8BlockWeightEncoding::Qau5) {
        for (size_t offset = 0; offset < source.size(); offset += 20) {
            std::fill(source.begin() + offset, source.begin() + offset + 16, 0x11);
            std::fill(source.begin() + offset + 16, source.begin() + offset + 20, 0x00);
        }
    } else if (encoding == Qsi8BlockWeightEncoding::Qsi6) {
        for (size_t offset = 0; offset < source.size(); offset += 12) {
            std::fill(source.begin() + offset, source.begin() + offset + 8, 0x11);
            std::fill(source.begin() + offset + 8, source.begin() + offset + 12, 0xaa);
        }
    }
    return source;
}

template <typename QParams>
struct PackedQsi8BlockQuantizedGemmBenchmarkFuncs
{
    using GetPackedSizeFunc = size_t (*)(size_t, size_t);
    using RunPackFunc = void (*)(size_t, size_t, size_t, size_t, size_t, const void *, void *);
    using MakeParamsFunc = QParams (*)(const float *, size_t, const int16_t *, size_t,
                                       const float *, const float *, size_t);
    using RunGemmFunc = void (*)(size_t, size_t, size_t, const void *, size_t, size_t, const void *,
                                 size_t, size_t, const void *, size_t, void *, size_t, const void *,
                                 float, float, const QParams *);

    size_t activation_block_size;
    size_t weight_block_size;
    Qsi8BlockWeightEncoding weight_encoding;
    GetPackedSizeFunc get_a_packed_size;
    GetPackedSizeFunc get_b_packed_size;
    RunPackFunc run_a_pack;
    RunPackFunc run_bt_pack;
    MakeParamsFunc make_params;
    RunGemmFunc run_gemm;
};

template <typename QParams>
void run_packed_qsi8_block_quantized_gemm_benchmark(
    ::benchmark::State &state, const PackedQsi8BlockQuantizedGemmBenchmarkFuncs<QParams> &funcs)
{
    const size_t m = (size_t)state.range(0);
    const size_t n = (size_t)state.range(1);
    const size_t k = (size_t)state.range(2);
    std::vector<int8_t> a(m * k, 1);
    const std::vector<uint8_t> b_source =
        make_qsi8_block_weight_benchmark_source(n, k, funcs.weight_encoding);
    std::vector<uint8_t> a_packed(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> b_packed(funcs.get_b_packed_size(n, k));
    std::vector<float> scale_a(m * (k / funcs.activation_block_size), 0.01f);
    std::vector<int16_t> bsums_a(m * (k / 16), 16);
    std::vector<float> scale_b(n * (k / funcs.weight_block_size), 0.01f);
    std::vector<float> real_zero_b(n * (k / funcs.weight_block_size), 0.0f);
    std::vector<float> d(m * n);

    funcs.run_a_pack(m, k, k, k, 0, a.data(), a_packed.data());
    funcs.run_bt_pack(n, k, k, k, 0, b_source.data(), b_packed.data());
    const QParams params =
        funcs.make_params(scale_a.data(), k / funcs.activation_block_size, bsums_a.data(), k / 16,
                          scale_b.data(), real_zero_b.data(), k / funcs.weight_block_size);

    for (auto _ : state) {
        funcs.run_gemm(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, nullptr, n, d.data(),
                       n, nullptr, -FLT_MAX, FLT_MAX, &params);
        ::benchmark::DoNotOptimize(d.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

template <typename QParams>
void register_packed_qsi8_block_quantized_gemm_benchmark(
    const char *name, const PackedQsi8BlockQuantizedGemmBenchmarkFuncs<QParams> &funcs)
{
    const GemmShape &shape = global_gemm_shape();
    const size_t k =
        std::max(funcs.activation_block_size,
                 ((shape.k + funcs.activation_block_size - 1) / funcs.activation_block_size) *
                     funcs.activation_block_size);
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_packed_qsi8_block_quantized_gemm_benchmark(state, funcs);
                                   })
        ->Args({(int64_t)shape.m, (int64_t)shape.n, (int64_t)k})
        ->ArgNames({"m", "n", "k"});
}

// ============================================================================
// Q2_K/Q3_K-derived factorized-metadata GEMM benchmark runner
// ============================================================================

enum class Qsi8KBlockBenchmarkEncoding
{
    Qau2,
    Qsi3,
    Qau4,
    Qsi4,
};

inline size_t get_qsi8_kblock_benchmark_row_bytes(Qsi8KBlockBenchmarkEncoding encoding, size_t k)
{
    if (encoding == Qsi8KBlockBenchmarkEncoding::Qau2)
        return k / 4;
    if (encoding == Qsi8KBlockBenchmarkEncoding::Qsi3)
        return (k / 16) * 6;
    return k / 2;
}

inline std::vector<uint8_t> make_qsi8_kblock_benchmark_source(size_t n, size_t k,
                                                              Qsi8KBlockBenchmarkEncoding encoding)
{
    const size_t row_bytes = get_qsi8_kblock_benchmark_row_bytes(encoding, k);
    std::vector<uint8_t> source(n * row_bytes, 0);
    if (encoding == Qsi8KBlockBenchmarkEncoding::Qau2) {
        std::fill(source.begin(), source.end(), 0x55);
    } else if (encoding == Qsi8KBlockBenchmarkEncoding::Qsi3) {
        for (size_t offset = 0; offset < source.size(); offset += 6) {
            std::fill(source.begin() + offset, source.begin() + offset + 4, 0x55);
            source[offset + 4] = 0xff;
            source[offset + 5] = 0xff;
        }
    } else {
        std::fill(source.begin(), source.end(), 0x11);
    }
    return source;
}

template <typename QParams>
struct PackedQsi8KBlockGemmBenchmarkFuncs
{
    using GetPackedSizeFunc = size_t (*)(size_t, size_t);
    using RunPackFunc = void (*)(size_t, size_t, size_t, size_t, size_t, const void *, void *);
    using MakeParamsFunc = QParams (*)(const float *, size_t, const int16_t *, size_t,
                                       const int8_t *, const uint8_t *, size_t, const float *,
                                       const float *, size_t);
    using RunGemmFunc = void (*)(size_t, size_t, size_t, const void *, size_t, size_t, const void *,
                                 size_t, size_t, const void *, size_t, void *, size_t, const void *,
                                 float, float, const QParams *);

    Qsi8KBlockBenchmarkEncoding weight_encoding;
    GetPackedSizeFunc get_a_packed_size;
    GetPackedSizeFunc get_b_packed_size;
    RunPackFunc run_a_pack;
    RunPackFunc run_bt_pack;
    MakeParamsFunc make_params;
    RunGemmFunc run_gemm;
};

template <typename QParams>
void run_packed_qsi8_kblock_gemm_benchmark(::benchmark::State &state,
                                           const PackedQsi8KBlockGemmBenchmarkFuncs<QParams> &funcs)
{
    const size_t m = (size_t)state.range(0);
    const size_t n = (size_t)state.range(1);
    const size_t k = (size_t)state.range(2);
    const size_t groups = k / 16;
    const size_t super_blocks = k / 256;
    std::vector<int8_t> a(m * k, 1);
    const std::vector<uint8_t> b_source =
        make_qsi8_kblock_benchmark_source(n, k, funcs.weight_encoding);
    std::vector<uint8_t> a_packed(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> b_packed(funcs.get_b_packed_size(n, k));
    std::vector<float> scale_a(m * super_blocks, 0.01f);
    std::vector<int16_t> bsums_a(m * groups, 16);
    std::vector<int8_t> qscale_b(n * groups, 1);
    std::vector<uint8_t> qmin_b(n * groups, 1);
    std::vector<float> super_scale_b(n * super_blocks, 0.01f);
    std::vector<float> super_min_b(n * super_blocks, 0.0f);
    std::vector<float> d(m * n);

    funcs.run_a_pack(m, k, k, k, 0, a.data(), a_packed.data());
    funcs.run_bt_pack(n, k, k, k, 0, b_source.data(), b_packed.data());
    const QParams params = funcs.make_params(
        scale_a.data(), super_blocks, bsums_a.data(), groups, qscale_b.data(), qmin_b.data(),
        groups, super_scale_b.data(), super_min_b.data(), super_blocks);
    for (auto _ : state) {
        funcs.run_gemm(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, nullptr, n, d.data(),
                       n, nullptr, -FLT_MAX, FLT_MAX, &params);
        ::benchmark::DoNotOptimize(d.data());
    }
    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

template <typename QParams>
void register_packed_qsi8_kblock_gemm_benchmark(
    const char *name, const PackedQsi8KBlockGemmBenchmarkFuncs<QParams> &funcs)
{
    const GemmShape &shape = global_gemm_shape();
    const size_t k = std::max((size_t)256, ((shape.k + 255) / 256) * 256);
    ::benchmark::RegisterBenchmark(
        name,
        [funcs](::benchmark::State &state) { run_packed_qsi8_kblock_gemm_benchmark(state, funcs); })
        ->Args({(int64_t)shape.m, (int64_t)shape.n, (int64_t)k})
        ->ArgNames({"m", "n", "k"});
}

// ============================================================================
// Microscaled fp4 (E2M1) x fp4 (E2M1) GEMM with fp8 (E4M3) block scales.
// Output type TD ∈ {float, bfloat16_t}. bf16 path additionally carries an
// optional 1xN bias and an optional global_scale via params; fp32 path does
// neither. We dispatch on TD with `if constexpr` so a single funcs/runner pair
// covers both.
// ============================================================================

template <typename TD>
struct Mxfp4RunGemmTraits;

template <>
struct Mxfp4RunGemmTraits<float>
{
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl,         //
                                 const void *A, size_t lda, size_t k_idx_a,       //
                                 const void *scale_a, size_t ldsa,                //
                                 const void *B, size_t ldb, size_t k_idx_b,       //
                                 const void *scale_b, size_t ldsb,                //
                                 const void *C, size_t ldc, void *D, size_t ldd,  //
                                 float clamp_min, float clamp_max);
};

template <>
struct Mxfp4RunGemmTraits<bfloat16_t>
{
    using RunGemmFunc = void (*)(size_t m, size_t n, size_t k, size_t bl,         //
                                 const void *A, size_t lda, size_t k_idx_a,       //
                                 const void *scale_a, size_t ldsa,                //
                                 const void *B, size_t ldb, size_t k_idx_b,       //
                                 const void *scale_b, size_t ldsb,                //
                                 const void *C, size_t ldc, void *D, size_t ldd,  //
                                 const void *bias,                                //
                                 bfloat16_t clamp_min, bfloat16_t clamp_max,      //
                                 const struct tqt_rvfp4c16_rvfp4c16_params *params);
};

// ============================================================================
// Function pointer types for non-packed mxfp4 GEMM variants (a_bt layout).
// ============================================================================

template <typename TD>
struct Mxfp4GemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetAOffsetFunc = size_t (*)(size_t m_idx, size_t k_idx, size_t lda, size_t bl);
    using GetBOffsetFunc = size_t (*)(size_t n_idx, size_t k_idx, size_t ldb, size_t bl);
    using GetScaleOffsetFunc = size_t (*)(size_t idx, size_t b_idx, size_t lds);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using RunGemmFunc = typename Mxfp4RunGemmTraits<TD>::RunGemmFunc;

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetAOffsetFunc get_a_offset;
    GetBOffsetFunc get_b_offset;
    GetScaleOffsetFunc get_scale_a_offset;
    GetScaleOffsetFunc get_scale_b_offset;
    GetDOffsetFunc get_d_offset;
    RunGemmFunc run_gemm;
};

// ============================================================================
// Function pointer types for packed mxfp4 GEMM variants (ap_bp layout).
// fp4 data and fp8 scales are packed into independent buffers.
// ============================================================================

template <typename TD>
struct PackedMxfp4GemmFuncs
{
    using GetMStepFunc = size_t (*)(void);
    using GetNStepFunc = size_t (*)(void);
    using GetPackedOffsetFunc = size_t (*)(size_t idx, size_t k_idx, size_t ld, size_t bl);
    using GetScalePackedOffsetFunc = size_t (*)(size_t idx, size_t b_idx, size_t lds);
    using GetDOffsetFunc = size_t (*)(size_t m_idx, size_t n_idx, size_t ldd);
    using GetPackedSizeFunc = size_t (*)(size_t a_or_n, size_t k_or_blocks);
    using RunDataPackFunc = void (*)(size_t a_or_n, size_t k, size_t lda, size_t lda_packed,
                                     size_t k_idx, const void *A, void *A_packed);
    using RunScalePackFunc = void (*)(size_t a_or_n, size_t blocks, size_t ldsa, size_t ldsa_packed,
                                      size_t b_idx, const void *scale, void *scale_packed);
    using RunGemmFunc = typename Mxfp4RunGemmTraits<TD>::RunGemmFunc;

    GetMStepFunc get_m_step;
    GetNStepFunc get_n_step;
    GetPackedOffsetFunc get_a_packed_offset;
    GetPackedOffsetFunc get_b_packed_offset;
    GetPackedSizeFunc get_a_packed_size;
    GetPackedSizeFunc get_b_packed_size;
    GetScalePackedOffsetFunc get_scale_a_packed_offset;
    GetScalePackedOffsetFunc get_scale_b_packed_offset;
    GetPackedSizeFunc get_scale_a_packed_size;
    GetPackedSizeFunc get_scale_b_packed_size;
    GetDOffsetFunc get_d_offset;
    RunDataPackFunc run_a_pack;
    RunDataPackFunc run_bt_pack;
    RunScalePackFunc run_scale_a_pack;
    RunScalePackFunc run_scale_bt_pack;
    RunGemmFunc run_gemm;
};

// ============================================================================
// RVFP4 data and E4M3 scale initializers.
// ============================================================================

inline void fill_mxfp4_data(std::vector<uint8_t> &buf, unsigned int seed)
{
    std::srand(seed);
    for (auto &b : buf) {
        b = static_cast<uint8_t>(std::rand() & 0xFF);
    }
}

inline void fill_mxfp8_scale(std::vector<uint8_t> &buf, unsigned int seed)
{
    // exponent 1..8, mantissa 0..7 (sign bit = 0, no NaN); positive E4M3 range ~ 0.25..16
    std::srand(seed);
    for (auto &b : buf) {
        const uint32_t e = 1u + (std::rand() & 0x7);
        const uint32_t m = std::rand() & 0x7;
        b = static_cast<uint8_t>((e << 3) | m);
    }
}

// ============================================================================
// Non-packed mxfp4 GEMM benchmark runner.
// ============================================================================

template <typename TD>
void run_mxfp4_gemm_benchmark(::benchmark::State &state, const Mxfp4GemmFuncs<TD> &funcs)
{
    (void)funcs;  // silence unused-warning before use in the lambda below
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));
    const size_t bl = static_cast<size_t>(state.range(3));
    const size_t blocks = k / bl;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldsa = blocks;
    const size_t ldsb = blocks;
    const size_t ldd = n;

    std::vector<uint8_t> A((m * k) / 2);
    std::vector<uint8_t> B((n * k) / 2);
    std::vector<uint8_t> scale_a(m * ldsa);
    std::vector<uint8_t> scale_b(n * ldsb);
    fill_mxfp4_data(A, 0xA001);
    fill_mxfp4_data(B, 0xB002);
    fill_mxfp8_scale(scale_a, 0xC003);
    fill_mxfp8_scale(scale_b, 0xD004);

    std::vector<TD> D(m * n, static_cast<TD>(0));

    for (auto _ : state) {
        if constexpr (std::is_same_v<TD, float>) {
            funcs.run_gemm(m, n, k, bl,                             //
                           A.data(), lda, 0, scale_a.data(), ldsa,  //
                           B.data(), ldb, 0, scale_b.data(), ldsb,  //
                           nullptr, ldd, D.data(), ldd, -FLT_MAX, FLT_MAX);
        } else {
            funcs.run_gemm(m, n, k, bl,                             //
                           A.data(), lda, 0, scale_a.data(), ldsa,  //
                           B.data(), ldb, 0, scale_b.data(), ldsb,  //
                           nullptr, ldd, D.data(), ldd, nullptr,    //
                           static_cast<bfloat16_t>(-FLT_MAX), static_cast<bfloat16_t>(FLT_MAX),
                           nullptr);
        }
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Packed mxfp4 GEMM benchmark runner. Packs both fp4 data and fp8 scales
// outside the timing loop so the measured cost is the GEMM itself.
// ============================================================================

template <typename TD>
void run_packed_mxfp4_gemm_benchmark(::benchmark::State &state,
                                     const PackedMxfp4GemmFuncs<TD> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));
    const size_t bl = static_cast<size_t>(state.range(3));
    const size_t blocks = k / bl;
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldsa = blocks;
    const size_t ldsb = blocks;
    const size_t ldd = n;

    std::vector<uint8_t> A((m * k) / 2);
    std::vector<uint8_t> B((n * k) / 2);
    std::vector<uint8_t> scale_a(m * ldsa);
    std::vector<uint8_t> scale_b(n * ldsb);
    fill_mxfp4_data(A, 0xA001);
    fill_mxfp4_data(B, 0xB002);
    fill_mxfp8_scale(scale_a, 0xC003);
    fill_mxfp8_scale(scale_b, 0xD004);

    std::vector<uint8_t> A_packed(funcs.get_a_packed_size(m, k));
    std::vector<uint8_t> B_packed(funcs.get_b_packed_size(n, k));
    std::vector<uint8_t> scale_a_packed(funcs.get_scale_a_packed_size(m, blocks));
    std::vector<uint8_t> scale_b_packed(funcs.get_scale_b_packed_size(n, blocks));

    funcs.run_a_pack(m, k, lda, lda, 0, A.data(), A_packed.data());
    funcs.run_bt_pack(n, k, ldb, ldb, 0, B.data(), B_packed.data());
    funcs.run_scale_a_pack(m, blocks, ldsa, ldsa, 0, scale_a.data(), scale_a_packed.data());
    funcs.run_scale_bt_pack(n, blocks, ldsb, ldsb, 0, scale_b.data(), scale_b_packed.data());

    std::vector<TD> D(m * n, static_cast<TD>(0));

    for (auto _ : state) {
        if constexpr (std::is_same_v<TD, float>) {
            funcs.run_gemm(m, n, k, bl,                                           //
                           A_packed.data(), lda, 0, scale_a_packed.data(), ldsa,  //
                           B_packed.data(), ldb, 0, scale_b_packed.data(), ldsb,  //
                           nullptr, ldd, D.data(), ldd, -FLT_MAX, FLT_MAX);
        } else {
            funcs.run_gemm(m, n, k, bl,                                           //
                           A_packed.data(), lda, 0, scale_a_packed.data(), ldsa,  //
                           B_packed.data(), ldb, 0, scale_b_packed.data(), ldsb,  //
                           nullptr, ldd, D.data(), ldd, nullptr,                  //
                           static_cast<bfloat16_t>(-FLT_MAX), static_cast<bfloat16_t>(FLT_MAX),
                           nullptr);
        }
        ::benchmark::DoNotOptimize(D.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(gemm_flops(m, n, k), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Register helpers for the two mxfp4 runners.
// ============================================================================

template <typename TD>
inline void register_mxfp4_gemm_benchmark(const char *name, const Mxfp4GemmFuncs<TD> &funcs)
{
    auto *benchmark_instance = ::benchmark::RegisterBenchmark(
        name, [funcs](::benchmark::State &state) { run_mxfp4_gemm_benchmark<TD>(state, funcs); });

    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

template <typename TD>
inline void register_packed_mxfp4_gemm_benchmark(const char *name,
                                                 const PackedMxfp4GemmFuncs<TD> &funcs)
{
    auto *benchmark_instance = ::benchmark::RegisterBenchmark(
        name,
        [funcs](::benchmark::State &state) { run_packed_mxfp4_gemm_benchmark<TD>(state, funcs); });

    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k), 32})
        ->ArgNames({"m", "n", "k", "bl"});
}

}  // namespace benchmark
}  // namespace torq_tile
