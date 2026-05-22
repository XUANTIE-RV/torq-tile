//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <benchmark/benchmark.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "benchmark/common/benchmark_common.h"
#include "src/common/tqt_common.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// Function pointer types for non-packed GEMM variants
// (mirrors test/common/gemm_test_runner.h but without gtest dependency)
// ============================================================================

template <typename ClampT>
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

template <typename ClampT>
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

template <typename InputT, typename OutputT, typename ClampT>
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

template <typename InputT, typename OutputT, typename ClampT>
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
// (InputT: A/B element type, OutputT: D element type; for same-type GEMM use InputT == OutputT)
// ============================================================================

template <typename InputT, typename OutputT, BLayout kBLayout, typename ClampT>
void run_nonpacked_gemm_benchmark(::benchmark::State &state,
                                  const NonPackedGemmFuncs<ClampT> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));

    // Allocate and fill input buffers (A and B use InputT)
    std::vector<InputT> A(m * k);
    fill_random(A.data(), m * k);

    const size_t lda = k;
    size_t ldb;
    std::vector<InputT> B;
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

template <typename InputT, typename OutputT, BLayout kBLayout, typename ClampT>
inline void register_nonpacked_gemm_benchmark(const char *name,
                                              const NonPackedGemmFuncs<ClampT> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_nonpacked_gemm_benchmark<InputT, OutputT, kBLayout, ClampT>(state, funcs);
        });

    const auto &shape = global_gemm_shape();
    benchmark_instance
        ->Args({static_cast<int64_t>(shape.m), static_cast<int64_t>(shape.n),
                static_cast<int64_t>(shape.k)})
        ->ArgNames({"m", "n", "k"});
}

// ============================================================================
// Packed GEMM benchmark function
// (InputT: A/B element type, OutputT: D element type; for same-type GEMM use InputT == OutputT)
// ============================================================================

template <typename InputT, typename OutputT, typename ClampT>
void run_packed_gemm_benchmark(::benchmark::State &state, const PackedGemmFuncs<ClampT> &funcs)
{
    const size_t m = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(state.range(1));
    const size_t k = static_cast<size_t>(state.range(2));

    const size_t lda = k;
    const size_t ldb_orig = k;
    const size_t ldd = n;

    // Allocate and fill input buffers (B in transposed layout for pack input)
    std::vector<InputT> A(m * k);
    std::vector<InputT> B_orig(n * k);
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

template <typename InputT, typename OutputT, typename ClampT>
inline void register_packed_gemm_benchmark(const char *name, const PackedGemmFuncs<ClampT> &funcs)
{
    auto *benchmark_instance =
        ::benchmark::RegisterBenchmark(name, [funcs](::benchmark::State &state) {
            run_packed_gemm_benchmark<InputT, OutputT, ClampT>(state, funcs);
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
                                 void *D, size_t ldd, const void *bias, TD clamp_min, TD clamp_max);

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
                       nullptr, ldd, D.data(), ldd, nullptr, clamp_min, clamp_max);
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
                                 TD clamp_max);

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
                       clamp_max);
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

}  // namespace benchmark
}  // namespace torq_tile
