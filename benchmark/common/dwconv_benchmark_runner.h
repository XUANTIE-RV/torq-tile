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
#include "benchmark/common/conv_benchmark_runner.h"  // reuse ConvShape + global_conv_shape
#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

namespace torq_tile
{
namespace benchmark
{

// ============================================================================
// dwconv FLOPS: 2 * OC * OH * OW * KH * KW (each output element does KH*KW MACs)
// ============================================================================

inline double dwconv_flops(const ConvShape &s)
{
    return 2.0 * static_cast<double>(s.oc) * static_cast<double>(s.oh()) *
           static_cast<double>(s.ow()) * static_cast<double>(s.kh) * static_cast<double>(s.kw);
}

// Build dwconv conv_params from a ConvShape, forcing groups = IC and OC = IC (dm = 1).
inline tqt_conv_params build_dwconv_params(const ConvShape &cs)
{
    tqt_conv_params p = cs.to_params();
    p.groups = cs.ic;
    p.output_channel = cs.ic;
    return p;
}

// ============================================================================
// Function pointer types for non-packed dwconv (raw cf or cl input/output)
// ============================================================================

template <typename ClampT>
struct NonPackedDwconvBenchFuncs
{
    void (*run_dwconv)(const void *input, const void *weight, const void *bias, void *output,
                       const tqt_conv_params *params, ClampT clamp_min, ClampT clamp_max);
};

// ============================================================================
// Function pointer types for generic packed dwconv (per-iteration pack/unpack)
// ============================================================================

template <typename ClampT>
struct GenericPackedDwconvBenchFuncs
{
    size_t (*get_packed_input_size)(const tqt_conv_params *params);
    size_t (*get_packed_weight_size)(const tqt_conv_params *params);
    size_t (*get_output_size)(const tqt_conv_params *params);
    void (*run_pack_input)(const void *src, void *dst, const tqt_conv_params *params);
    void (*run_pack_weight)(const void *src, void *dst, const tqt_conv_params *params);
    void (*run_unpack_output)(const void *src, void *dst, const tqt_conv_params *params);
    void (*run_dwconv)(const void *packed_input, const void *packed_weight, const void *bias,
                       void *packed_output, const tqt_conv_params *params, ClampT clamp_min,
                       ClampT clamp_max);
};

// ============================================================================
// Function pointer types for specialized packed dwconv (oh_step-sliced)
// ============================================================================

template <typename ClampT>
struct SpecPackedDwconvBenchFuncs
{
    size_t (*get_packed_input_size)(const tqt_conv_params *params);
    size_t (*get_packed_weight_size)(const tqt_conv_params *params);
    size_t (*get_output_size)(const tqt_conv_params *params);
    size_t (*get_oh_step)();
    void (*run_pack_input)(const void *src, void *dst, const tqt_conv_params *params);
    void (*run_pack_weight)(const void *src, void *dst, const tqt_conv_params *params);
    void (*run_dwconv)(size_t oh_start, size_t oh_end, const void *packed_input,
                       const void *packed_weight, const void *bias, void *packed_output,
                       const tqt_conv_params *params, ClampT clamp_min, ClampT clamp_max);
};

// ============================================================================
// Non-packed dwconv benchmark
// ============================================================================

template <typename T, typename ClampT>
inline void run_nonpacked_dwconv_benchmark(::benchmark::State &state,
                                           const NonPackedDwconvBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params p = build_dwconv_params(cs);
    const size_t IC = p.input_channel, OC = p.output_channel;

    std::vector<T> input(IC * p.input_shape[1] * p.input_shape[2], static_cast<T>(1.0f));
    std::vector<T> weight(OC * p.kernel_shape[1] * p.kernel_shape[2], static_cast<T>(0.5f));
    std::vector<T> bias(OC, static_cast<T>(0.0f));
    std::vector<T> output(OC * p.output_shape[1] * p.output_shape[2]);

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_dwconv(input.data(), weight.data(), bias.data(), output.data(), &p, clamp_min,
                         clamp_max);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(dwconv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Generic packed dwconv benchmark (pack input + unpack output inside the loop;
// weight packed once outside since weights are static at inference time)
// ============================================================================

template <typename T, typename ClampT>
inline void run_generic_packed_dwconv_benchmark(::benchmark::State &state,
                                                const GenericPackedDwconvBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params p = build_dwconv_params(cs);
    const size_t IC = p.input_channel, OC = p.output_channel;

    std::vector<T> input(IC * p.input_shape[1] * p.input_shape[2], static_cast<T>(1.0f));
    std::vector<T> weight(OC * p.kernel_shape[1] * p.kernel_shape[2], static_cast<T>(0.5f));
    std::vector<T> bias(OC, static_cast<T>(0.0f));
    std::vector<T> output(OC * p.output_shape[1] * p.output_shape[2]);

    std::vector<uint8_t> packed_in(funcs.get_packed_input_size(&p));
    std::vector<uint8_t> packed_wt(funcs.get_packed_weight_size(&p));
    std::vector<uint8_t> packed_out(funcs.get_output_size(&p));
    funcs.run_pack_weight(weight.data(), packed_wt.data(), &p);

    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        funcs.run_pack_input(input.data(), packed_in.data(), &p);
        funcs.run_dwconv(packed_in.data(), packed_wt.data(), bias.data(), packed_out.data(), &p,
                         clamp_min, clamp_max);
        funcs.run_unpack_output(packed_out.data(), output.data(), &p);
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(dwconv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Specialized packed dwconv benchmark (compute-only timing; pack/unpack are
// amortized outside the timing loop, mirroring the typical inference path
// where weight is packed once and OH chunks are scheduled per inference)
// ============================================================================

template <typename T, typename ClampT>
inline void run_spec_packed_dwconv_benchmark(::benchmark::State &state,
                                             const SpecPackedDwconvBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params p = build_dwconv_params(cs);
    const size_t IC = p.input_channel, OC = p.output_channel;

    std::vector<T> input(IC * p.input_shape[1] * p.input_shape[2], static_cast<T>(1.0f));
    std::vector<T> weight(OC * p.kernel_shape[1] * p.kernel_shape[2], static_cast<T>(0.5f));
    std::vector<T> bias(OC, static_cast<T>(0.0f));

    std::vector<uint8_t> packed_in(funcs.get_packed_input_size(&p));
    std::vector<uint8_t> packed_wt(funcs.get_packed_weight_size(&p));
    std::vector<uint8_t> packed_out(funcs.get_output_size(&p));
    funcs.run_pack_input(input.data(), packed_in.data(), &p);
    funcs.run_pack_weight(weight.data(), packed_wt.data(), &p);

    const size_t oh_step = funcs.get_oh_step();
    const size_t OH = p.output_shape[1];
    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        for (size_t oh = 0; oh < OH; oh += oh_step) {
            const size_t oh_end = (oh + oh_step < OH) ? (oh + oh_step) : OH;
            funcs.run_dwconv(oh, oh_end, packed_in.data(), packed_wt.data(), bias.data(),
                             packed_out.data(), &p, clamp_min, clamp_max);
        }
        ::benchmark::DoNotOptimize(packed_out.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(dwconv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

// ============================================================================
// Registration helpers
// ============================================================================

template <typename T, typename ClampT>
inline void register_nonpacked_dwconv_benchmark(const char *name,
                                                const NonPackedDwconvBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_dwconv_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.ic),  // OC = IC (dm=1)
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

template <typename T, typename ClampT>
inline void register_generic_packed_dwconv_benchmark(
    const char *name, const GenericPackedDwconvBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_generic_packed_dwconv_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.ic),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

template <typename T, typename ClampT>
inline void register_spec_packed_dwconv_benchmark(const char *name,
                                                  const SpecPackedDwconvBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_spec_packed_dwconv_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.ic),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

}  // namespace benchmark
}  // namespace torq_tile
