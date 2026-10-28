//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
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

// Register fixed-shape kernels only when the global shape satisfies their contract.
inline bool matches_specialized_dwconv_shape(size_t kh, size_t kw, size_t stride_h, size_t stride_w)
{
    const auto &s = global_conv_shape();
    return s.ic > 0 && s.kh == kh && s.kw == kw && s.stride_h == stride_h &&
           s.stride_w == stride_w && s.ih + 2 * s.pad_h >= kh && s.iw + 2 * s.pad_w >= kw;
}

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
                                                  const SpecPackedDwconvBenchFuncs<ClampT> &funcs,
                                                  size_t kh, size_t kw, size_t stride_h,
                                                  size_t stride_w)
{
    if (!matches_specialized_dwconv_shape(kh, kw, stride_h, stride_w))
        return;
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_spec_packed_dwconv_benchmark<T, ClampT>(state, funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.ic),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

// ============================================================================
// Function pointer types for specialized dwconv with weight-only packing
//   input/output raw NHWC (cl), only weight packed; oh_step-sliced
// ============================================================================

template <typename ClampT>
struct SpecWeightPackedDwconvBenchFuncs
{
    size_t (*get_input_size)(const tqt_conv_params *params);
    size_t (*get_output_size)(const tqt_conv_params *params);
    size_t (*get_packed_weight_size)(const tqt_conv_params *params);
    size_t (*get_oh_step)();
    void (*run_pack_weight)(const void *src, void *dst, const tqt_conv_params *params);
    void (*run_dwconv)(size_t oh_start, size_t oh_end, const void *input, const void *packed_weight,
                       const void *bias, void *output, const tqt_conv_params *params,
                       ClampT clamp_min, ClampT clamp_max);
};

template <typename T, typename ClampT>
inline void run_spec_weight_packed_dwconv_benchmark(
    ::benchmark::State &state, const SpecWeightPackedDwconvBenchFuncs<ClampT> &funcs)
{
    const auto &cs = global_conv_shape();
    tqt_conv_params p = build_dwconv_params(cs);
    const size_t IC = p.input_channel, OC = p.output_channel;

    std::vector<T> input(IC * p.input_shape[1] * p.input_shape[2], static_cast<T>(1.0f));
    std::vector<T> weight(OC * p.kernel_shape[1] * p.kernel_shape[2], static_cast<T>(0.5f));
    std::vector<T> bias(OC, static_cast<T>(0.0f));
    std::vector<T> output(OC * p.output_shape[1] * p.output_shape[2]);

    std::vector<uint8_t> packed_wt(funcs.get_packed_weight_size(&p));
    funcs.run_pack_weight(weight.data(), packed_wt.data(), &p);

    const size_t oh_step = funcs.get_oh_step();
    const size_t OH = p.output_shape[1];
    const ClampT clamp_min = static_cast<ClampT>(-FLT_MAX);
    const ClampT clamp_max = static_cast<ClampT>(FLT_MAX);

    for (auto _ : state) {
        for (size_t oh = 0; oh < OH; oh += oh_step) {
            const size_t oh_end = (oh + oh_step < OH) ? (oh + oh_step) : OH;
            funcs.run_dwconv(oh, oh_end, input.data(), packed_wt.data(), bias.data(), output.data(),
                             &p, clamp_min, clamp_max);
        }
        ::benchmark::DoNotOptimize(output.data());
    }

    state.counters["GFLOPS"] =
        ::benchmark::Counter(dwconv_flops(cs), ::benchmark::Counter::kIsIterationInvariantRate,
                             ::benchmark::Counter::OneK::kIs1000);
}

template <typename T, typename ClampT>
inline void register_spec_weight_packed_dwconv_benchmark(
    const char *name, const SpecWeightPackedDwconvBenchFuncs<ClampT> &funcs, size_t kh, size_t kw,
    size_t stride_h, size_t stride_w)
{
    if (!matches_specialized_dwconv_shape(kh, kw, stride_h, stride_w))
        return;
    const auto &cs = global_conv_shape();
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_spec_weight_packed_dwconv_benchmark<T, ClampT>(state,
                                                                                          funcs);
                                   })
        ->Args({static_cast<int64_t>(cs.ic), static_cast<int64_t>(cs.ic),
                static_cast<int64_t>(cs.ih), static_cast<int64_t>(cs.kh)})
        ->ArgNames({"IC", "OC", "HW", "K"});
}

template <typename T>
void register_nonpacked_specialized_dwconv(const char *name,
                                           const NonPackedDwconvBenchFuncs<T> &funcs, bool one_d)
{
    const auto &s = global_conv_shape();
    if (!matches_specialized_dwconv_shape(one_d ? 1 : 3, one_d ? 4 : 3, 1, 1) ||
        (one_d && (s.ih != 1 || s.pad_h != 0)))
        return;
    ::benchmark::RegisterBenchmark(name,
                                   [funcs](::benchmark::State &state) {
                                       run_nonpacked_dwconv_benchmark<T, T>(state, funcs);
                                       auto shape = global_conv_shape();
                                       shape.oc = shape.ic;
                                       state.counters["GFLOPS"] = ::benchmark::Counter(
                                           dwconv_flops(shape),
                                           ::benchmark::Counter::kIsIterationInvariantRate,
                                           ::benchmark::Counter::OneK::kIs1000);
                                   })
        ->Args({int64_t(s.ic), int64_t(s.ih), int64_t(s.iw), int64_t(s.kh), int64_t(s.kw),
                int64_t(s.pad_h), int64_t(s.pad_w)})
        ->ArgNames({"IC", "IH", "IW", "KH", "KW", "PH", "PW"});
}

// CF input -> pack -> OH-sliced compute -> unpack -> CF output, every iteration.
// Weight packing is outside the timing loop; existing compute-only benchmarks are unchanged.
template <typename T>
void register_packed_full_tensor_dwconv(const char *name, const SpecPackedDwconvBenchFuncs<T> &f,
                                        void (*unpack)(const void *, void *,
                                                       const tqt_conv_params *))
{
    if (!matches_specialized_dwconv_shape(3, 3, 1, 1))
        return;
    const auto &s = global_conv_shape();
    ::benchmark::RegisterBenchmark(
        name,
        [f, unpack](::benchmark::State &state) {
            const tqt_conv_params p = build_dwconv_params(global_conv_shape());
            const size_t ni = p.input_channel * p.input_shape[1] * p.input_shape[2];
            const size_t no = p.output_channel * p.output_shape[1] * p.output_shape[2];
            std::vector<T> input(ni, T(1)), weight(p.output_channel * 9, T(0.5));
            std::vector<T> bias(p.output_channel, T(0)), output(no);
            std::vector<uint8_t> pi(f.get_packed_input_size(&p)), pw(f.get_packed_weight_size(&p));
            std::vector<uint8_t> po(f.get_output_size(&p));
            f.run_pack_weight(weight.data(), pw.data(), &p);
            const size_t step = f.get_oh_step();
            for (auto _ : state) {
                f.run_pack_input(input.data(), pi.data(), &p);
                for (size_t h = 0; h < p.output_shape[1]; h += step) {
                    const size_t end = std::min(h + step, p.output_shape[1]);
                    f.run_dwconv(h, end, pi.data(), pw.data(), bias.data(), po.data(), &p,
                                 T(-INFINITY), T(INFINITY));
                }
                unpack(po.data(), output.data(), &p);
                ::benchmark::DoNotOptimize(output.data());
            }
            auto shape = global_conv_shape();
            shape.oc = shape.ic;
            state.counters["GFLOPS"] = ::benchmark::Counter(
                dwconv_flops(shape), ::benchmark::Counter::kIsIterationInvariantRate,
                ::benchmark::Counter::OneK::kIs1000);
        })
        ->Args({int64_t(s.ic), int64_t(s.ih), int64_t(s.iw)})
        ->ArgNames({"IC", "IH", "IW"});
}

}  // namespace benchmark
}  // namespace torq_tile
