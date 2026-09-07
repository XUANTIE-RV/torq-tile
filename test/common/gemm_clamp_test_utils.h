//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "src/common/tqt_common.h"
#include "test/common/test_common.h"

namespace torq_tile
{
namespace test
{

// Only dedicated tests select the probe; ordinary tests retain their original function pointer.
inline thread_local bool gemm_clamp_probe_active = false;

struct ClampProbeCoverage
{
    size_t calls = 0;
    size_t split_k_calls = 0;
    std::set<std::tuple<size_t, size_t, size_t, size_t>> shapes;
};

struct ClampProbeShadow
{
    size_t m = 0, n = 0, k_end = 0;
    std::vector<float> values;
};

inline thread_local std::map<std::string, ClampProbeCoverage> clamp_probe_coverage;
inline thread_local std::map<std::pair<std::string, uintptr_t>, ClampProbeShadow>
    clamp_probe_shadows;

template <auto Run>
std::string clamp_probe_name()
{
    const std::string signature = __PRETTY_FUNCTION__;
    const size_t begin = signature.find("Run = ") + 6;
    return signature.substr(begin, signature.find_first_of(";]", begin) - begin);
}

struct ScopedGemmClampProbe
{
    ScopedGemmClampProbe()
    {
        gemm_clamp_probe_active = true;
        clamp_probe_coverage.clear();
        clamp_probe_shadows.clear();
    }
    ~ScopedGemmClampProbe()
    {
        for (const auto &entry : clamp_probe_coverage) {
            std::ostringstream value;
            value << "calls=" << entry.second.calls
                  << ";ranges=31;split_k=" << entry.second.split_k_calls << ";shapes=";
            for (const auto &shape : entry.second.shapes) {
                value << std::get<0>(shape) << ',' << std::get<1>(shape) << ','
                      << std::get<2>(shape) << ',' << std::get<3>(shape) << '|';
            }
            ::testing::Test::RecordProperty("clamp_probe_" + entry.first, value.str());
        }
        gemm_clamp_probe_active = false;
        clamp_probe_shadows.clear();
    }
};

enum class ClampSource
{
    kColumnBias,
    kRowBias,
    kMatrixC,
};

inline void expect_clamp_value(float actual, float expected)
{
    if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual)) << "actual=" << actual;
    } else if (std::isinf(expected)) {
        EXPECT_TRUE(std::isinf(actual)) << "actual=" << actual;
        EXPECT_EQ(std::signbit(actual), std::signbit(expected));
    } else {
        EXPECT_TRUE(std::isfinite(actual)) << "expected=" << expected;
        EXPECT_FLOAT_EQ(actual, expected);
    }
}

// These values are supplied through a documented floating bias/C operand. Quantized
// buffers, their packing, scales and zero points are never changed by the probe.
inline float clamp_probe_input(size_t index)
{
    switch (index % 8) {
        case 0:
            return std::numeric_limits<float>::quiet_NaN();
        case 1:
            return INFINITY;
        case 2:
            return -INFINITY;
        case 4:
            return 20.0f;
        case 5:
            return -20.0f;
        case 6:
            return 1.0f;
        default:
            return 0.0f;
    }
}

template <typename T>
float clamp_type_max()
{
    if constexpr (std::is_same_v<T, float16_t>) {
        return 65504.0f;
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        return 0x1.fep127f;
    } else {
        return std::numeric_limits<float>::max();
    }
}

// Append a small legal dimension matrix to the existing target's parameter generator.
// Preserve the seed's K/block contract, and resolve vector/matrix steps on the runner CPU.
inline auto with_clamp_dimensions(
    const ::testing::internal::ParamGenerator<GemmTestParams> &original, size_t m_step,
    size_t n_step, size_t n_groups, bool rvv, bool aligned_n = false)
{
    std::vector<GemmTestParams> cases;
    for (const auto &item : original) {
        cases.push_back(item);
    }
    GemmTestParams seed = cases.front();
    seed.m_tile = seed.n_tile = seed.k_tile = 0;
    seed.a_chunk = seed.b_chunk = seed.c_chunk = seed.d_chunk = {};
    seed.name = "ClampDimensions";
    std::set<size_t> rows{1, m_step, m_step + 1};
    if (rvv) {
        for (size_t m = 1; m <= m_step; ++m)
            rows.insert(m);
    } else {
        rows.insert(std::max(size_t{1}, m_step / 2));
        rows.insert(m_step / 2 + 1);
    }
    std::set<size_t> columns{n_step};
    if (aligned_n) {
        columns.insert(2 * n_step);
    } else {
        const size_t quantum = std::max(size_t{1}, n_step / n_groups);
        for (size_t group = 0; group < n_groups; ++group)
            columns.insert(group * quantum + 1);
        columns.insert(n_step + 3);
    }
    for (size_t m : rows) {
        for (size_t n : columns) {
            seed.m = m;
            seed.n = n;
            cases.push_back(seed);
        }
    }
    return ::testing::ValuesIn(cases);
}

template <auto Run, typename TD, size_t SourceIndex, size_t DIndex, size_t LddIndex,
          size_t ClampIndex, ClampSource Source, bool NarrowClamp, size_t KIndex, bool HasFloatingC,
          typename Signature>
struct GemmClampProbe;

template <auto Run, typename TD, size_t SourceIndex, size_t DIndex, size_t LddIndex,
          size_t ClampIndex, ClampSource Source, bool NarrowClamp, size_t KIndex, bool HasFloatingC,
          typename... Args>
struct GemmClampProbe<Run, TD, SourceIndex, DIndex, LddIndex, ClampIndex, Source, NarrowClamp,
                      KIndex, HasFloatingC, void (*)(Args...)>
{
    static void run(Args... args)
    {
        auto original = std::make_tuple(args...);
        auto call = original;
        using ClampT = std::tuple_element_t<ClampIndex, decltype(call)>;
        const size_t m = std::get<0>(call);
        const size_t n = std::get<1>(call);
        if (m == 0 || n == 0) {
            std::apply(Run, original);
            return;
        }
        const std::string variant = clamp_probe_name<Run>();
        SCOPED_TRACE(variant);
        auto &coverage = clamp_probe_coverage[variant];
        ++coverage.calls;
        coverage.shapes.emplace(m, n, std::get<2>(original), std::get<KIndex>(original));
        const size_t stride = n + 3;
        const TD sentinel = static_cast<TD>(123.0f);
        std::vector<TD> baseline(m * stride + 8, sentinel);
        std::vector<TD> output(m * stride + 8, sentinel);
        const size_t source_size = Source == ClampSource::kMatrixC   ? m * stride
                                   : Source == ClampSource::kRowBias ? m
                                                                     : n;
        std::vector<TD> injected(source_size, static_cast<TD>(0.0f));
        std::get<SourceIndex>(call) = injected.data();
        if constexpr (Source == ClampSource::kMatrixC) {
            std::get<SourceIndex + 1>(call) = stride;
        }
        std::get<DIndex>(call) = baseline.data();
        std::get<LddIndex>(call) = stride;
        std::get<ClampIndex>(call) = static_cast<ClampT>(-INFINITY);
        std::get<ClampIndex + 1>(call) = static_cast<ClampT>(INFINITY);
        std::apply(Run, call);

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const size_t source_idx = Source == ClampSource::kMatrixC   ? i * stride + j
                                          : Source == ClampSource::kRowBias ? i
                                                                            : j;
                injected[source_idx] = static_cast<TD>(clamp_probe_input(source_idx));
            }
        }
        const float max_finite = clamp_type_max<ClampT>();
        const float bounds[][2] = {{-INFINITY, INFINITY},
                                   {-10.0f, 10.0f},
                                   {-INFINITY, 10.0f},
                                   {-10.0f, INFINITY},
                                   {-max_finite, max_finite}};
        const char *names[] = {"Unbounded", "Finite", "UpperBound", "LowerBound", "MaxFinite"};
        ClampProbeShadow next_shadow{m, n, std::get<KIndex>(original) + std::get<2>(original), {}};
        for (size_t b = 0; b < 5; ++b) {
            SCOPED_TRACE(names[b]);
            SCOPED_TRACE(::testing::Message()
                         << "m=" << m << " n=" << n << " k=" << std::get<2>(call));
            std::fill(output.begin(), output.end(), sentinel);
            std::get<DIndex>(call) = output.data();
            std::get<ClampIndex>(call) = static_cast<ClampT>(bounds[b][0]);
            std::get<ClampIndex + 1>(call) = static_cast<ClampT>(bounds[b][1]);
            std::apply(Run, call);
            if (b == 0) {
                for (const TD value : output) {
                    next_shadow.values.push_back(static_cast<float>(value));
                }
            }
            float lo = static_cast<float>(std::get<ClampIndex>(call));
            float hi = static_cast<float>(std::get<ClampIndex + 1>(call));
            if constexpr (NarrowClamp) {
                // These two W4A16 float-output interfaces clamp in their original FP16 precision.
                lo = static_cast<float>(static_cast<float16_t>(lo));
                hi = static_cast<float>(static_cast<float16_t>(hi));
            }
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    SCOPED_TRACE(::testing::Message() << "row=" << i << " col=" << j);
                    const size_t source_idx = Source == ClampSource::kMatrixC   ? i * stride + j
                                              : Source == ClampSource::kRowBias ? i
                                                                                : j;
                    const float special = clamp_probe_input(source_idx);
                    const float base = static_cast<float>(baseline[i * stride + j]);
                    float expected = std::isfinite(special) ? base + special : special;
                    // All audited floating epilogues apply maximum then minimum. The scalar
                    // TQT_MAX(value, lo) paths also select lo when value is NaN.
                    if (b != 0) {
                        expected = std::fmin(std::fmax(expected, lo), hi);
                    }
                    expected = static_cast<float>(static_cast<TD>(expected));
                    if (std::isfinite(special) && special != 0.0f && std::isfinite(expected)) {
                        // The zero-bias control may have rounded before adding this finite bias.
                        const float epsilon = NarrowClamp || std::is_same_v<TD, float16_t>
                                                  ? 0x1p-10f
                                              : std::is_same_v<TD, bfloat16_t> ? 0x1p-7f
                                                                               : 0x1p-23f;
                        const float tolerance =
                            2.0f * epsilon * std::max({1.0f, std::fabs(base), std::fabs(expected)});
                        EXPECT_TRUE(std::isfinite(static_cast<float>(output[i * stride + j])));
                        EXPECT_NEAR(static_cast<float>(output[i * stride + j]), expected,
                                    tolerance);
                    } else {
                        expect_clamp_value(static_cast<float>(output[i * stride + j]), expected);
                    }
                }
                for (size_t j = n; j < stride; ++j) {
                    EXPECT_EQ(static_cast<float>(output[i * stride + j]), 123.0f);
                }
            }
            for (size_t i = m * stride; i < output.size(); ++i) {
                EXPECT_EQ(static_cast<float>(output[i]), 123.0f);
            }
        }
        if constexpr (HasFloatingC) {
            constexpr size_t c_index = DIndex - 2;
            const auto previous = clamp_probe_shadows.find(
                {variant, reinterpret_cast<uintptr_t>(std::get<c_index>(original))});
            // Follow the runner's real C dependency and adjacent logical K origins. This
            // shadow execution preserves its packing and scale offsets across split-K.
            if (previous != clamp_probe_shadows.end() && previous->second.m == m &&
                previous->second.n == n && std::get<KIndex>(original) > 0 &&
                previous->second.k_end == std::get<KIndex>(original)) {
                SCOPED_TRACE("SplitKPropagation");
                ++coverage.split_k_calls;
                std::vector<TD> residual;
                for (float value : previous->second.values) {
                    residual.push_back(static_cast<TD>(value));
                }
                auto chained = original;
                std::get<c_index>(chained) = residual.data();
                std::get<c_index + 1>(chained) = stride;
                std::get<DIndex>(chained) = output.data();
                std::get<LddIndex>(chained) = stride;
                std::get<ClampIndex>(chained) = static_cast<ClampT>(-INFINITY);
                std::get<ClampIndex + 1>(chained) = static_cast<ClampT>(INFINITY);
                std::fill(output.begin(), output.end(), sentinel);
                std::apply(Run, chained);
                next_shadow.values.clear();
                for (const TD value : output) {
                    next_shadow.values.push_back(static_cast<float>(value));
                }
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        const float before = previous->second.values[i * stride + j];
                        const float after = static_cast<float>(output[i * stride + j]);
                        if (!std::isfinite(before)) {
                            expect_clamp_value(after, before);
                        } else {
                            EXPECT_TRUE(std::isfinite(after));
                        }
                    }
                }
                // A final finite clamp must still use the original max-then-min semantics.
                std::get<ClampIndex>(chained) = static_cast<ClampT>(-10.0f);
                std::get<ClampIndex + 1>(chained) = static_cast<ClampT>(10.0f);
                std::apply(Run, chained);
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        const float expected =
                            std::fmin(std::fmax(next_shadow.values[i * stride + j], -10.0f), 10.0f);
                        expect_clamp_value(static_cast<float>(output[i * stride + j]), expected);
                    }
                }
            }
            clamp_probe_shadows[{variant,
                                 reinterpret_cast<uintptr_t>(std::get<DIndex>(original))}] =
                std::move(next_shadow);
        }
        // Leave the exact output requested by the existing runner, including split-K C == D.
        std::apply(Run, original);
    }
};

template <auto Run, typename TD, size_t SourceIndex, size_t DIndex, size_t LddIndex,
          size_t ClampIndex, ClampSource Source = ClampSource::kColumnBias,
          bool NarrowClamp = false, size_t KIndex = 5, bool HasFloatingC = true>
auto clamp_test_run()
{
    using Probe = GemmClampProbe<Run, TD, SourceIndex, DIndex, LddIndex, ClampIndex, Source,
                                 NarrowClamp, KIndex, HasFloatingC, decltype(Run)>;
    return gemm_clamp_probe_active ? Probe::run : Run;
}

}  // namespace test
}  // namespace torq_tile
