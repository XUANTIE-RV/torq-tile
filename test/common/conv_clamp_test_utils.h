//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>

#include "test/common/dwconv_test_runner.h"
#include "test/common/igemm_test_runner.h"

namespace torq_tile
{
namespace test
{

template <typename T>
float conv_clamp_max_finite()
{
    if constexpr (std::is_same_v<T, float16_t>)
        return 65504.0f;
    else if constexpr (std::is_same_v<T, bfloat16_t>)
        return 0x1.fep127f;
    else
        return FLT_MAX;
}

template <typename T>
auto conv_clamp_ranges()
{
    const float max = conv_clamp_max_finite<T>();
    return std::array<std::pair<float, float>, 5>{
        {{-INFINITY, INFINITY}, {-1, 1}, {-INFINITY, 1}, {-1, INFINITY}, {-max, max}}};
}

inline float conv_clamp_special(size_t index)
{
    const float values[] = {NAN, INFINITY, -INFINITY, 20, -20, 0.5f, 0.0f, -0.0f};
    return values[index % 8];
}

// The range index is test data, independent of the production predicate.
inline float conv_clamp_expected(float value, size_t range, std::pair<float, float> bounds)
{
    return range == 0 ? value : std::fmin(std::fmax(value, bounds.first), bounds.second);
}

inline testing::AssertionResult conv_clamp_equal(float actual, float expected)
{
    const bool equal = std::isnan(expected) ? std::isnan(actual)
                       : std::isinf(expected)
                           ? std::isinf(actual) && std::signbit(actual) == std::signbit(expected)
                           : std::isfinite(actual) && actual == expected;
    if (equal)
        return testing::AssertionSuccess();
    return testing::AssertionFailure() << "actual=" << actual << " expected=" << expected;
}

template <typename T>
void run_clamp_dwconv(const NonPackedDwconvFuncs<T> &f, const tqt_conv_params &p, const T *a,
                      const T *w, const T *bias, T *out, T lo, T hi)
{
    f.run_dwconv(a, w, bias, out, &p, lo, hi);
}

template <typename T, typename Funcs>
void run_clamp_packed_dwconv(const Funcs &f, const tqt_conv_params &p, const T *a, const T *w,
                             const T *bias, T *out, T lo, T hi)
{
    std::vector<uint8_t> ap(f.get_packed_input_size(&p)), wp(f.get_packed_weight_size(&p));
    const size_t bytes = f.get_output_size(&p);
    std::vector<T> dp(bytes / sizeof(T) + 16, T(37));
    f.run_pack_input(a, ap.data(), &p);
    f.run_pack_weight(w, wp.data(), &p);
    if constexpr (std::is_same_v<Funcs, SpecPackedDwconvFuncs<T>>) {
        for (size_t h = 0; h < p.output_shape[1]; h += f.get_oh_step())
            f.run_dwconv(h, std::min(h + f.get_oh_step(), p.output_shape[1]), ap.data(), wp.data(),
                         bias, dp.data() + 8, &p, lo, hi);
    } else {
        f.run_dwconv(ap.data(), wp.data(), bias, dp.data() + 8, &p, lo, hi);
    }
    f.run_unpack_output(dp.data() + 8, out, &p);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(float(dp[i]), 37);
        EXPECT_EQ(float(dp[8 + bytes / sizeof(T) + i]), 37);
    }
}

template <typename T>
void run_clamp_dwconv(const GenericPackedDwconvFuncs<T> &f, const tqt_conv_params &p, const T *a,
                      const T *w, const T *bias, T *out, T lo, T hi)
{
    run_clamp_packed_dwconv(f, p, a, w, bias, out, lo, hi);
}

template <typename T>
void run_clamp_dwconv(const SpecPackedDwconvFuncs<T> &f, const tqt_conv_params &p, const T *a,
                      const T *w, const T *bias, T *out, T lo, T hi)
{
    run_clamp_packed_dwconv(f, p, a, w, bias, out, lo, hi);
}

template <typename T>
void run_clamp_dwconv(const SpecWeightPackedDwconvFuncs<T> &f, const tqt_conv_params &p, const T *a,
                      const T *w, const T *bias, T *out, T lo, T hi)
{
    std::vector<uint8_t> wp(f.get_packed_weight_size(&p));
    f.run_pack_weight(w, wp.data(), &p);
    for (size_t h = 0; h < p.output_shape[1]; h += f.get_oh_step())
        f.run_dwconv(h, std::min(h + f.get_oh_step(), p.output_shape[1]), a, wp.data(), bias, out,
                     &p, lo, hi);
}

template <typename T, typename Funcs>
void check_dwconv_clamp(const Funcs &f, ConvLayout layout, size_t kh, size_t kw, size_t stride,
                        size_t channels_step, const std::array<size_t, 5> &widths)
{
    const auto ranges = conv_clamp_ranges<T>();
    const size_t channels[] = {1, std::max(size_t(1), channels_step - 1), channels_step,
                               channels_step + 1, 2 * channels_step + 1};
    for (size_t shape = 0; shape < 5; ++shape) {
        for (size_t pad : {size_t(0), size_t(1)}) {
            const size_t c = channels[shape], dm = shape % 3 + 1;
            const size_t ph = pad * (kh / 2), pw = pad * ((kw - 1) / 2);
            const size_t ih = kh == 1 ? 1 : 4 * stride + kh - 2 * ph;
            const size_t iw = (widths[shape] - 1) * stride + kw - 2 * pw;
            auto tp = ConvTestParams(c, c * dm, c, ih, iw, kh, kw);
            tp.set_pad(ph, pw).set_stride(stride, stride);
            const auto p = tp.to_conv_params();
            const size_t spatial = p.output_shape[1] * p.output_shape[2];
            std::vector<T> a(dwconv_input_elements(p), T(1)), w(dwconv_weight_elements(p));
            std::vector<T> bias(p.output_channel), out(dwconv_output_elements(p) + 16, T(37));
            for (bool has_bias : {true, false}) {
                for (size_t special = 0; special < 8; ++special) {
                    std::fill(w.begin(), w.end(), T(0));
                    for (size_t oc = 0; oc < p.output_channel; ++oc) {
                        bias[oc] = T(conv_clamp_special(oc + special));
                        if (!has_bias) {
                            // The center tap is valid even on padded output borders.
                            const size_t tap = (kh / 2) * kw + kw / 2;
                            const size_t index = layout == ConvLayout::kChannelFirst
                                                     ? oc * kh * kw + tap
                                                     : tap * p.output_channel + oc;
                            w[index] = bias[oc];
                        }
                    }
                    for (size_t range = 0; range < ranges.size(); ++range) {
                        SCOPED_TRACE(testing::Message()
                                     << "shape=" << shape << " pad=" << pad << " bias=" << has_bias
                                     << " special=" << special << " range=" << range);
                        run_clamp_dwconv(f, p, a.data(), w.data(), has_bias ? bias.data() : nullptr,
                                         out.data() + 8, T(ranges[range].first),
                                         T(ranges[range].second));
                        for (size_t i = 0; i < dwconv_output_elements(p); ++i) {
                            const size_t oc = layout == ConvLayout::kChannelFirst
                                                  ? i / spatial
                                                  : i % p.output_channel;
                            ASSERT_TRUE(conv_clamp_equal(
                                float(out[i + 8]),
                                conv_clamp_expected(float(bias[oc]), range, ranges[range])))
                                << "element=" << i;
                        }
                        for (size_t i = 0; i < 8; ++i) {
                            ASSERT_EQ(float(out[i]), 37);
                            ASSERT_EQ(float(out[out.size() - 8 + i]), 37);
                        }
                    }
                }
            }
        }
    }
}

template <typename T, typename Funcs>
void check_dwconv_clamp(const Funcs &f, ConvLayout layout, size_t kh, size_t kw, size_t stride,
                        size_t channels_step, size_t width_step)
{
    check_dwconv_clamp<T>(
        f, layout, kh, kw, stride, channels_step,
        std::array<size_t, 5>{1, 2, width_step, width_step + 1, 2 * width_step + 3});
}

template <typename T, typename Funcs>
void check_igemm_clamp(const Funcs &f)
{
    constexpr bool cf = std::is_same_v<Funcs, NonPackedIgemmcfFuncs<T>> ||
                        std::is_same_v<Funcs, PackedIgemmcfFuncs<T>>;
    constexpr bool packed_cf = std::is_same_v<Funcs, PackedIgemmcfFuncs<T>>;
    constexpr bool packed_cl = std::is_same_v<Funcs, PackedIgemmclFuncs<T>>;
    const size_t ms = f.get_m_step(), ns = f.get_n_step();
    const auto ranges = conv_clamp_ranges<T>();
    for (size_t m : {size_t(1), ms - 1, ms, ms + 1, 2 * ms + 1}) {
        for (size_t n : {size_t(1), ns - 1, ns, ns + 1, 2 * ns + 1}) {
            const size_t oc = cf ? m : n, spatial = cf ? n : m;
            // Two KD planes and two ICs make both CF/CL two-way splits legal.
            auto tp = ConvTestParams(4, oc * 2, 2, 1, spatial, 1, 1);
            tp.input_shape[0] = tp.kernel_shape[0] = 2;
            const auto p = tp.to_conv_params();
            constexpr size_t k = 4, slice = 2;
            const size_t ld = n + 3;
            std::vector<T> a(4 * 2 * spatial, T(1)), w(oc * k), bias(oc);
            std::vector<T> out(m * ld + 16, T(37));
            for (int source = 0; source < 3; ++source) {
                for (size_t special = 0; special < 8; ++special) {
                    std::fill(w.begin(), w.end(), T(0));
                    for (size_t c = 0; c < oc; ++c) {
                        bias[c] = T(conv_clamp_special(c + special));
                        if (source == 1)
                            w[c * k] = bias[c];
                    }
                    std::vector<uint8_t> wp;
                    if constexpr (packed_cf) {
                        wp.resize(f.get_a_packed_size(oc, k));
                        f.run_a_pack(oc, k, k, k, 0, w.data(), wp.data());
                    } else if constexpr (packed_cl) {
                        wp.resize(f.get_b_packed_size(oc, k));
                        f.run_bt_pack(oc, k, k, k, 0, w.data(), wp.data());
                    }
                    auto run = [&](size_t origin, size_t length, float lo, float hi) {
                        // Raw kernels apply the K origin internally; packed kernels consume
                        // an already-offset panel, matching their public offset getters.
                        const void *weight = w.data();
                        if constexpr (packed_cf)
                            weight = wp.data() + f.get_a_packed_offset(0, origin, k, oc);
                        else if constexpr (packed_cl)
                            weight = wp.data() + f.get_b_packed_offset(0, origin, k, oc);
                        const void *c = origin ? out.data() + 8 : nullptr;
                        f.run_igemm(m, n, length, a.data(), 1, 0, origin, weight, k, origin, c, ld,
                                    out.data() + 8, ld, source == 0 ? bias.data() : nullptr, T(lo),
                                    T(hi), &p);
                    };
                    for (size_t range = 0; range < ranges.size(); ++range) {
                        for (bool split : {false, true}) {
                            if (source == 2 && !split)
                                continue;  // C is consumed only after a nonzero K origin.
                            SCOPED_TRACE(testing::Message()
                                         << "m=" << m << " n=" << n << " source=" << source
                                         << " special=" << special << " range=" << range
                                         << " split=" << split);
                            std::fill(out.begin(), out.end(), T(37));
                            if (split) {
                                run(0, slice, -1, 1);
                                for (size_t row = 0; row < m; ++row) {
                                    for (size_t col = 0; col < n; ++col) {
                                        const T value = bias[cf ? row : col];
                                        if (source == 2)
                                            out[8 + row * ld + col] = value;
                                        else
                                            ASSERT_TRUE(conv_clamp_equal(
                                                float(out[8 + row * ld + col]), float(value)));
                                    }
                                }
                                run(slice, slice, ranges[range].first, ranges[range].second);
                            } else {
                                run(0, k, ranges[range].first, ranges[range].second);
                            }
                            for (size_t row = 0; row < m; ++row) {
                                for (size_t col = 0; col < n; ++col)
                                    ASSERT_TRUE(conv_clamp_equal(
                                        float(out[8 + row * ld + col]),
                                        conv_clamp_expected(float(bias[cf ? row : col]), range,
                                                            ranges[range])));
                                for (size_t col = n; col < ld; ++col)
                                    ASSERT_EQ(float(out[8 + row * ld + col]), 37);
                            }
                            for (size_t i = 0; i < 8; ++i) {
                                ASSERT_EQ(float(out[i]), 37);
                                ASSERT_EQ(float(out[out.size() - 8 + i]), 37);
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace test
}  // namespace torq_tile
