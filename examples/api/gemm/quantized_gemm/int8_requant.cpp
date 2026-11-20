//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv)
#error These examples require a RISC-V target.
#endif

#include <torq_tile/tqt_compile_config.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_interface.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_i8_i8/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv.h"
#endif

namespace
{
using Kernel = tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV
    {"gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv",
     {tqt_get_m_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
      tqt_get_n_step_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
      tqt_get_a_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
      tqt_get_b_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
      tqt_get_c_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
      tqt_get_bias_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
      tqt_get_d_offset_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
      tqt_get_d_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv,
      tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8_qsi8cxt_4x1vl_rvv}},
#endif
};

// Keep reference computation and checks visible in this standalone teaching source.
size_t checked_product(size_t a, size_t b)
{
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("matrix size overflow");
    return a * b;
}
float input_value(size_t index)
{
    return (static_cast<int>(index % 17) - 8) * 0.0625f;
}
bool verify(size_t m, size_t n, const float *actual, size_t stride,
            const std::vector<float> &reference, float tolerance = 1e-4f)
{
    float max_error = 0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float a = actual[i * stride + j], r = reference[i * n + j];
            if (!std::isfinite(a) || !std::isfinite(r) ||
                std::fabs(a - r) > tolerance * (1.0f + std::fabs(r))) {
                std::fprintf(stderr, "Mismatch [%zu,%zu]: actual=%g reference=%g\n", i, j, a, r);
                return false;
            }
            max_error = std::max(max_error, std::fabs(a - r));
        }
    }
    std::printf("  maximum absolute error: %g\n", max_error);
    return true;
}

// Quantization loss is diagnostic, separate from kernel correctness.
void report_loss(const std::vector<float> &quantized, const std::vector<float> &original)
{
    double error = 0, energy = 0;
    float maximum = 0;
    for (size_t i = 0; i < original.size(); ++i) {
        if (!std::isfinite(quantized[i]) || !std::isfinite(original[i]))
            throw std::runtime_error("non-finite quantization result");
        const double delta = quantized[i] - original[i];
        error += delta * delta;
        energy += original[i] * original[i];
        maximum = std::max(maximum, static_cast<float>(std::fabs(delta)));
    }
    std::printf("  quantization loss: max_abs=%g normalized_RMSE=%g (diagnostic)\n", maximum,
                std::sqrt(error / std::max(energy, 1e-30)));
}

bool run_case(const Variant &v)
{
    const size_t m = 9, n = 2 * v.kernel.get_n_step() + 1, k = 32;
    const float sa = 0.125f, sd = 0.0625f;
    const int za = -3, zd = 5;
    std::vector<float> af(m * k), bf(n * k), sb(n), original(m * n), dequantized(m * n);
    std::vector<int8_t> a(m * k), b(n * k), out(m * n);
    std::vector<int32_t> bias(n);
    for (size_t i = 0; i < m * k; ++i) {
        af[i] = i < k ? 0.0f : input_value(i) * 3.1f;
        a[i] = static_cast<int8_t>(
            std::clamp(static_cast<int>(std::round(af[i] / sa)) + za, -128, 127));
    }
    for (size_t j = 0; j < n; ++j) {
        // Exact effective multipliers 0.5, 1 and 2 expose ties and saturation.
        sb[j] = 0.25f * static_cast<float>(1u << (j % 3));
        bias[j] = static_cast<int32_t>(j % 5) - 2;
        if (j == n - 1)
            bias[j] = 1000;
        for (size_t q = 0; q < k; ++q) {
            bf[j * k + q] = j == 0 ? 0.0f : input_value(j * k + q + 5) * 4.3f;
            b[j * k + q] = static_cast<int8_t>(
                std::clamp(static_cast<int>(std::round(bf[j * k + q] / sb[j])), -128, 127));
        }
    }
    tqt_qai8_qai8_qsi8cx_params params{sa, za, sb.data(), static_cast<int32_t>(n), sd, zd};
    v.kernel.run_gemm(m, n, k, a.data(), k, 0, b.data(), k, 0, nullptr, n, out.data(), n,
                      bias.data(), -100, 100, &params);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = bias[j];
            float real = bias[j] * sa * sb[j];
            for (size_t q = 0; q < k; ++q) {
                acc += (static_cast<int32_t>(a[i * k + q]) - za) * b[j * k + q];
                real += af[i * k + q] * bf[j * k + q];
            }
            // Effective multiplier is (1 << (j % 3)) / 2. Round once, ties to even.
            const int64_t numerator = static_cast<int64_t>(acc) * (1u << (j % 3));
            const int64_t magnitude = numerator < 0 ? -numerator : numerator;
            int64_t rounded = magnitude / 2;
            if ((magnitude % 2) && (rounded % 2))
                ++rounded;
            if (numerator < 0)
                rounded = -rounded;
            const int expected = std::clamp(static_cast<int>(rounded) + zd, -100, 100);
            if (out[i * n + j] != expected) {
                std::fprintf(stderr, "Mismatch [%zu,%zu]: actual=%d reference=%d\n", i, j,
                             out[i * n + j], expected);
                return false;
            }
            dequantized[i * n + j] = (out[i * n + j] - zd) * sd;
            original[i * n + j] = std::clamp(real, (-100 - zd) * sd, (100 - zd) * sd);
        }
    std::printf("%s: integer contract exact\n", v.name);
    report_loss(dequantized, original);
    return true;
}

}  // namespace

int main()
{
    try {
        bool ok = true;
        for (const auto &v : variants)
            ok = run_case(v) && ok;
        std::printf("Status: %s\n", ok ? "PASSED" : "FAILED");
        return ok ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
