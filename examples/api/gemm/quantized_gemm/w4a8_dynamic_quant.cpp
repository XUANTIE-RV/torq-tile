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
#include "torq_tile/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_interface.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt/tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv.h"
#endif

namespace
{
using Kernel = tqt_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV
    {"gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
      tqt_get_n_step_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
      tqt_get_a_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
      tqt_get_b_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
      tqt_get_c_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
      tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
      tqt_get_d_offset_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
      tqt_get_d_size_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv,
      tqt_run_gemm_1xnbias_clamp_f32_qai8dx_qsi4cxt_4x1vl_rvv}},
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
    const size_t m = 9, n = v.kernel.get_n_step() + 3, k = 64;
    std::vector<float> af(m * k), bf(n * k), sa(m), sb(n), bias(n, 0.125f), ref(m * n),
        original(m * n), out(m * n);
    std::vector<int8_t> a(m * k);
    std::vector<int32_t> za(m);
    std::vector<uint8_t> b(n * k / 2, 0);
    for (size_t i = 0; i < m; ++i) {
        float low = 0, high = 0;  // Include zero so constant and zero rows are well defined.
        for (size_t q = 0; q < k; ++q) {
            const float x = i == 0 ? 0.0f : input_value(i * k + q) * 1.13f + 0.2f;
            af[i * k + q] = x;
            low = std::min(low, x);
            high = std::max(high, x);
        }
        sa[i] = high > low ? (high - low) / 255 : 1.0f;
        za[i] = std::clamp(static_cast<int>(std::round(-128 - low / sa[i])), -128, 127);
        for (size_t q = 0; q < k; ++q)
            a[i * k + q] = static_cast<int8_t>(
                std::clamp(static_cast<int>(std::round(af[i * k + q] / sa[i])) + za[i], -128, 127));
    }
    for (size_t j = 0; j < n; ++j) {
        float maximum = 0;
        for (size_t q = 0; q < k; ++q) {
            bf[j * k + q] = j == 0 ? 0.0f : input_value(j * k + q + 5) * 1.31f;
            maximum = std::max(maximum, std::fabs(bf[j * k + q]));
        }
        sb[j] = maximum ? maximum / 7 : 1.0f;
        for (size_t q = 0; q < k; ++q) {
            const int code =
                std::clamp(static_cast<int>(std::round(bf[j * k + q] / sb[j])), -8, 7) + 8;
            b[j * k / 2 + q / 2] |= static_cast<uint8_t>(code << (4 * (q % 2)));
        }
    }
    tqt_qai8dx_qsi4cx_params params{sa.data(), za.data(), sb.data()};
    // Whole-matrix call keeps scale pointers at their full row/channel origins.
    v.kernel.run_gemm(m, n, k, a.data(), k, 0, b.data(), k, 0, nullptr, n, out.data(), n,
                      bias.data(), -10, 10, &params);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            float real = 0;
            for (size_t q = 0; q < k; ++q) {
                const int code = ((b[j * k / 2 + q / 2] >> (4 * (q % 2))) & 15) - 8;
                acc += (static_cast<int32_t>(a[i * k + q]) - za[i]) * code;
                real = std::fma(af[i * k + q], bf[j * k + q], real);
            }
            ref[i * n + j] = std::clamp(acc * sa[i] * sb[j] + bias[j], -10.0f, 10.0f);
            original[i * n + j] = std::clamp(real + bias[j], -10.0f, 10.0f);
        }
    std::printf("%s: row scales/zero points, channel weight scales\n", v.name);
    if (!verify(m, n, out.data(), n, ref))
        return false;
    report_loss(ref, original);
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
