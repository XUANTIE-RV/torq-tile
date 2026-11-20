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

#if TORQ_TILE_HAS_RVV_F16
#include "torq_tile/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_interface.h"
#endif
#if TORQ_TILE_HAS_RVV_F16
#include "torq_tile/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qsi4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv.h"
#endif

namespace
{
using Kernel = tqt_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV_F16
    {"gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv",
     {tqt_get_m_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_c_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_d_offset_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_d_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv,
      tqt_run_gemm_1xnbias_clamp_f32_f16p_qsi4c2p_8x1vl_rvv}},
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
    const size_t m = 9, n = v.kernel.get_n_step() + 3, k = 64, bl = 32, blocks = k / bl;
    std::vector<float16_t> a(m * k), scale(n * blocks);
    std::vector<float> original_b(n * k), ref(m * n), original(m * n), out(m * n), bias(n, 0.125f);
    std::vector<uint8_t> codes(n * k / 2, 0);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<float16_t>(input_value(i));
    for (size_t j = 0; j < n; ++j)
        for (size_t block = 0; block < blocks; ++block) {
            float maximum = 0;
            for (size_t q = block * bl; q < (block + 1) * bl; ++q) {
                const float x = (j == 0 && block == 0) ? 0.0f : input_value(j * k + q + 5) * 1.13f;
                original_b[j * k + q] = x;
                maximum = std::max(maximum, std::fabs(x));
            }
            // Store the scale first, then quantize against that represented scale.
            scale[j * blocks + block] = static_cast<float16_t>(maximum ? maximum / 7 : 1.0f);
            for (size_t q = block * bl; q < (block + 1) * bl; ++q) {
                const int code =
                    std::clamp(
                        static_cast<int>(std::round(original_b[j * k + q] /
                                                    static_cast<float>(scale[j * blocks + block]))),
                        -8, 7) +
                    8;
                codes[j * k / 2 + q / 2] |= static_cast<uint8_t>(code << (4 * (q % 2)));
            }
        }
    std::vector<uint8_t> ap(v.kernel.get_a_packed_size(m, k)), bp(v.kernel.get_b_packed_size(n, k));
    v.kernel.run_a_pack(m, k, k, k, 0, a.data(), ap.data());
    v.kernel.run_bt_pack(n, k, k, k, 0, codes.data(), bp.data());
    v.kernel.run_gemm(m, n, k, bl, ap.data(), k, 0, bp.data(), k, 0, scale.data(), blocks, nullptr,
                      n, out.data(), n, bias.data(), -10, 10, nullptr);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            float sum = 0, real = 0;
            for (size_t block = 0; block < blocks; ++block) {
                float partial = 0;
                for (size_t q = block * bl; q < (block + 1) * bl; ++q) {
                    const int code = (codes[j * k / 2 + q / 2] >> (4 * (q % 2))) & 15;
                    partial = std::fma(static_cast<float>(a[i * k + q]),
                                       static_cast<float>(code - 8), partial);
                    real = std::fma(static_cast<float>(a[i * k + q]), original_b[j * k + q], real);
                }
                sum = std::fma(partial, static_cast<float>(scale[j * blocks + block]), sum);
            }
            ref[i * n + j] = std::clamp(sum + bias[j], -10.0f, 10.0f);
            original[i * n + j] = std::clamp(real + bias[j], -10.0f, 10.0f);
        }
    std::printf("%s: BL=%zu, two K groups, zero weight group\n", v.name, bl);
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
