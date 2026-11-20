//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv)
#error These examples require a RISC-V target.
#endif

#include <torq_tile/tqt_compile_config.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

// SiLU activation on a fused [M,2N] gate/up buffer with a padded row stride.
#if TORQ_TILE_HAS_RVV
#include "torq_tile/activation/swiglu_f32_f32_f32/swiglu_f32_f32_f32/tqt_swiglu_f32_f32_f32_interface.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/activation/swiglu_f32_f32_f32/swiglu_f32_f32_f32/tqt_swiglu_f32_f32_f32_1x4vl_rvv.h"
#endif

namespace
{
using Kernel = tqt_swiglu_f32_f32_f32_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV
    {"swiglu_f32_f32_f32_1x4vl_rvv",
     {tqt_get_m_step_swiglu_f32_f32_f32_1x4vl_rvv, tqt_get_n_step_swiglu_f32_f32_f32_1x4vl_rvv,
      tqt_run_swiglu_f32_f32_f32_1x4vl_rvv}},
#endif
};

// Keep reference computation and checks visible in this standalone teaching source.
float input_value(size_t index)
{
    return (static_cast<int>(index % 17) - 8) * 0.0625f;
}

bool run_case(const Variant &v)
{
    const size_t m = 3, n = v.kernel.get_n_step() + 3, stride = 2 * n + 7;
    std::vector<float> input(m * stride, 12345.0f), out(m * (n + 5), 12345.0f), ref(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const float gate = (static_cast<int>((i * n + j) % 41) - 20) * 0.5f;
            input[i * stride + j] = gate;
            input[i * stride + n + j] = input_value(j + 5) * 10;
            ref[i * n + j] = gate / (1.0f + std::exp(-gate)) * input[i * stride + n + j];
        }
    }
    v.kernel.run_swiglu(m, n, input.data(), stride, input.data() + n, stride, out.data(), n + 5);
    std::printf("%s M=%zu N=%zu stride=%zu\n", v.name, m, n, stride);
    // Absolute error bounds over [-10,10], inherited from the corresponding tests.
    auto check = [&](const float *data, size_t ld) {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                if (!std::isfinite(data[i * ld + j]) ||
                    std::fabs(data[i * ld + j] - ref[i * n + j]) > 5e-4f)
                    return false;
        return true;
    };
    if (!check(out.data(), n + 5))
        return false;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = n; j < n + 5; ++j)
            if (out[i * (n + 5) + j] != 12345.0f)
                return false;
    v.kernel.run_swiglu(m, n, input.data(), stride, input.data() + n, stride, input.data(), stride);
    return check(input.data(), stride);
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
