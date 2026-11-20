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
#include <cstdio>
#include <exception>
#include <vector>

// Contiguous equal-length inputs or one scalar; includes a non-multiple tail.
#if TORQ_TILE_HAS_RVV
#include "torq_tile/elementwise/add_f32_f32/addvv_f32_f32_f32/tqt_addvv_f32_f32_f32_interface.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/elementwise/add_f32_f32/addvv_f32_f32_f32/tqt_addvv_f32_f32_f32_8vl_rvv.h"
#endif

namespace
{
using Kernel = tqt_addvv_f32_f32_f32_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV
    {"addvv_f32_f32_f32_8vl_rvv",
     {tqt_get_n_step_addvv_f32_f32_f32_8vl_rvv, tqt_run_addvv_f32_f32_f32_8vl_rvv}},
#endif
};

// Keep reference computation and checks visible in this standalone teaching source.
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

bool run_case(const Variant &v)
{
    const size_t n = 2 * v.kernel.get_n_step() + 3;
    std::vector<float> a(n), b(n), out(n), ref(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = input_value(i);
        b[i] = input_value(i + 3);
        ref[i] = a[i] + b[i];
    }
    v.kernel.run_addvv(n, a.data(), b.data(), out.data());
    std::printf("%s N=%zu\n", v.name, n);
    if (!verify(1, n, out.data(), n, ref))
        return false;
    // Exact alias is supported; arbitrary partial overlap is not demonstrated.
    v.kernel.run_addvv(n, a.data(), b.data(), a.data());
    return verify(1, n, a.data(), n, ref);
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
