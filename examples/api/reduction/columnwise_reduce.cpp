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

// Strided input, contiguous reduced output and optional elementwise initial values.
#if TORQ_TILE_HAS_RVV
#include "torq_tile/reduction/sum_f32_f32/sumcol_f32_f32/tqt_sumcol_f32_f32_interface.h"
#endif
#if TORQ_TILE_HAS_RVV
#include "torq_tile/reduction/sum_f32_f32/sumcol_f32_f32/tqt_sumcol_f32_f32_2x8vl_rvv.h"
#endif

namespace
{
using Kernel = tqt_sumcol_f32_f32_ukernel;
struct Variant
{
    const char *name;
    Kernel kernel;
};
const Variant variants[] = {
#if TORQ_TILE_HAS_RVV
    {"sumcol_f32_f32_2x8vl_rvv",
     {tqt_get_m_step_sumcol_f32_f32_2x8vl_rvv, tqt_get_n_step_sumcol_f32_f32_2x8vl_rvv,
      tqt_run_sumcol_f32_f32_2x8vl_rvv}},
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
    const size_t m = 2 * v.kernel.get_m_step() + 1, n = 2 * v.kernel.get_n_step() + 3;
    const size_t lda = n + 5, count = n;
    std::vector<float> a(m * lda), init(count, 0.25f), out(count), ref(count);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = input_value(i);
    for (bool use_init : {false, true}) {
        std::fill(ref.begin(), ref.end(), use_init ? 0.25f : 0.0f);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                ref[j] += a[i * lda + j];
        v.kernel.run_sumcol(m, n, a.data(), lda, out.data(), use_init ? init.data() : nullptr);
        std::printf("%s [%zu,%zu] -> [%zu], init=%d\n", v.name, m, n, count, use_init);
        if (!verify(1, count, out.data(), count, ref))
            return false;
    }
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
