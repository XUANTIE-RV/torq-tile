//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv)
#error This consumer requires a RISC-V target.
#endif

#include <cmath>
#include <cstdio>
#include <vector>

#ifdef TQT_INSTALLED_CONSUMER
#include "torq_tile/tqt_compile_config.h"
#endif
#if TQT_INTEGRATION_RVV
#if defined(TQT_INSTALLED_CONSUMER) && !TORQ_TILE_HAS_RVV
#error The installed library does not contain the selected backend.
#endif
#ifdef TQT_INSTALLED_CONSUMER
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv.h"
#else
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv.h"
#endif
#define GET_M tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv
#define GET_N tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv
#define RUN tqt_run_gemm_1xnbias_clamp_f32_f32_f32_8x3vl_rvv
#endif
#if TQT_INTEGRATION_ZTT
#if defined(TQT_INSTALLED_CONSUMER) && !TORQ_TILE_HAS_ZTT_GEMM_F32
#error The installed library does not contain the selected backend.
#endif
#ifdef TQT_INSTALLED_CONSUMER
#include "torq_tile/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt.h"
#else
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt.h"
#endif
#define GET_M tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt
#define GET_N tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt
#define RUN tqt_run_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt
#endif

int main()
{
    // Exact dyadic inputs make the reference comparison reproducible.
    // Aligned dimensions also satisfy the ordinary Ztt API.
    const size_t m = 2 * GET_M(), n = 2 * GET_N(), k = 3 * GET_N();
    std::vector<float> a(m * k, 0.5f), b(k * n, 0.25f), bias(n, 0.125f);
    std::vector<float> c(m * n, 0.25f), out(m * n);
    RUN(m, n, k, a.data(), k, 0, b.data(), n, 0, c.data(), n, out.data(), n, bias.data(), -INFINITY,
        INFINITY);
    const float expected = k * 0.125f + 0.375f;
    for (float value : out) {
        if (!std::isfinite(value) || std::fabs(value - expected) > 1e-5f) {
            std::fprintf(stderr, "Mismatch: actual=%g expected=%g\n", value, expected);
            return 1;
        }
    }
    std::puts("Status: PASSED");
    return 0;
}
