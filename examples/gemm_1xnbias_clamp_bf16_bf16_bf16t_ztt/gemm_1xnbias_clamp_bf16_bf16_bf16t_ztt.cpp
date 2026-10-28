//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_ztt)
#error This example requires Ztt support.
#endif

#include <cmath>
#include <cstdio>
#include <vector>

#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16t/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16_bf16_bf16t/tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt.h"

using T = bfloat16_t;

static const tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_ukernel native = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_ztt,
};

static const tqt_gemm_1xnbias_clamp_bf16_bf16_bf16t_ukernel widening = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16_bf16_bf16t_1nlenxpnlen_f32acc_ztt,
};

template <bool Wide, typename Kernel>
static bool run_case(const Kernel &kernel)
{
    const size_t s = kernel.get_m_step(), step = kernel.get_n_step();
    const size_t m = 2 * s, n = 2 * step, k = 2 * step;
    std::vector<T> a(m * k), b(k * n), bt(n * k), c(m * n, (T)0.125f), d(m * n), bias(n, (T)0.25f);
    for (size_t i = 0; i < m; i++)
        for (size_t q = 0; q < k; q++)
            a[i * k + q] = (T)((int(i % 3) - 1) * 0.25f);
    for (size_t q = 0; q < k; q++)
        for (size_t j = 0; j < n; j++)
            b[q * n + j] = bt[j * k + q] = (T)((int(j % 5) - 2) * 0.5f);
    kernel.run_gemm(m, n, k, a.data(), k, 0, bt.data(), k, 0, c.data(), n, d.data(), n, bias.data(),
                    (T)-INFINITY, (T)INFINITY);
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            float sum = (float)c[i * n + j];
            for (size_t q = 0; q < k; q++) {
                if constexpr (Wide || sizeof(T) == sizeof(float))
                    sum = std::fma((float)a[i * k + q], (float)b[q * n + j], sum);
                else
                    sum =
                        (float)(T)std::fma((double)a[i * k + q], (double)b[q * n + j], (double)sum);
            }
            const float expected = (float)(T)(sum + (float)bias[j]);
            if ((float)d[i * n + j] != expected) {
                std::printf("Mismatch at %zu,%zu: %g != %g\n", i, j, (double)d[i * n + j],
                            (double)expected);
                return false;
            }
        }
    return true;
}
int main()
{
    bool ok = run_case<false>(native);
    ok = run_case<true>(widening) && ok;
    std::printf("Status: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
