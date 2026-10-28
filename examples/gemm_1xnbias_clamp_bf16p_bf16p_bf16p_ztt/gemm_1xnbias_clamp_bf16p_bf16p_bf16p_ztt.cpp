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

#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16p_bf16p_bf16p/tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt.h"
#include "src/gemm/gemm_bf16_bf16/gemm_1xnbias_clamp_bf16p_bf16p_bf16p/tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt.h"

using T = bfloat16_t;

static const tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_ukernel native = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_d_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_get_bias_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_a_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_b_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_bt_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_c_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_bias_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_d_unpack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_ztt,
};

static const tqt_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_ukernel widening = {
    tqt_get_m_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_bias_packed_size_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_a_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_b_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_bt_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_c_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_bias_pack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_d_unpack_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_gemm_1xnbias_clamp_bf16p_bf16p_bf16p_1nlenxpnlen_f32acc_ztt,
};

template <bool Wide, typename Kernel>
static bool run_case(const Kernel &kernel)
{
    const size_t s = kernel.get_m_step(), step = kernel.get_n_step();
    const size_t m = s + 1, n = step + 3, k = step + 1;
    std::vector<T> a(m * k), b(k * n), bt(n * k), c(m * n, (T)0.125f), d(m * n), bias(n, (T)0.25f);
    for (size_t i = 0; i < m; i++)
        for (size_t q = 0; q < k; q++)
            a[i * k + q] = (T)((int(i % 3) - 1) * 0.25f);
    for (size_t q = 0; q < k; q++)
        for (size_t j = 0; j < n; j++)
            b[q * n + j] = bt[j * k + q] = (T)((int(j % 5) - 2) * 0.5f);
    std::vector<T> ap(kernel.get_a_packed_size(m, k) / sizeof(T));
    std::vector<T> bp(kernel.get_b_packed_size(n, k) / sizeof(T));
    std::vector<T> cp(kernel.get_d_packed_size(m, n) / sizeof(T)), dp(cp.size());
    std::vector<T> biasp(kernel.get_bias_packed_size(n) / sizeof(T));
    kernel.run_a_pack(m, k, k, a.data(), ap.data());
    kernel.run_b_pack(n, k, n, b.data(), bp.data());
    kernel.run_c_pack(m, n, n, c.data(), cp.data());
    kernel.run_bias_pack(n, bias.data(), biasp.data());
    kernel.run_gemm(m, n, k, ap.data(), bp.data(), cp.data(), dp.data(), biasp.data(), (T)-INFINITY,
                    (T)INFINITY);
    kernel.run_d_unpack(m, n, n, dp.data(), d.data());
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
