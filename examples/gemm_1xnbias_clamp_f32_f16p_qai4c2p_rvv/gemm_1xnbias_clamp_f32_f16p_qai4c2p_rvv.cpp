//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This example requires RVV, zfh and zvfh.
#endif

#include <cmath>
#include <cstdio>
#include <vector>

#include "src/gemm/gemm_f16_i4/gemm_1xnbias_clamp_f32_f16p_qai4c2p/tqt_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv.h"

int main()
{
    const size_t m = 9, n = tqt_get_n_step_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv() + 1,
                 k = 64, bl = 32, lds = 2;
    std::vector<float16_t> a(m * k, (float16_t)0.5f);
    std::vector<uint8_t> b(n * k / 2, 0xe3);
    std::vector<float> scale(n * lds), offset(n * lds), bias(n, 0.25f), d(m * n);
    for (size_t j = 0; j < n; ++j) {
        scale[j * lds] = 0.125f;
        scale[j * lds + 1] = 0.0f;
        offset[j * lds] = -0.5f;
        offset[j * lds + 1] = 1.25f;
    }
    std::vector<uint8_t> ap(
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(m, k)),
        bp(tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(n, k));
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(m, k, k, k, 0, a.data(),
                                                                 ap.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(n, k, k, k, 0, b.data(),
                                                                  bp.data());
    tqt_run_gemm_1xnbias_clamp_f32_f16p_qai4c2p_8x1vl_rvv(
        m, n, k, bl, ap.data(), k, 0, bp.data(), k, 0, scale.data(), lds, offset.data(), lds,
        nullptr, n, d.data(), n, bias.data(), -100.0f, 100.0f);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double expected = bias[j];
            for (size_t q = 0; q < k; ++q) {
                const unsigned code = (b[j * k / 2 + q / 2] >> (4 * (q & 1))) & 15;
                expected += (double)a[i * k + q] *
                            (code * scale[j * lds + q / bl] + offset[j * lds + q / bl]);
            }
            if (std::fabs(d[i * n + j] - expected) > 1e-5)
                return 1;
        }
    }
    std::puts("Status: PASSED");
    return 0;
}
