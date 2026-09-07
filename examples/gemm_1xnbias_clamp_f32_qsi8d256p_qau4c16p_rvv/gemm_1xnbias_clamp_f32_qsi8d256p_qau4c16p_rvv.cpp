//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "src/gemm/gemm_i8_u4/gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv.h"

int main()
{
    constexpr size_t k = 256;
    constexpr size_t activation_block = 256;
    constexpr size_t weight_block = 16;
    constexpr size_t bits_per_weight = 4;
    const size_t m = 5;
    const size_t n = tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv() + 3;

    std::vector<int8_t> a(m * k, 1);
    std::vector<uint8_t> b(n * k * bits_per_weight / 8, 1);
    std::fill(b.begin(), b.end(), 0x11);
    std::vector<int8_t> a_packed(m * k);
    std::vector<uint8_t> b_packed(b.size());
    std::vector<float> scale_a(m * (k / activation_block), 1.0f);
    std::vector<int16_t> bsums_a(m * (k / 16), 16);
    std::vector<uint8_t> qscale_b(n * (k / weight_block), 1);
    std::vector<uint8_t> qmin_b(n * (k / weight_block), 0);
    std::vector<float> super_scale_b(n * (k / activation_block), 1.0f);
    std::vector<float> super_min_b(n * (k / activation_block), 0.0f);
    std::vector<float> output(m * n);

    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv(m, k, k, k, 0, a.data(),
                                                                       a_packed.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv(n, k, k, k, 0, b.data(),
                                                                        b_packed.data());
    const tqt_qsi8d256_qau4c16_params params = {scale_a.data(),
                                                k / 256,
                                                bsums_a.data(),
                                                k / 16,
                                                qscale_b.data(),
                                                qmin_b.data(),
                                                k / 16,
                                                super_scale_b.data(),
                                                super_min_b.data(),
                                                k / activation_block};
    tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv(
        m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, nullptr, n, output.data(), n,
        nullptr, -1.0e30f, 1.0e30f, &params);

    for (size_t i = 0; i < output.size(); ++i) {
        if (std::abs(output[i] - (float)k) > 1.0e-4f) {
            std::printf("Status: FAILED (index=%zu: got %f, expected %zu)\n", i, output[i], k);
            return 1;
        }
    }
    std::printf("m=%zu n=%zu k=%zu\n", m, n, k);
    std::printf("Status: PASSED\n");
    return 0;
}
