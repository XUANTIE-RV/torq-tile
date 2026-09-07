//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "src/gemm/gemm_i8_b1/gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv.h"

int main()
{
    constexpr size_t k = 128;
    const size_t m = tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv() + 1;
    const size_t n = tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv() + 3;
    std::vector<int8_t> a(m * k, 1);
    std::vector<uint8_t> b(n * k / 8, 0xff);
    std::vector<uint8_t> a_packed(
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv(m, k));
    std::vector<uint8_t> b_packed(
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv(n, k));
    std::vector<float> scale_a(m * (k / 32), 1.0f), scale_b(n * (k / 128), 1.0f);
    std::vector<float> output(m * n);

    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv(m, k, k, k, 0, a.data(),
                                                                       a_packed.data());
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv(n, k, k, k, 0, b.data(),
                                                                        b_packed.data());
    const tqt_qsi8d32_qsb1c128_params params = {scale_a.data(), k / 32, scale_b.data(), k / 128};
    tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsb1c128p_4x1vl_rvv(
        m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, nullptr, n, output.data(), n,
        nullptr, -1.0e30f, 1.0e30f, &params);
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - static_cast<float>(k)) > 1.0e-4f) {
            std::printf("Status: FAILED (index=%zu, got=%f)\n", i, output[i]);
            return 1;
        }
    }
    std::printf("m=%zu n=%zu k=%zu\nStatus: PASSED\n", m, n, k);
    return 0;
}
