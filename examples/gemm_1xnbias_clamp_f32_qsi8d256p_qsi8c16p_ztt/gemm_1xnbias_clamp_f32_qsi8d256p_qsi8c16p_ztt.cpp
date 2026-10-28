//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_ztt)
#error This file must be compiled for riscv, ztt.
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_interface.h"

static const tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_ukernel ukernel = {
    tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_scale_a_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_scale_b_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
};

int main()
{
    constexpr size_t k = 256;
    const size_t m = ukernel.get_m_step() + 1;
    const size_t n = ukernel.get_n_step() + 3;
    std::vector<int8_t> a(m * k, 1), b(n * k, 1);
    std::vector<uint8_t> a_packed(ukernel.get_a_packed_size(m, k));
    std::vector<uint8_t> b_packed(ukernel.get_b_packed_size(n, k));
    std::vector<float> scale_a(m, 1.0f), scale_b(n * (k / 16), 1.0f), output(m * n);

    ukernel.run_a_pack(m, k, k, k, 0, a.data(), a_packed.data());
    ukernel.run_bt_pack(n, k, k, k, 0, b.data(), b_packed.data());
    const tqt_qsi8d256_qsi8c16_params params = {scale_a.data(), 1, scale_b.data(), k / 16};
    ukernel.run_gemm(m, n, k, a_packed.data(), k, 0, b_packed.data(), k, 0, nullptr, n,
                     output.data(), n, nullptr, -1.0e30f, 1.0e30f, &params);
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - (float)k) > 1.0e-4f) {
            std::printf("Status: FAILED (index=%zu, got=%f)\n", i, output[i]);
            return 1;
        }
    }
    std::printf("m=%zu n=%zu k=%zu\nStatus: PASSED\n", m, n, k);
    return 0;
}
