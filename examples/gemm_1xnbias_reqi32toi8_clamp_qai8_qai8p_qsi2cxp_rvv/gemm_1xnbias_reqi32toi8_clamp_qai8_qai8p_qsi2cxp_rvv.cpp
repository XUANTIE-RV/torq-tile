//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v)
#error This example requires RISC-V vector support.
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "src/gemm/gemm_i8_i2/gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp/tqt_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv.h"

int main()
{
    const size_t m = 5, n = 19, k = 40;
    std::vector<int8_t> a(m * k), d(m * n);
    std::vector<uint8_t> b(n * k / 4);
    std::vector<int32_t> bias(n);
    std::vector<float> scale(n, 0.5f);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = (int8_t)(37 + (int)(i % 9) - 4);
    for (size_t i = 0; i < b.size(); ++i)
        b[i] = (uint8_t)(i * 83 + 29);
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] = (int32_t)(i % 11) - 5;
    const int8_t custom[4] = {-5, 3, 0, 7};
    const int8_t standard[4] = {-2, -1, 0, 1};
    tqt_qai8_qai8_qsi2cx_params params{1.0f, 37, scale.data(), 1.0f, -7};
    std::vector<int8_t> ap(
        tqt_get_a_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(m, k));
    std::vector<uint8_t> bp(
        tqt_get_b_packed_size_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(n, k));
    tqt_run_a_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(m, k, k, k, 0,
                                                                              a.data(), ap.data());
    tqt_run_bt_pack_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(n, k, k, k, 0,
                                                                               b.data(), bp.data());
    for (bool use_custom : {false, true}) {
        const int8_t *lut = use_custom ? custom : nullptr;
        tqt_run_gemm_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi2cxp_4x1vl_rvv(
            m, n, k, ap.data(), k, 0, bp.data(), k, 0, d.data(), n, bias.data(), -53, 79, lut,
            &params);
        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < n; ++col) {
                int64_t dot = bias[col];
                for (size_t j = 0; j < k; ++j) {
                    const int code = (b[col * (k / 4) + j / 4] >> (2 * (j % 4))) & 3;
                    dot += (int64_t)(a[row * k + j] - 37) *
                           (use_custom ? custom[code] : standard[code]);
                }
                const double scaled = (double)dot * 0.5;
                const double floor_value = std::floor(scaled);
                const double fraction = scaled - floor_value;
                const int rounded =
                    (int)floor_value +
                    (fraction > 0.5 || (fraction == 0.5 && ((int64_t)floor_value % 2 != 0)));
                const int expected = std::max(-53, std::min(79, rounded - 7));
                if (d[row * n + col] != expected) {
                    std::printf("Status: FAILED (row=%zu col=%zu)\n", row, col);
                    return 1;
                }
            }
        }
    }
    std::printf("Status: PASSED\n");
    return 0;
}
