//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_i5/gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

static tqt_qsi8d32_qsi5c32_params make_qsi8d32_qsi5c32_params(const float *scale_a, size_t ldsa,
                                                              const float *scale_b, size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

TEST(GemmI8I5Rvv, Q5_0SourceAwareTailsAndSplitK)
{
    const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qsi5c32_params> funcs = {
        Qx0SourceEncoding::I5,
        5,
        32,
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        make_qsi8d32_qsi5c32_params,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>()};
    run_packed_qsi8_qx0_source_test(5, funcs.get_n_step() + 3, 64, funcs);
}

TEST(GemmI8I5Rvv, ClampSpecialValues_Q5_0SourceAwareTailsAndSplitK)
{
    ScopedGemmClampProbe probe;
    const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qsi5c32_params> funcs = {
        Qx0SourceEncoding::I5,
        5,
        32,
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv,
        make_qsi8d32_qsi5c32_params,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>()};
    for (size_t rows = 1;
         rows <= tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv() + 1; ++rows) {
        for (size_t columns :
             {size_t{1}, tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(),
              tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv() + 3}) {
            run_packed_qsi8_qx0_source_test(rows, columns, 64, funcs);
        }
    }
}

namespace
{
// Exact scales expose overflow without floating-point comparison tolerance.
struct Qsi5ExtremeData
{
    size_t m, n, k;
    std::vector<int8_t> a, b, ap;
    std::vector<uint8_t> codes, bp;
    std::vector<float> sa, sb, d, bias;
    tqt_qsi8d32_qsi5c32_params params;

    Qsi5ExtremeData(size_t rows, size_t cols, size_t depth, int pattern)
        : m(rows),
          n(cols),
          k(depth),
          a(m * k),
          b(n * k),
          ap(m * k),
          codes(n * k / 32 * 20),
          bp(n * k / 32 * 20),
          sa(m * k / 32, 0.5f),
          sb(n * k / 32, 0.125f),
          d(m * n),
          bias(n, 0.25f),
          params{sa.data(), k / 32, sb.data(), k / 32}
    {
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = pattern == 0 ? -128 : pattern == 1 ? 127 : int((i * 29) % 256) - 128;
        for (size_t i = 0; i < b.size(); ++i)
            b[i] = pattern < 2 ? -16 : pattern == 2 ? 15 : int((i * 13) % 32) - 16;
        for (size_t row = 0; row < n; ++row) {
            for (size_t kk = 0; kk < k; ++kk) {
                const int value = b[row * k + kk];
                auto *group = codes.data() + row * k / 32 * 20 + kk / 32 * 20;
                const size_t slot = kk % 32;
                group[slot / 2] |= uint8_t(value & 15) << (slot % 2 * 4);
                group[16 + slot / 8] |= uint8_t(value < 0) << (slot % 8);
            }
        }
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(m, k, k, k, 0, a.data(),
                                                                          ap.data());
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(n, k, k, k, 0,
                                                                           codes.data(), bp.data());
    }

    void run(bool split = false, bool clamp = false)
    {
        const float lo = clamp ? -7.0f : -INFINITY;
        const float hi = clamp ? 9.0f : INFINITY;
        if (split && k > 32) {
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(
                m, n, 32, ap.data(), k, 0, bp.data(), k, 0, nullptr, n, d.data(), n, nullptr,
                -INFINITY, INFINITY, &params);
            const size_t ao =
                tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(0, 32, k,
                                                                                           m);
            const size_t bo =
                tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(0, 32, k,
                                                                                           n);
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(
                m, n, k - 32, ap.data() + ao, k, 32, bp.data() + bo, k, 32, d.data(), n, d.data(),
                n, bias.data(), lo, hi, &params);
        } else {
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv(
                m, n, k, ap.data(), k, 0, bp.data(), k, 0, nullptr, n, d.data(), n, bias.data(), lo,
                hi, &params);
        }
    }

    bool check(bool clamp) const
    {
        for (size_t row = 0; row < m; ++row)
            for (size_t col = 0; col < n; ++col) {
                int sum = 0;
                for (size_t kk = 0; kk < k; ++kk)
                    sum += int(a[row * k + kk]) * int(b[col * k + kk]);
                float expected = float(sum) * 0.0625f + bias[col];
                if (clamp)
                    expected = std::min(9.0f, std::max(-7.0f, expected));
                if (d[row * n + col] != expected) {
                    std::fprintf(stderr, "m=%zu n=%zu k=%zu row=%zu col=%zu got=%g expected=%g\n",
                                 m, n, k, row, col, d[row * n + col], expected);
                    return false;
                }
            }
        return true;
    }
};

TEST(GemmI8I5Rvv, NarrowAccumulationExtremes)
{
    const size_t nr = tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qsi5c32p_4x1vl_rvv();
    for (size_t m : {1u, 2u, 3u, 4u, 5u}) {
        for (size_t n : {size_t{1}, nr - 1, nr, nr + 1}) {
            for (size_t k : {32u, 64u, 576u}) {
                for (int pattern = 0; pattern < 4; ++pattern) {
                    Qsi5ExtremeData data(m, n, k, pattern);
                    for (bool split : {false, true}) {
                        for (bool clamp : {false, true}) {
                            data.run(split, clamp);
                            ASSERT_TRUE(data.check(clamp));
                        }
                    }
                }
            }
        }
    }
}
}  // namespace
