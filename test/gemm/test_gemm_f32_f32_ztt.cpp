//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "torq_tile/tqt_compile_config.h"

#if TORQ_TILE_HAS_ZTT_GEMM_F32
#include <riscv_ztt.h>

#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32/tqt_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32_f32_f32t/tqt_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt.h"
#include "src/gemm/gemm_f32_f32/gemm_1xnbias_clamp_f32p_f32p_f32p/tqt_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt.h"
#include "test/common/gemm_test_runner.h"

namespace
{
// Status bits are write-zero-to-clear; an unsupported operation sets UN.
__attribute__((noinline)) static void ztt_set_unsupported_status()
{
    asm volatile("msettyp m0, zero\n\tmadd.ew m0, m0, m0" : : : "memory");
}

}  // namespace

namespace ztt_f32_native
{
using namespace torq_tile::test;
static const tqt_gemm_1xnbias_clamp_f32_f32_f32_ukernel rowmajor = {
    tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_f32_f32_f32_1nlenxpnlen_ztt,
};

static const tqt_gemm_1xnbias_clamp_f32_f32_f32t_ukernel transposed = {
    tqt_get_m_step_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_f32_f32_f32t_1nlenxpnlen_ztt,
};

static const tqt_gemm_1xnbias_clamp_f32p_f32p_f32p_ukernel packed = {
    tqt_get_m_step_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_get_d_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_get_bias_packed_size_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_run_a_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_run_b_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_run_bt_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_run_c_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_run_bias_pack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_run_d_unpack_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_f32p_f32p_f32p_1nlenxpnlen_ztt,
};

using T = float;
constexpr bool wide = false;
TEST(ZttGemm_f32, Correctness)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    for (int layout : {0, 1, 2}) {
        for (size_t m : {s, 2 * s}) {
            for (size_t n : {step, 3 * step}) {
                for (size_t k : {step, 2 * step, 3 * step}) {
                    run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, m, n,
                                                         k, false, false, false);
                }
            }
        }
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, step,
                                             false, false, false);
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, 2 * s, 2 * step,
                                             2 * step, true, true, false);
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, step,
                                             true, true, true, -0.25f, 0.5f);
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, 0, true,
                                             false, true);
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, 0,
                                             false, true, false);
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step,
                                             16 * step, false, false, false);
    }
}
TEST(ZttGemm_f32, PackedTailsAndLayout)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    for (size_t m : {size_t(1), s - 1, s, s + 1})
        for (size_t n : {size_t(1), step - 1, step, 2 * step + 3})
            for (size_t k : {size_t(1), step - 1, step, 2 * step + 1}) {
                run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, 2, m, n, k, true,
                                                     true, true);
            }
    run_square_packed_gemm_layout_test<T>(packed);
}
TEST(ZttGemm_f32, ResidualBiasClampAndAlias)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    for (int layout : {0, 1, 2}) {
        for (bool residual : {false, true}) {
            for (bool bias : {false, true}) {
                for (bool bounded : {false, true}) {
                    const size_t tail = layout == 2 ? 1 : 0;
                    run_square_packed_gemm_test<T, wide>(
                        rowmajor, transposed, packed, layout, s + tail, 2 * step + tail,
                        3 * step + tail, residual, bias, residual, bounded ? -0.25f : -INFINITY,
                        bounded ? 0.5f : INFINITY);
                }
            }
        }
    }
}
TEST(ZttGemm_f32, OffsetsAndAlignedSubwindows)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step(), ld = 4 * step + 5;
    constexpr float sentinel = -12345.0f;
    std::vector<T> a(3 * s * ld, 1.0f), b(3 * step * ld, 2.0f), d(3 * s * ld, sentinel);
    const T *ap = a.data() + rowmajor.get_a_offset(s, step, ld) / sizeof(T);
    for (bool bt : {false, true}) {
        std::fill(d.begin(), d.end(), sentinel);
        const size_t offset = bt ? transposed.get_b_offset(2 * step, step, ld)
                                 : rowmajor.get_b_offset(2 * step, step, ld);
        EXPECT_EQ(offset, (bt ? 2 * step * ld + step : step * ld + 2 * step) * sizeof(T));
        const T *bp = b.data() + offset / sizeof(T);
        T *dp = d.data() + rowmajor.get_d_offset(s, 2 * step, ld) / sizeof(T);
        if (bt) {
            transposed.run_gemm(s, step, step, ap, ld, step, bp, ld, step, nullptr, ld, dp, ld,
                                nullptr, -INFINITY, INFINITY);
        } else {
            rowmajor.run_gemm(s, step, step, ap, ld, step, bp, ld, step, nullptr, ld, dp, ld,
                              nullptr, -INFINITY, INFINITY);
        }
        for (size_t i = 0; i < 3 * s; ++i) {
            for (size_t j = 0; j < ld; ++j) {
                EXPECT_EQ(d[i * ld + j], i >= s && i < 2 * s && j >= 2 * step && j < 3 * step
                                             ? 2.0f * step
                                             : sentinel);
            }
        }
    }
}
TEST(ZttGemm_f32, SpecialValues)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    const float ranges[][2] = {
        {-INFINITY, INFINITY}, {-1, 1}, {NAN, 1},      {-1, NAN},     {NAN, NAN},
        {-0.0f, 0.0f},         {1, -1}, {0, INFINITY}, {-INFINITY, 0}};
    for (int layout : {0, 1, 2})
        for (const auto &r : ranges) {
            run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, 0,
                                                 true, false, false, r[0], r[1], true);
        }
}
TEST(ZttGemm_f32, GettersZeroAndOwnership)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step(), ld = 3 * step + 5;
    EXPECT_EQ(step, s * std::max(size_t(1), __riscv_ztt_get_ameudsz() / (8 * sizeof(T))));
    EXPECT_EQ(rowmajor.get_n_step(), step);
    EXPECT_EQ(transposed.get_n_step(), step);
    EXPECT_EQ(rowmajor.get_a_offset(s, step, ld), (s * ld + step) * sizeof(T));
    EXPECT_EQ(rowmajor.get_b_offset(step, step, ld), (step * ld + step) * sizeof(T));
    EXPECT_EQ(transposed.get_b_offset(step, 0, ld), step * ld * sizeof(T));
    EXPECT_EQ(rowmajor.get_c_offset(s, step, ld), rowmajor.get_d_offset(s, step, ld));
    EXPECT_EQ(rowmajor.get_bias_offset(step), step * sizeof(T));
    EXPECT_EQ(rowmajor.get_d_size(s, step), s * step * sizeof(T));
    EXPECT_EQ(packed.get_a_packed_size(0, step), 0u);
    EXPECT_EQ(packed.get_b_packed_size(step, 0), 0u);
    EXPECT_EQ(packed.get_d_packed_size(s, 0), 0u);
    EXPECT_EQ(packed.get_bias_packed_size(0), 0u);
    packed.run_a_pack(0, step, 0, nullptr, nullptr);
    packed.run_b_pack(0, step, 0, nullptr, nullptr);
    packed.run_bt_pack(0, step, 0, nullptr, nullptr);
    packed.run_c_pack(0, step, 0, nullptr, nullptr);
    packed.run_bias_pack(0, nullptr, nullptr);
    packed.run_d_unpack(0, step, 0, nullptr, nullptr);
    packed.run_gemm(0, step, step, nullptr, nullptr, nullptr, nullptr, nullptr, (T)-INFINITY,
                    (T)INFINITY);
    for (int layout : {0, 1, 2}) {
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, 0, step, step,
                                             false, false, false);
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, 0, step,
                                             false, false, false);
    }
    EXPECT_EQ(__riscv_ztt_get_ameown() & 1, 0u);
    for (int iteration = 0; iteration < 2; ++iteration) {
        for (int layout : {0, 1, 2}) {
            run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step,
                                                 step, true, true, false);
            EXPECT_EQ(__riscv_ztt_get_ameown() & 1, 0u);
        }
    }
    ASSERT_NE(__riscv_ztt_ame_acquire(0) & 1, 0u);
    for (int layout : {0, 1, 2}) {
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, step,
                                             true, true, false);
        EXPECT_NE(__riscv_ztt_get_ameown() & 1, 0u);
        EXPECT_EQ(__riscv_ztt_get_amestatus() & 1, 0u);
    }
    __riscv_ztt_ame_release();
}
TEST(ZttGemm_f32DeathTest, RejectsInvalidInputs)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    std::vector<T> data(4 * step * step);
    EXPECT_DEATH(
        rowmajor.run_gemm(s + 1, step, step, data.data(), step, 0, data.data(), step, 0, nullptr,
                          step, data.data(), step, nullptr, (T)-INFINITY, (T)INFINITY),
        "");
    EXPECT_DEATH(
        rowmajor.run_gemm(s, step + 1, step, data.data(), step, 0, data.data(), step, 0, nullptr,
                          step, data.data(), step, nullptr, (T)-INFINITY, (T)INFINITY),
        "");
    EXPECT_DEATH(
        transposed.run_gemm(s, step, step + 1, data.data(), step, 0, data.data(), step, 0, nullptr,
                            step, data.data(), step, nullptr, (T)-INFINITY, (T)INFINITY),
        "");
    EXPECT_DEATH(packed.get_a_packed_size(SIZE_MAX, step), "");
    EXPECT_DEATH(packed.get_b_packed_size(step, SIZE_MAX), "");
    EXPECT_DEATH(packed.run_a_pack(s, step, SIZE_MAX, data.data(), data.data() + step), "");
    EXPECT_DEATH(packed.run_d_unpack(s, step, SIZE_MAX, data.data(), data.data() + step), "");
    EXPECT_DEATH(rowmajor.get_a_offset(s, step, SIZE_MAX), "");
    ASSERT_NE(__riscv_ztt_ame_acquire(0) & 1, 0u);
    ztt_set_unsupported_status();
    ASSERT_NE(__riscv_ztt_get_amestatus() & 1, 0u);
    EXPECT_DEATH(rowmajor.run_gemm(s, step, 0, nullptr, 0, 0, nullptr, 0, 0, nullptr, 0,
                                   data.data(), step, nullptr, -INFINITY, INFINITY),
                 "");
    EXPECT_DEATH(transposed.run_gemm(s, step, 0, nullptr, 0, 0, nullptr, 0, 0, nullptr, 0,
                                     data.data(), step, nullptr, -INFINITY, INFINITY),
                 "");
    EXPECT_DEATH(packed.run_gemm(s, step, 0, nullptr, nullptr, nullptr, data.data(), nullptr,
                                 (T)-INFINITY, (T)INFINITY),
                 "");
    EXPECT_NE(__riscv_ztt_get_amestatus() & 1, 0u);
    EXPECT_NE(__riscv_ztt_get_ameown() & 1, 0u);
    __riscv_ztt_ame_release();
}
}  // namespace ztt_f32_native

#endif
