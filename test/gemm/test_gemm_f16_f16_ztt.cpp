//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <riscv_ztt.h>

#include <limits>

#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16/tqt_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16_f16_f16t/tqt_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16p_f16p_f16p/tqt_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt.h"
#include "src/gemm/gemm_f16_f16/gemm_1xnbias_clamp_f16p_f16p_f16p/tqt_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt.h"
#include "test/common/gemm_test_runner.h"

namespace
{
// Status bits are write-zero-to-clear; an unsupported operation sets UN.
__attribute__((noinline)) static void ztt_set_unsupported_status()
{
    asm volatile("msettyp m0, zero\n\tmadd.ew m0, m0, m0" : : : "memory");
}

}  // namespace

namespace ztt_f16_native
{
using namespace torq_tile::test;
static const tqt_gemm_1xnbias_clamp_f16_f16_f16_ukernel rowmajor = {
    tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_ztt,
};

static const tqt_gemm_1xnbias_clamp_f16_f16_f16t_ukernel transposed = {
    tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_ztt,
};

static const tqt_gemm_1xnbias_clamp_f16p_f16p_f16p_ukernel packed = {
    tqt_get_m_step_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_get_d_packed_size_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_get_bias_packed_size_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_run_a_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_run_b_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_run_bt_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_run_c_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_run_bias_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_run_d_unpack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
    tqt_run_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_ztt,
};

using T = float16_t;
constexpr bool wide = false;
TEST(ZttAdaptive_f16_native, Correctness)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    for (int layout : {0, 1, 2}) {
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
TEST(ZttAdaptive_f16_native, PackedTailsAndLayout)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    for (size_t m : {size_t(1), s + 1})
        for (size_t n : {size_t(1), step + 3})
            for (size_t k : {size_t(1), step + 1}) {
                run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, 2, m, n, k, true,
                                                     true, true);
            }
    run_square_packed_gemm_layout_test<T>(packed);
}
TEST(ZttAdaptive_f16_native, SpecialValues)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    const float ranges[][2] = {{-INFINITY, INFINITY}, {-1, 1}, {NAN, 1}, {-1, NAN}, {NAN, NAN},
                               {-0.0f, 0.0f},         {1, -1}};
    for (int layout : {0, 1, 2})
        for (const auto &r : ranges) {
            run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, 0,
                                                 true, false, false, r[0], r[1], true);
        }
}
TEST(ZttAdaptive_f16_native, GettersZeroAndOwnership)
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
    ASSERT_NE(__riscv_ztt_ame_acquire(0) & 1, 0u);
    for (int layout : {0, 1, 2}) {
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, step,
                                             true, true, false);
        EXPECT_NE(__riscv_ztt_get_ameown() & 1, 0u);
        EXPECT_EQ(__riscv_ztt_get_amestatus() & 1, 0u);
    }
    __riscv_ztt_ame_release();
}
TEST(ZttAdaptive_f16_nativeDeathTest, RejectsInvalidInputs)
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
    EXPECT_DEATH(packed.run_gemm(s, step, 0, nullptr, nullptr, nullptr, data.data(), nullptr,
                                 (T)-INFINITY, (T)INFINITY),
                 "");
    EXPECT_NE(__riscv_ztt_get_amestatus() & 1, 0u);
    __riscv_ztt_ame_release();
}
}  // namespace ztt_f16_native
namespace ztt_f16_f32acc
{
using namespace torq_tile::test;
static const tqt_gemm_1xnbias_clamp_f16_f16_f16_ukernel rowmajor = {
    tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt,
    tqt_run_gemm_1xnbias_clamp_f16_f16_f16_1nlenxpnlen_f32acc_ztt,
};

static const tqt_gemm_1xnbias_clamp_f16_f16_f16t_ukernel transposed = {
    tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt,
    tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_1nlenxpnlen_f32acc_ztt,
};

static const tqt_gemm_1xnbias_clamp_f16p_f16p_f16p_ukernel packed = {
    tqt_get_m_step_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_n_step_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_a_packed_size_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_d_packed_size_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_get_bias_packed_size_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_a_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_b_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_bt_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_c_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_bias_pack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_d_unpack_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
    tqt_run_gemm_1xnbias_clamp_f16p_f16p_f16p_1nlenxpnlen_f32acc_ztt,
};

using T = float16_t;
constexpr bool wide = true;
TEST(ZttAdaptive_f16_f32acc, Correctness)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    for (int layout : {0, 1, 2}) {
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
TEST(ZttAdaptive_f16_f32acc, PackedTailsAndLayout)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    for (size_t m : {size_t(1), s + 1})
        for (size_t n : {size_t(1), step + 3})
            for (size_t k : {size_t(1), step + 1}) {
                run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, 2, m, n, k, true,
                                                     true, true);
            }
    run_square_packed_gemm_layout_test<T>(packed);
}
TEST(ZttAdaptive_f16_f32acc, SpecialValues)
{
    const size_t s = packed.get_m_step(), step = packed.get_n_step();
    const float ranges[][2] = {{-INFINITY, INFINITY}, {-1, 1}, {NAN, 1}, {-1, NAN}, {NAN, NAN},
                               {-0.0f, 0.0f},         {1, -1}};
    for (int layout : {0, 1, 2})
        for (const auto &r : ranges) {
            run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, 0,
                                                 true, false, false, r[0], r[1], true);
        }
}
TEST(ZttAdaptive_f16_f32acc, GettersZeroAndOwnership)
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
    ASSERT_NE(__riscv_ztt_ame_acquire(0) & 1, 0u);
    for (int layout : {0, 1, 2}) {
        run_square_packed_gemm_test<T, wide>(rowmajor, transposed, packed, layout, s, step, step,
                                             true, true, false);
        EXPECT_NE(__riscv_ztt_get_ameown() & 1, 0u);
        EXPECT_EQ(__riscv_ztt_get_amestatus() & 1, 0u);
    }
    __riscv_ztt_ame_release();
}
TEST(ZttAdaptive_f16_f32accDeathTest, RejectsInvalidInputs)
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
    EXPECT_DEATH(packed.run_gemm(s, step, 0, nullptr, nullptr, nullptr, data.data(), nullptr,
                                 (T)-INFINITY, (T)INFINITY),
                 "");
    EXPECT_NE(__riscv_ztt_get_amestatus() & 1, 0u);
    __riscv_ztt_ame_release();
}
}  // namespace ztt_f16_f32acc

TEST(ZttAdaptive_f16, PrecisionAndSharedPacking)
{
    using namespace ztt_f16_native;
    torq_tile::test::run_gemm_accumulation_precision_test<T>(
        rowmajor, ztt_f16_f32acc::rowmajor, transposed, ztt_f16_f32acc::transposed, packed,
        ztt_f16_f32acc::packed, 2048.0f);
}
