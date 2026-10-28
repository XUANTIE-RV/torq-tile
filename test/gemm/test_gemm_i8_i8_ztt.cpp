//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <riscv_ztt.h>

#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/quantized_gemm_test_runner.h"
#include "torq_tile/tqt_compile_config.h"

using namespace torq_tile::test;

static tqt_qsi8d256_qsi8c16_params make_params(const float *sa, size_t ldsa, const int16_t *sums,
                                               size_t ldsums, const float *sb, const float *zb,
                                               size_t ldsb)
{
    (void)sums;
    (void)ldsums;
    (void)zb;
    return {sa, ldsa, sb, ldsb};
}

static PackedQsi8BlockQuantizedGemmTestFuncs<tqt_qsi8d256_qsi8c16_params> make_funcs()
{
    return {256,
            16,
            false,
            31,
            Qsi8BlockWeightEncoding::Qsi8,
            tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
            tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
            tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
            tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
            tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
            make_params,
            clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt, float,
                           13, 11, 12, 14, ClampSource::kColumnBias, false, 5, true>(),
            Qsi8BlockPackedLayout::ZttSquare,
            true};
}

TEST(GemmI8Ztt, TailsAndSplitK)
{
    const auto funcs = make_funcs();
    const size_t s = funcs.get_n_step();
    for (size_t m : {size_t(1), std::max(size_t(1), s - 1), s, s + 1})
        run_packed_qsi8_block_quantized_gemm_test(m, s + 1, 512, funcs);
}

TEST(GemmI8Ztt, KBoundaries)
{
    const auto funcs = make_funcs();
    for (size_t k : {256, 1024, 2048, 3072})
        run_packed_qsi8_block_quantized_gemm_test(1, 1, k, funcs);
}

TEST(GemmI8Ztt, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    run_packed_qsi8_block_quantized_gemm_test(1, 1, 256, make_funcs());
}

TEST(GemmI8Ztt, CallerContracts)
{
    run_qsi8_block_ztt_contract_test(make_funcs());
}

TEST(GemmI8Ztt, ExactIntegerGroups)
{
    run_qsi8_block_ztt_integer_test(make_funcs());
}

TEST(GemmI8Ztt, ModelPackingAndSlices)
{
    run_qsi8_block_ztt_model_slices(make_funcs());
}

TEST(GemmI8Ztt, NlenBoundary)
{
    auto funcs = make_funcs();
    funcs.test_clamp = false;
    run_packed_qsi8_block_quantized_gemm_test(1, 1, 256, funcs);
}

TEST(GemmI8Ztt, Ownership)
{
    EXPECT_EQ(__riscv_ztt_get_ameown() & 1, 0u);
    ASSERT_NE(__riscv_ztt_ame_acquire(0) & 1, 0u);
    auto funcs = make_funcs();
    funcs.test_clamp = false;
    run_packed_qsi8_block_quantized_gemm_test(1, 1, 256, funcs);
    EXPECT_NE(__riscv_ztt_get_ameown() & 1, 0u);
    EXPECT_EQ(__riscv_ztt_get_amestatus() & 1, 0u);
    __riscv_ztt_ame_release();
    EXPECT_EQ(__riscv_ztt_get_ameown() & 1, 0u);
}

__attribute__((noinline)) static void set_unsupported_status()
{
    asm volatile("msettyp m0, zero\n\tmadd.ew m0, m0, m0" : : : "memory");
}

TEST(GemmI8ZttDeathTest, InvalidInputsAndStickyStatus)
{
    const auto funcs = make_funcs();
    float d = 0;
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_DEATH(funcs.get_a_packed_size(SIZE_MAX, 256), "");
    EXPECT_DEATH(funcs.get_b_packed_size(1, 257), "");
    EXPECT_DEATH(funcs.run_gemm(1, 1, 256, nullptr, 256, 0, nullptr, 256, 256, nullptr, 1, &d, 1,
                                nullptr, -inf, inf, nullptr),
                 "");
    ASSERT_NE(__riscv_ztt_ame_acquire(0) & 1, 0u);
    set_unsupported_status();
    ASSERT_NE(__riscv_ztt_get_amestatus() & 1, 0u);
    EXPECT_DEATH(funcs.run_gemm(1, 1, 0, nullptr, 0, 0, nullptr, 0, 0, nullptr, 1, &d, 1, nullptr,
                                -inf, inf, nullptr),
                 "");
    EXPECT_NE(__riscv_ztt_get_ameown() & 1, 0u);
    EXPECT_NE(__riscv_ztt_get_amestatus() & 1, 0u);
    __riscv_ztt_ame_release();
}

TEST(GemmI8Ztt, NTails)
{
    auto funcs = make_funcs();
    funcs.test_clamp = false;
    const size_t s = funcs.get_n_step();
    for (size_t n : {std::max(size_t(1), s - 1), s})
        run_packed_qsi8_block_quantized_gemm_test(1, n, 256, funcs);
}
