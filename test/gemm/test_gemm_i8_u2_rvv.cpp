//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_u2/gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_u2/gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_u4/gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

static tqt_qsi8d256_qau2c16_params make_qau2c16_params(const Qsi8KBlockGemmTestData &data)
{
    return {data.scale_a.data(),
            data.k / 256,
            data.bsums_a.data(),
            data.k / 16,
            reinterpret_cast<const uint8_t *>(data.qscale_b.data()),
            data.qmin_b.data(),
            data.k / 16,
            data.super_scale_b.data(),
            data.super_min_b.data(),
            data.k / 256};
}

static tqt_qsi8d256_qau4c16_params make_qau4c16_params(const Qsi8KBlockGemmTestData &data)
{
    return {data.scale_a.data(),
            data.k / 256,
            data.bsums_a.data(),
            data.k / 16,
            reinterpret_cast<const uint8_t *>(data.qscale_b.data()),
            data.qmin_b.data(),
            data.k / 16,
            data.super_scale_b.data(),
            data.super_min_b.data(),
            data.k / 256};
}

static PackedQsi8KBlockGemmTestFuncs<tqt_qsi8d256_qau2c16_params> make_qau2c16_funcs()
{
    return {true,
            Qsi8KBlockWeightEncoding::Qau2,
            Qsi8BlockPackedLayout::NInterleaved,
            tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            make_qau2c16_params,
            clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv, float, 13,
                           11, 12, 14, ClampSource::kColumnBias, false, 5, true>()};
}

static PackedQsi8KBlockGemmTestFuncs<tqt_qsi8d256_qau4c16_params> make_qau4c16_rvv_funcs()
{
    return {true,
            Qsi8KBlockWeightEncoding::Qau4,
            Qsi8BlockPackedLayout::NInterleaved,
            tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            make_qau4c16_params,
            clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv, float, 13,
                           11, 12, 14, ClampSource::kColumnBias, false, 5, true>()};
}

TEST(GemmI8U2Rvv, CompactExpandedTailsAndSplitK)
{
    const auto compact = make_qau2c16_funcs();
    const auto expanded = make_qau4c16_rvv_funcs();
    for (size_t rows = 1; rows <= 5; ++rows) {
        for (size_t columns : {size_t{1}, compact.get_n_step(), compact.get_n_step() + 3}) {
            run_packed_qsi8_kblock_gemm_equivalence_test(rows, columns, 512, compact, expanded);
        }
    }
}

TEST(GemmI8U2Rvv, ClampSpecialValues_CompactExpandedTailsAndSplitK)
{
    ScopedGemmClampProbe probe;
    const auto compact = make_qau2c16_funcs();
    const auto expanded = make_qau4c16_rvv_funcs();
    for (size_t rows = 1;
         rows <= tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv() + 1; ++rows) {
        for (size_t columns :
             {size_t{1}, tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv(),
              tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv() + 3}) {
            run_packed_qsi8_kblock_gemm_equivalence_test(rows, columns, 512, compact, expanded);
        }
    }
    const size_t m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv();
    const size_t n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv();
    for (size_t tail = 1; tail <= m_step; ++tail) {
        for (size_t n_tail : {size_t{1}, size_t{3}, n_step}) {
            run_packed_qsi8_kblock_gemm_equivalence_test(m_step + tail, n_step + n_tail, 512,
                                                         compact, expanded);
        }
    }
}

static tqt_qsi8d32_qau2z1c64_params make_qsi8d32_qau2z1c64_params(const float *scale_a, size_t ldsa,
                                                                  const float *scale_b, size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

TEST(GemmI8U2Rvv, Q2_0SourceAwareTailsAndSplitK)
{
    const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qau2z1c64_params> funcs = {
        Qx0SourceEncoding::U2Z1,
        2,
        64,
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        make_qsi8d32_qau2z1c64_params,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>()};
    run_packed_qsi8_qx0_source_test(5, funcs.get_n_step() + 3, 128, funcs);
}

TEST(GemmI8U2Rvv, ClampSpecialValues_Q2_0SourceAwareTailsAndSplitK)
{
    ScopedGemmClampProbe probe;
    const PackedQsi8Qx0GemmTestFuncs<tqt_qsi8d32_qau2z1c64_params> funcs = {
        Qx0SourceEncoding::U2Z1,
        2,
        64,
        tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv,
        make_qsi8d32_qau2z1c64_params,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>()};
    for (size_t rows = 1;
         rows <= tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv() + 1;
         ++rows) {
        for (size_t columns :
             {size_t{1}, tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv(),
              tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv() + 3}) {
            run_packed_qsi8_qx0_source_test(rows, columns, 128, funcs);
        }
    }
}

TEST(GemmI8U2Rvv, FullParameterRangeStridesAndInPlaceSplitK)
{
    const auto funcs = make_qau2c16_funcs();
    const size_t nr = funcs.get_n_step();
    for (size_t m : {1u, 2u, 3u, 4u, 5u, 8u, 128u}) {
        for (size_t n : {size_t{1}, nr - 1, nr, nr + 1, 2 * nr + 3}) {
            for (size_t k : {256u, 512u, 768u, 1024u}) {
                SCOPED_TRACE(::testing::Message() << "m=" << m << " n=" << n << " k=" << k);
                run_packed_qsi8_kblock_gemm_extreme_test(m, n, k, funcs);
            }
        }
    }
}

TEST(GemmI8U2Rvv, ZeroKPreservesBiasAndInPlaceC)
{
    run_packed_qsi8_kblock_gemm_zero_k_test(make_qau2c16_funcs());
}
