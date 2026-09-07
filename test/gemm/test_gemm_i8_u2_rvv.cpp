//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_u2/gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_u2/gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p/tqt_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_u4/gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv.h"
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
            tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv,
            make_qau2c16_params,
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau2c16p_4x1vl_rvv};
}

static PackedQsi8KBlockGemmTestFuncs<tqt_qsi8d256_qau4c16_params> make_qau4c16_rvv_funcs()
{
    return {true,
            Qsi8KBlockWeightEncoding::Qau4,
            tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv,
            make_qau4c16_params,
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c16p_4x1vl_rvv};
}

TEST(GemmI8U2Rvv, CompactExpandedTailsAndSplitK)
{
    const auto compact = make_qau2c16_funcs();
    const auto expanded = make_qau4c16_rvv_funcs();
    run_packed_qsi8_kblock_gemm_equivalence_test(5, compact.get_n_step() + 3, 512, compact,
                                                 expanded);
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
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d32p_qau2z1c64p_4x1vl_rvv};
    run_packed_qsi8_qx0_source_test(5, funcs.get_n_step() + 3, 128, funcs);
}
