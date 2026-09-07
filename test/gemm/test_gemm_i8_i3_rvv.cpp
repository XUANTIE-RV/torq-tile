//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_i3/gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

static tqt_qsi8d256_qsi3c16_params make_qsi3c16_params(const Qsi8KBlockGemmTestData &data)
{
    return {data.scale_a.data(),       data.k / 256, data.qscale_b.data(), data.k / 16,
            data.super_scale_b.data(), data.k / 256};
}

static tqt_qsi8d256_qsi4c16_params make_qsi4c16_params(const Qsi8KBlockGemmTestData &data)
{
    return {data.scale_a.data(),       data.k / 256, data.qscale_b.data(), data.k / 16,
            data.super_scale_b.data(), data.k / 256};
}

static PackedQsi8KBlockGemmTestFuncs<tqt_qsi8d256_qsi3c16_params> make_qsi3c16_funcs()
{
    return {false,
            Qsi8KBlockWeightEncoding::Qsi3,
            tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
            tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
            tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
            tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
            tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv,
            make_qsi3c16_params,
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv};
}

static PackedQsi8KBlockGemmTestFuncs<tqt_qsi8d256_qsi4c16_params> make_qsi4c16_rvv_funcs()
{
    return {false,
            Qsi8KBlockWeightEncoding::Qsi4,
            tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
            tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
            tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
            tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
            tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
            tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
            tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv,
            make_qsi4c16_params,
            tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv};
}

TEST(GemmI8I3Rvv, CompactExpandedTailsAndSplitK)
{
    const auto compact = make_qsi3c16_funcs();
    const auto expanded = make_qsi4c16_rvv_funcs();
    run_packed_qsi8_kblock_gemm_equivalence_test(5, compact.get_n_step() + 3, 512, compact,
                                                 expanded);
}
