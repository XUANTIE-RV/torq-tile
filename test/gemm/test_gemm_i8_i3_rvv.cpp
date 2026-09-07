//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_i3/gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i4/gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
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
            clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv, float, 13,
                           11, 12, 14, ClampSource::kColumnBias, false, 5, true>()};
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
            clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi4c16p_4x1vl_rvv, float, 13,
                           11, 12, 14, ClampSource::kColumnBias, false, 5, true>()};
}

TEST(GemmI8I3Rvv, CompactExpandedTailsAndSplitK)
{
    const auto compact = make_qsi3c16_funcs();
    const auto expanded = make_qsi4c16_rvv_funcs();
    run_packed_qsi8_kblock_gemm_equivalence_test(5, compact.get_n_step() + 3, 512, compact,
                                                 expanded);
}

TEST(GemmI8I3Rvv, ClampSpecialValues_CompactExpandedTailsAndSplitK)
{
    ScopedGemmClampProbe probe;
    const auto compact = make_qsi3c16_funcs();
    const auto expanded = make_qsi4c16_rvv_funcs();
    for (size_t rows = 1;
         rows <= tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv() + 1; ++rows) {
        for (size_t columns :
             {size_t{1}, tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv(),
              tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv() + 3}) {
            run_packed_qsi8_kblock_gemm_equivalence_test(rows, columns, 512, compact, expanded,
                                                         false);
        }
    }
    // Small first panels expose a pre-existing split-K origin bug in these RVV kernels.
    // Keep their full-K probes above and exercise split-K after a complete first panel.
    const size_t m_step = tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv();
    const size_t n_step = tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi3c16p_4x1vl_rvv();
    for (size_t tail = 1; tail <= m_step; ++tail) {
        for (size_t n_tail : {size_t{1}, size_t{3}, n_step}) {
            run_packed_qsi8_kblock_gemm_equivalence_test(m_step + tail, n_step + n_tail, 512,
                                                         compact, expanded);
        }
    }
}
