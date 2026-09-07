//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_u5/gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_u8/gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

static tqt_qsi8d256_qau5c32_params make_qsi8d256_qau5c32_test_params(
    const float *scale_a, size_t ldsa, const int16_t *bsums_a, size_t ldbsuma, const float *scale_b,
    const float *real_zero_b, size_t ldsb)
{
    return {scale_a, ldsa, bsums_a, ldbsuma, scale_b, real_zero_b, ldsb};
}

static tqt_qsi8d256_qau8c32_params make_qsi8d256_qau8c32_test_params(
    const float *scale_a, size_t ldsa, const int16_t *bsums_a, size_t ldbsuma, const float *scale_b,
    const float *real_zero_b, size_t ldsb)
{
    return {scale_a, ldsa, bsums_a, ldbsuma, scale_b, real_zero_b, ldsb};
}

static PackedQsi8BlockQuantizedGemmTestFuncs<tqt_qsi8d256_qau5c32_params>
make_qsi8d256_qau5c32_test_funcs()
{
    return {
        256,
        32,
        true,
        31,
        Qsi8BlockWeightEncoding::Qau5,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv,
        make_qsi8d256_qau5c32_test_params,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

static PackedQsi8BlockQuantizedGemmTestFuncs<tqt_qsi8d256_qau8c32_params>
make_qsi8d256_qau8c32_test_funcs()
{
    return {
        256,
        32,
        true,
        31,
        Qsi8BlockWeightEncoding::Qau8,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv,
        make_qsi8d256_qau8c32_test_params,
        clamp_test_run<tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau8c32p_4x1vl_rvv, float, 13, 11,
                       12, 14, ClampSource::kColumnBias, false, 5, true>(),
    };
}

TEST(GemmI8U5Rvv, Qau5c32PlanePacking)
{
    static constexpr int8_t values[] = {0, 15, 16, 31};
    auto data = make_qsi8_block_quantized_gemm_test_data(1, 256, 256, 32, true, 31);
    for (size_t k = 0; k < data.k; ++k)
        data.b[k] = values[k % 4];
    const std::vector<uint8_t> packed = pack_qau5c32_source(data);
    for (size_t k = 0; k < data.k; ++k)
        EXPECT_EQ(unpack_qau5c32_source(packed.data(), k), data.b[k]) << "k=" << k;
}

TEST(GemmI8U5Rvv, Qsi8d256Qau5c32)
{
    const auto funcs = make_qsi8d256_qau5c32_test_funcs();
    const auto expanded_funcs = make_qsi8d256_qau8c32_test_funcs();
    const size_t n_step = funcs.get_n_step();
    for (size_t m = 1; m <= 5; ++m) {
        run_packed_qsi8_block_quantized_gemm_test(m, n_step + 3, 512, m == 5, funcs,
                                                  expanded_funcs);
    }
    for (size_t n : {size_t{1}, n_step - 1, n_step, n_step + 3}) {
        run_packed_qsi8_block_quantized_gemm_test(5, n, 256, false, funcs, expanded_funcs);
    }
}

TEST(GemmI8U5Rvv, ClampSpecialValues_Qsi8d256Qau5c32)
{
    ScopedGemmClampProbe probe;
    const auto funcs = make_qsi8d256_qau5c32_test_funcs();
    const auto expanded_funcs = make_qsi8d256_qau8c32_test_funcs();
    const size_t n_step = funcs.get_n_step();
    for (size_t m = 1; m <= 5; ++m) {
        run_packed_qsi8_block_quantized_gemm_test(m, n_step + 3, 512, true, funcs, expanded_funcs);
    }

    for (size_t rows = 1;
         rows <= tqt_get_m_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv() + 1; ++rows) {
        for (size_t columns :
             {size_t{1}, tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv(),
              tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau5c32p_4x1vl_rvv() + 3}) {
            run_packed_qsi8_block_quantized_gemm_test(rows, columns, 256, false, funcs,
                                                      expanded_funcs);
        }
    }
}
