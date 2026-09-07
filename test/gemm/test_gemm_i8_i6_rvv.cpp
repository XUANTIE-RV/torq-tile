//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/gemm/gemm_i8_i6/gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv.h"
#include "test/common/quantized_gemm_test_runner.h"

using namespace torq_tile::test;

static tqt_qsi8d256_qsi6c16_params make_qsi8d256_qsi6c16_test_params(
    const float *scale_a, size_t ldsa, const int16_t *bsums_a, size_t ldbsuma, const float *scale_b,
    const float *, size_t ldsb)
{
    return {scale_a, ldsa, bsums_a, ldbsuma, scale_b, ldsb};
}

static tqt_qsi8d256_qsi8c16_params make_qsi8d256_qsi8c16_test_params(const float *scale_a,
                                                                     size_t ldsa, const int16_t *,
                                                                     size_t, const float *scale_b,
                                                                     const float *, size_t ldsb)
{
    return {scale_a, ldsa, scale_b, ldsb};
}

static PackedQsi8BlockQuantizedGemmTestFuncs<tqt_qsi8d256_qsi6c16_params>
make_qsi8d256_qsi6c16_test_funcs()
{
    return {
        256,
        16,
        false,
        31,
        Qsi8BlockWeightEncoding::Qsi6,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
        make_qsi8d256_qsi6c16_test_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi6c16p_4x1vl_rvv,
    };
}

static PackedQsi8BlockQuantizedGemmTestFuncs<tqt_qsi8d256_qsi8c16_params>
make_qsi8d256_qsi8c16_test_funcs()
{
    return {
        256,
        16,
        false,
        31,
        Qsi8BlockWeightEncoding::Qsi8,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv,
        make_qsi8d256_qsi8c16_test_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_4x1vl_rvv,
    };
}

TEST(GemmI8I6Rvv, Qsi6c16PlanePacking)
{
    static constexpr int8_t values[] = {-32, -17, -16, -1, 0, 15, 16, 31};
    auto data = make_qsi8_block_quantized_gemm_test_data(1, 256, 256, 16, false, 31);
    for (size_t k = 0; k < data.k; ++k)
        data.b[k] = values[k % 8];
    const std::vector<uint8_t> packed = pack_qsi6c16_source(data);
    for (size_t k = 0; k < data.k; ++k)
        EXPECT_EQ(unpack_qsi6c16_source(packed.data(), k), data.b[k]) << "k=" << k;
}

TEST(GemmI8I6Rvv, Qsi8d256Qsi6c16)
{
    const auto funcs = make_qsi8d256_qsi6c16_test_funcs();
    const auto expanded_funcs = make_qsi8d256_qsi8c16_test_funcs();
    const size_t n_step = funcs.get_n_step();
    for (size_t m = 1; m <= 5; ++m) {
        run_packed_qsi8_block_quantized_gemm_test(m, n_step + 3, 512, m == 5, funcs,
                                                  expanded_funcs);
    }
    for (size_t n : {size_t{1}, n_step - 1, n_step, n_step + 3}) {
        run_packed_qsi8_block_quantized_gemm_test(5, n, 256, false, funcs, expanded_funcs);
    }
}
