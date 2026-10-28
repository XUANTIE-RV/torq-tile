//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/common/gemm_benchmark_runner.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt.h"

namespace torq_tile
{
namespace benchmark
{
static tqt_qsi8d256_qsi8c16_params make_params(const float *sa, size_t ldsa, const int16_t *sums,
                                               size_t ldsums, const float *sb, const float *zb,
                                               size_t ldsb)
{
    (void)sums;
    (void)ldsums;
    (void)zb;
    return {sa, ldsa, sb, ldsb};
}

void register_gemm_i8_i8_ztt_benchmarks()
{
    const PackedQsi8BlockQuantizedGemmBenchmarkFuncs<tqt_qsi8d256_qsi8c16_params> funcs = {
        256,
        16,
        Qsi8BlockWeightEncoding::Qsi8,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
        make_params,
        tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt};
    register_packed_qsi8_block_quantized_gemm_benchmark(
        "tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt", funcs);
}
}  // namespace benchmark
}  // namespace torq_tile
