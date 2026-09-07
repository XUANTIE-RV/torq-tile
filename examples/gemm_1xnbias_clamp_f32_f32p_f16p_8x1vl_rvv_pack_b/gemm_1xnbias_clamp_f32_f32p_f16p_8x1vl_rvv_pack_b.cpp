//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for RISC-V with RVV F16 widening support.
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "src/common/tqt_common.h"
#include "tqt_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv.h"
#include "tqt_gemm_1xnbias_clamp_f32_f32p_f16p_interface.h"

namespace
{

constexpr size_t kM = 17;
constexpr size_t kN = 19;
constexpr size_t kK = 31;

template <typename T>
void fill_random(std::vector<T> &data, float low, float high)
{
    for (T &value : data) {
        const float unit = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        value = static_cast<T>(low + (high - low) * unit);
    }
}

template <typename WeightT, bool NarrowA, bool TransposedB>
void run_reference(const std::vector<float> &A, const std::vector<WeightT> &B,
                   const std::vector<float> &C, const std::vector<float> &bias,
                   std::vector<float> &D)
{
    for (size_t i = 0; i < kM; ++i) {
        for (size_t j = 0; j < kN; ++j) {
            float acc = C[i * kN + j] + bias[j];
            for (size_t p = 0; p < kK; ++p) {
                const float a = NarrowA ? static_cast<float>(static_cast<WeightT>(A[i * kK + p]))
                                        : A[i * kK + p];
                const size_t b_idx = TransposedB ? j * kK + p : p * kN + j;
                acc += a * static_cast<float>(B[b_idx]);
            }
            D[i * kN + j] = std::min(4.0f, std::max(-3.0f, acc));
        }
    }
}

bool check_result(const char *name, const std::vector<float> &reference,
                  const std::vector<float> &actual)
{
    double dot = 0.0;
    double reference_norm = 0.0;
    double actual_norm = 0.0;
    float max_error = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        dot += static_cast<double>(reference[i]) * actual[i];
        reference_norm += static_cast<double>(reference[i]) * reference[i];
        actual_norm += static_cast<double>(actual[i]) * actual[i];
        max_error = std::max(max_error, std::fabs(reference[i] - actual[i]));
    }
    const double cosine = dot / std::sqrt(reference_norm * actual_norm);
    const bool passed = max_error <= 1.0e-3f && cosine >= 0.99999;
    std::printf("TEST[%s]\n", name);
    std::printf("- Max absolute error: %.8f\n", max_error);
    std::printf("- Cosine similarity: %.8f\n", cosine);
    std::printf("- Status: %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

template <typename WeightT, bool NarrowA, bool TransposedB, typename Ukernel>
int run_packed(const char *name, const Ukernel &ukernel)
{
    std::srand(1);
    std::vector<float> A(kM * kK);
    std::vector<WeightT> B(TransposedB ? kN * kK : kK * kN);
    std::vector<float> C(kM * kN);
    std::vector<float> bias(kN);
    std::vector<float> reference(kM * kN);
    std::vector<float> actual(kM * kN);
    fill_random(A, -1.0f, 1.0f);
    fill_random(B, -1.0f, 1.0f);
    fill_random(C, -0.5f, 0.5f);
    fill_random(bias, -0.25f, 0.25f);
    run_reference<WeightT, NarrowA, TransposedB>(A, B, C, bias, reference);

    std::vector<unsigned char> A_packed(ukernel.get_a_packed_size(kM, kK));
    std::vector<unsigned char> B_packed(ukernel.get_b_packed_size(kN, kK));
    ukernel.run_a_pack(kM, kK, kK, kK, 0, A.data(), A_packed.data());
    if (TransposedB) {
        ukernel.run_bt_pack(kN, kK, kK, kK, 0, B.data(), B_packed.data());
    } else {
        ukernel.run_b_pack(kN, kK, kN, kK, 0, B.data(), B_packed.data());
    }
    ukernel.run_gemm(kM, kN, kK, A_packed.data(), kK, 0, B_packed.data(), kK, 0, C.data(), kN,
                     actual.data(), kN, bias.data(), -3.0f, 4.0f);
    return check_result(name, reference, actual) ? 0 : 1;
}

}  // namespace

int main()
{
    const tqt_gemm_1xnbias_clamp_f32_f32p_f16p_ukernel ukernel{
        tqt_get_m_step_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_a_packed_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_a_pack_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_b_pack_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_bt_pack_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv};
    return run_packed<float16_t, false, false>("gemm_1xnbias_clamp_f32_f32p_f16p_8x1vl_rvv_pack_b",
                                               ukernel);
}
