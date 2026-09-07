//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "src/gemm/gemm_u8_u8/gemm_mx1bias_1xnbias_clamp_f32_u8_u8t/tqt_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv.h"
#include "src/gemm/gemm_u8_u8/gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p/tqt_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv.h"
#include "src/gemm/gemm_u8_u8/gemm_mx1bias_1xnbias_i32_u8_u8t/tqt_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv.h"
#include "src/gemm/gemm_u8_u8/gemm_mx1bias_1xnbias_i32_u8p_u8p/tqt_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv.h"
#include "src/gemm/gemm_u8_u8/tqt_params_u8_u8.h"
#include "test/common/gemm_clamp_test_utils.h"
#include "test/common/verify.h"

using namespace torq_tile::test;

// ============================================================================
// Test parameters
// ============================================================================

struct U8GemmTestParams
{
    size_t m;
    size_t n;
    size_t k;
    bool has_c;
    std::string name;

    U8GemmTestParams(size_t m_, size_t n_, size_t k_, std::string name_ = "")
        : m(m_), n(n_), k(k_), has_c(false), name(std::move(name_))
    {
    }

    U8GemmTestParams &set_c(bool v)
    {
        has_c = v;
        return *this;
    }
};

struct U8GemmTestParamNameGenerator
{
    std::string operator()(const testing::TestParamInfo<U8GemmTestParams> &info) const
    {
        const auto &p = info.param;
        std::string test_name =
            "M" + std::to_string(p.m) + "_N" + std::to_string(p.n) + "_K" + std::to_string(p.k);
        if (!p.name.empty()) {
            test_name += "_" + p.name;
        }
        return test_name;
    }
};

// ============================================================================
// Common test values
// ============================================================================

static auto u8_common_test_values()
{
    return testing::Values(U8GemmTestParams(32, 32, 32),   //
                           U8GemmTestParams(3, 8, 11),     //
                           U8GemmTestParams(5, 11, 10),    //
                           U8GemmTestParams(9, 19, 23),    //
                           U8GemmTestParams(15, 32, 25),   //
                           U8GemmTestParams(60, 63, 127),  //
                           U8GemmTestParams(32, 32, 32, "WithC").set_c(true),
                           U8GemmTestParams(60, 63, 127, "WithC").set_c(true));
}

static auto clamp_u8_dimension_values()
{
    const testing::internal::ParamGenerator<U8GemmTestParams> original = u8_common_test_values();
    std::vector<U8GemmTestParams> cases;
    for (const auto &item : original) {
        cases.push_back(item);
    }
    auto seed = cases.front();
    seed.name = "ClampDimensions";
    const size_t ms = tqt_get_m_step_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv();
    const size_t ns = tqt_get_n_step_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv();
    std::set<size_t> rows{size_t{1}, ms, ms + 1};
    for (size_t m = 1; m <= ms; ++m)
        rows.insert(m);
    for (size_t m : rows) {
        for (size_t n : {size_t{1}, ns / 2 + 1, ns, ns + 3}) {
            seed.m = m;
            seed.n = n;
            cases.push_back(seed);
        }
    }
    return testing::ValuesIn(cases);
}

// ============================================================================
// Random data generators (uint8 is not supported by UniformRandomGenerator)
// ============================================================================

static void fill_random_u8(uint8_t *data, size_t count, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<uint8_t>(dist(rng));
    }
}

static void fill_random_i32(int32_t *data, size_t count, std::mt19937 &rng, int32_t lo = -1000,
                            int32_t hi = 1000)
{
    std::uniform_int_distribution<int32_t> dist(lo, hi);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

static void fill_random_f32(float *data, size_t count, std::mt19937 &rng, float lo = -1.0f,
                            float hi = 1.0f)
{
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

// ============================================================================
// Reference: u8 x u8 -> i32 with mx1bias + 1xnbias
//
// D[m,n] = C[m,n] + bias_m[m] + bias_n[n] + sum_k( A[m,k] * B[n,k] )
// A is [M,K] row-major uint8.  B is [N,K] transposed uint8.
// ============================================================================

static void reference_gemm_i32_u8(size_t m, size_t n, size_t k, const uint8_t *A, size_t lda,
                                  const uint8_t *B, size_t ldb, const int32_t *C, size_t ldc,
                                  int32_t *D, size_t ldd, const int32_t *bias_m,
                                  const int32_t *bias_n)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = C ? C[i * ldc + j] : 0;
            acc += bias_m[i] + bias_n[j];
            for (size_t p = 0; p < k; ++p) {
                acc += static_cast<int32_t>(A[i * lda + p]) * static_cast<int32_t>(B[j * ldb + p]);
            }
            D[i * ldd + j] = acc;
        }
    }
}

// ============================================================================
// gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv
// Non-packed: uint8 A[M,K] x uint8 B[N,K] -> int32 D[M,N]
// ============================================================================

class Test_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv
    : public testing::TestWithParam<U8GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv, Correctness)
{
    const auto &params = GetParam();
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    std::mt19937 rng(42);

    // Generate random data
    std::vector<uint8_t> A(m * k);
    std::vector<uint8_t> B(n * k);  // B is [N,K] transposed
    std::vector<int32_t> bias_m(m);
    std::vector<int32_t> bias_n(n);
    std::vector<int32_t> C(m * n);
    std::vector<int32_t> D_ref(m * n, 0);
    std::vector<int32_t> D_act(m * n, 0);

    fill_random_u8(A.data(), m * k, rng);
    fill_random_u8(B.data(), n * k, rng);
    fill_random_i32(bias_m.data(), m, rng);
    fill_random_i32(bias_n.data(), n, rng);
    if (params.has_c) {
        fill_random_i32(C.data(), m * n, rng);
    }

    const int32_t *c_ptr = params.has_c ? C.data() : nullptr;

    // Reference computation
    reference_gemm_i32_u8(m, n, k, A.data(), k, B.data(), k, c_ptr, n, D_ref.data(), n,
                          bias_m.data(), bias_n.data());

    // Kernel computation using tiling loop
    const size_t m_step = tqt_get_m_step_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv();
    const size_t n_step = tqt_get_n_step_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_off =
                tqt_get_a_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(m_idx, 0, k);
            const size_t b_off =
                tqt_get_b_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(n_idx, 0, k);
            const size_t c_off =
                tqt_get_c_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(m_idx, n_idx, n);
            const size_t d_off =
                tqt_get_d_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(m_idx, n_idx, n);
            const size_t bm_off =
                tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(m_idx);
            const size_t bn_off =
                tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(n_idx);

            const void *a_ptr = reinterpret_cast<const char *>(A.data()) + a_off;
            const void *b_ptr = reinterpret_cast<const char *>(B.data()) + b_off;
            const void *c_tile =
                params.has_c ? reinterpret_cast<const char *>(C.data()) + c_off : nullptr;
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_off;
            const void *bm_ptr = reinterpret_cast<const char *>(bias_m.data()) + bm_off;
            const void *bn_ptr = reinterpret_cast<const char *>(bias_n.data()) + bn_off;

            tqt_run_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv(actual_m, actual_n, k, a_ptr, k, 0,
                                                              b_ptr, k, 0, c_tile, n, d_ptr, n,
                                                              bm_ptr, bn_ptr);
        }
    }

    // Verify: exact match for int32
    auto result = verify_gemm_result<int32_t>(D_ref.data(), D_act.data(), m * n, 0.0f, 0.9999f);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_mx1bias_1xnbias_i32_u8_u8t_4x1vl_rvv,
                         u8_common_test_values(), U8GemmTestParamNameGenerator());

// ============================================================================
// gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv
// Packed: uint8 A[M,K] -> packed, uint8 B[N,K] -> packed -> int32 D[M,N]
// ============================================================================

class Test_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv
    : public testing::TestWithParam<U8GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv, Correctness)
{
    const auto &params = GetParam();
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    std::mt19937 rng(42);

    // Generate random data
    std::vector<uint8_t> A(m * k);
    std::vector<uint8_t> B(n * k);  // B is [N,K] transposed
    std::vector<int32_t> bias_m(m);
    std::vector<int32_t> bias_n(n);
    std::vector<int32_t> C(m * n);
    std::vector<int32_t> D_ref(m * n, 0);
    std::vector<int32_t> D_act(m * n, 0);

    fill_random_u8(A.data(), m * k, rng);
    fill_random_u8(B.data(), n * k, rng);
    fill_random_i32(bias_m.data(), m, rng);
    fill_random_i32(bias_n.data(), n, rng);
    if (params.has_c) {
        fill_random_i32(C.data(), m * n, rng);
    }

    const int32_t *c_ptr = params.has_c ? C.data() : nullptr;

    // Reference computation (B is [N,K] transposed)
    reference_gemm_i32_u8(m, n, k, A.data(), k, B.data(), k, c_ptr, n, D_ref.data(), n,
                          bias_m.data(), bias_n.data());

    // Pack A and B
    const size_t a_packed_size =
        tqt_get_a_packed_size_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(m, k);
    const size_t b_packed_size =
        tqt_get_b_packed_size_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    tqt_run_a_pack_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(m, k, k, k, 0, A.data(),
                                                              A_packed.data());
    // B is [N,K] transposed -> use bt_pack
    tqt_run_bt_pack_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(n, k, k, k, 0, B.data(),
                                                               B_packed.data());

    // Kernel computation using tiling loop
    const size_t m_step = tqt_get_m_step_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv();
    const size_t n_step = tqt_get_n_step_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_off =
                tqt_get_a_packed_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(m_idx, 0, k, m);
            const size_t b_off =
                tqt_get_b_packed_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(n_idx, 0, k, n);
            const size_t c_off =
                tqt_get_c_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(m_idx, n_idx, n);
            const size_t d_off =
                tqt_get_d_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(m_idx, n_idx, n);
            const size_t bm_off =
                tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(m_idx);
            const size_t bn_off =
                tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(n_idx);

            const void *a_ptr = A_packed.data() + a_off;
            const void *b_ptr = B_packed.data() + b_off;
            const void *c_tile =
                params.has_c ? reinterpret_cast<const char *>(C.data()) + c_off : nullptr;
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_off;
            const void *bm_ptr = reinterpret_cast<const char *>(bias_m.data()) + bm_off;
            const void *bn_ptr = reinterpret_cast<const char *>(bias_n.data()) + bn_off;

            tqt_run_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv(actual_m, actual_n, k, a_ptr, k, 0,
                                                               b_ptr, k, 0, c_tile, n, d_ptr, n,
                                                               bm_ptr, bn_ptr);
        }
    }

    // Verify: exact match for int32
    auto result = verify_gemm_result<int32_t>(D_ref.data(), D_act.data(), m * n, 0.0f, 0.9999f);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_mx1bias_1xnbias_i32_u8p_u8p_4x1vl_rvv,
                         u8_common_test_values(), U8GemmTestParamNameGenerator());

// ============================================================================
// gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv
// Non-packed: uint8 A[M,K] x uint8 B[N,K] -> float D[M,N]
// D = clamp(scvtf(acc_i32) * scale + dst_bias[n], min, max)
// ============================================================================

class Test_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv
    : public testing::TestWithParam<U8GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv, Correctness)
{
    const auto &params = GetParam();
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    std::mt19937 rng(42);

    // Generate random data
    std::vector<uint8_t> A(m * k);
    std::vector<uint8_t> B(n * k);  // B is [N,K] transposed
    std::vector<int32_t> bias_m(m);
    std::vector<int32_t> bias_n(n);
    std::vector<int32_t> C(m * n);
    std::vector<float> dst_bias(n);

    fill_random_u8(A.data(), m * k, rng);
    fill_random_u8(B.data(), n * k, rng);
    fill_random_i32(bias_m.data(), m, rng);
    fill_random_i32(bias_n.data(), n, rng);
    if (params.has_c) {
        fill_random_i32(C.data(), m * n, rng);
    }
    fill_random_f32(dst_bias.data(), n, rng, -10.0f, 10.0f);

    const float scale = 0.001f;
    const float clamp_min = -FLT_MAX;
    const float clamp_max = FLT_MAX;

    struct tqt_u8_u8_clamp_f32_params qparams;
    qparams.scale = scale;

    // Reference computation: i32 accumulation then dequant to f32
    std::vector<float> D_ref(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = params.has_c ? C[i * n + j] : 0;
            acc += bias_m[i] + bias_n[j];
            for (size_t p = 0; p < k; ++p) {
                acc += static_cast<int32_t>(A[i * k + p]) * static_cast<int32_t>(B[j * k + p]);
            }
            float val = static_cast<float>(acc) * scale + dst_bias[j];
            val = std::max(clamp_min, std::min(clamp_max, val));
            D_ref[i * n + j] = val;
        }
    }

    // Kernel computation using tiling loop
    std::vector<float> D_act(m * n, 0.0f);

    const size_t m_step = tqt_get_m_step_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv();
    const size_t n_step = tqt_get_n_step_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_off =
                tqt_get_a_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(m_idx, 0, k);
            const size_t b_off =
                tqt_get_b_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(n_idx, 0, k);
            const size_t c_off =
                tqt_get_c_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(m_idx, n_idx, n);
            const size_t d_off =
                tqt_get_d_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(m_idx, n_idx, n);
            const size_t bm_off =
                tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(m_idx);
            const size_t bn_off =
                tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(n_idx);
            const size_t db_off =
                tqt_get_dst_bias_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(n_idx);

            const void *a_ptr = reinterpret_cast<const char *>(A.data()) + a_off;
            const void *b_ptr = reinterpret_cast<const char *>(B.data()) + b_off;
            const void *c_tile =
                params.has_c ? reinterpret_cast<const char *>(C.data()) + c_off : nullptr;
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_off;
            const void *bm_ptr = reinterpret_cast<const char *>(bias_m.data()) + bm_off;
            const void *bn_ptr = reinterpret_cast<const char *>(bias_n.data()) + bn_off;
            const void *db_ptr = reinterpret_cast<const char *>(dst_bias.data()) + db_off;

            clamp_test_run<tqt_run_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv, float, 15, 11,
                           12, 16, ClampSource::kColumnBias, false, 5, false>()(
                actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, c_tile, n, d_ptr, n, bm_ptr,
                bn_ptr, db_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Verify with tolerance for float conversion
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result =
        verify_gemm_result<float>(D_ref.data(), D_act.data(), m * n, tolerance, cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

TEST_P(Test_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    const auto &params = GetParam();
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    std::mt19937 rng(42);

    // Generate random data
    std::vector<uint8_t> A(m * k);
    std::vector<uint8_t> B(n * k);  // B is [N,K] transposed
    std::vector<int32_t> bias_m(m);
    std::vector<int32_t> bias_n(n);
    std::vector<int32_t> C(m * n);
    std::vector<float> dst_bias(n);

    fill_random_u8(A.data(), m * k, rng);
    fill_random_u8(B.data(), n * k, rng);
    fill_random_i32(bias_m.data(), m, rng);
    fill_random_i32(bias_n.data(), n, rng);
    if (params.has_c) {
        fill_random_i32(C.data(), m * n, rng);
    }
    fill_random_f32(dst_bias.data(), n, rng, -10.0f, 10.0f);

    const float scale = 0.001f;
    const float clamp_min = -FLT_MAX;
    const float clamp_max = FLT_MAX;

    struct tqt_u8_u8_clamp_f32_params qparams;
    qparams.scale = scale;

    // Reference computation: i32 accumulation then dequant to f32
    std::vector<float> D_ref(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = params.has_c ? C[i * n + j] : 0;
            acc += bias_m[i] + bias_n[j];
            for (size_t p = 0; p < k; ++p) {
                acc += static_cast<int32_t>(A[i * k + p]) * static_cast<int32_t>(B[j * k + p]);
            }
            float val = static_cast<float>(acc) * scale + dst_bias[j];
            val = std::max(clamp_min, std::min(clamp_max, val));
            D_ref[i * n + j] = val;
        }
    }

    // Kernel computation using tiling loop
    std::vector<float> D_act(m * n, 0.0f);

    const size_t m_step = tqt_get_m_step_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv();
    const size_t n_step = tqt_get_n_step_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_off =
                tqt_get_a_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(m_idx, 0, k);
            const size_t b_off =
                tqt_get_b_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(n_idx, 0, k);
            const size_t c_off =
                tqt_get_c_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(m_idx, n_idx, n);
            const size_t d_off =
                tqt_get_d_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(m_idx, n_idx, n);
            const size_t bm_off =
                tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(m_idx);
            const size_t bn_off =
                tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(n_idx);
            const size_t db_off =
                tqt_get_dst_bias_offset_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv(n_idx);

            const void *a_ptr = reinterpret_cast<const char *>(A.data()) + a_off;
            const void *b_ptr = reinterpret_cast<const char *>(B.data()) + b_off;
            const void *c_tile =
                params.has_c ? reinterpret_cast<const char *>(C.data()) + c_off : nullptr;
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_off;
            const void *bm_ptr = reinterpret_cast<const char *>(bias_m.data()) + bm_off;
            const void *bn_ptr = reinterpret_cast<const char *>(bias_n.data()) + bn_off;
            const void *db_ptr = reinterpret_cast<const char *>(dst_bias.data()) + db_off;

            clamp_test_run<tqt_run_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv, float, 15, 11,
                           12, 16, ClampSource::kColumnBias, false, 5, false>()(
                actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, c_tile, n, d_ptr, n, bm_ptr,
                bn_ptr, db_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Verify with tolerance for float conversion
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result =
        verify_gemm_result<float>(D_ref.data(), D_act.data(), m * n, tolerance, cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_mx1bias_1xnbias_clamp_f32_u8_u8t_4x1vl_rvv,
                         clamp_u8_dimension_values(), U8GemmTestParamNameGenerator());

// ============================================================================
// gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv
// Packed: uint8 A[M,K] -> packed, uint8 B[N,K] -> packed -> float D[M,N]
// D = clamp(scvtf(acc_i32) * scale + dst_bias[n], min, max)
// ============================================================================

class Test_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv
    : public testing::TestWithParam<U8GemmTestParams>
{
};

TEST_P(Test_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv, Correctness)
{
    const auto &params = GetParam();
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    std::mt19937 rng(42);

    // Generate random data
    std::vector<uint8_t> A(m * k);
    std::vector<uint8_t> B(n * k);  // B is [N,K] transposed
    std::vector<int32_t> bias_m(m);
    std::vector<int32_t> bias_n(n);
    std::vector<int32_t> C(m * n);
    std::vector<float> dst_bias(n);

    fill_random_u8(A.data(), m * k, rng);
    fill_random_u8(B.data(), n * k, rng);
    fill_random_i32(bias_m.data(), m, rng);
    fill_random_i32(bias_n.data(), n, rng);
    if (params.has_c) {
        fill_random_i32(C.data(), m * n, rng);
    }
    fill_random_f32(dst_bias.data(), n, rng, -10.0f, 10.0f);

    const float scale = 0.001f;
    const float clamp_min = -FLT_MAX;
    const float clamp_max = FLT_MAX;

    struct tqt_u8_u8_clamp_f32_params qparams;
    qparams.scale = scale;

    // Reference computation: i32 accumulation then dequant to f32
    std::vector<float> D_ref(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = params.has_c ? C[i * n + j] : 0;
            acc += bias_m[i] + bias_n[j];
            for (size_t p = 0; p < k; ++p) {
                acc += static_cast<int32_t>(A[i * k + p]) * static_cast<int32_t>(B[j * k + p]);
            }
            float val = static_cast<float>(acc) * scale + dst_bias[j];
            val = std::max(clamp_min, std::min(clamp_max, val));
            D_ref[i * n + j] = val;
        }
    }

    // Pack A and B
    const size_t a_packed_size =
        tqt_get_a_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m, k);
    const size_t b_packed_size =
        tqt_get_b_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    tqt_run_a_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m, k, k, k, 0, A.data(),
                                                                    A_packed.data());
    // B is [N,K] transposed -> use bt_pack
    tqt_run_bt_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n, k, k, k, 0, B.data(),
                                                                     B_packed.data());

    // Kernel computation using tiling loop
    std::vector<float> D_act(m * n, 0.0f);

    const size_t m_step = tqt_get_m_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv();
    const size_t n_step = tqt_get_n_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_off =
                tqt_get_a_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m_idx, 0,
                                                                                         k, m);
            const size_t b_off =
                tqt_get_b_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n_idx, 0,
                                                                                         k, n);
            const size_t c_off =
                tqt_get_c_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m_idx, n_idx, n);
            const size_t d_off =
                tqt_get_d_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m_idx, n_idx, n);
            const size_t bm_off =
                tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m_idx);
            const size_t bn_off =
                tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n_idx);
            const size_t db_off =
                tqt_get_dst_bias_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n_idx);

            const void *a_ptr = A_packed.data() + a_off;
            const void *b_ptr = B_packed.data() + b_off;
            const void *c_tile =
                params.has_c ? reinterpret_cast<const char *>(C.data()) + c_off : nullptr;
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_off;
            const void *bm_ptr = reinterpret_cast<const char *>(bias_m.data()) + bm_off;
            const void *bn_ptr = reinterpret_cast<const char *>(bias_n.data()) + bn_off;
            const void *db_ptr = reinterpret_cast<const char *>(dst_bias.data()) + db_off;

            clamp_test_run<tqt_run_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv, float, 15, 11,
                           12, 16, ClampSource::kColumnBias, false, 5, false>()(
                actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, c_tile, n, d_ptr, n, bm_ptr,
                bn_ptr, db_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Verify with tolerance for float conversion
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result =
        verify_gemm_result<float>(D_ref.data(), D_act.data(), m * n, tolerance, cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

TEST_P(Test_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv, ClampSpecialValues)
{
    ScopedGemmClampProbe probe;
    const auto &params = GetParam();
    const size_t m = params.m;
    const size_t n = params.n;
    const size_t k = params.k;

    std::mt19937 rng(42);

    // Generate random data
    std::vector<uint8_t> A(m * k);
    std::vector<uint8_t> B(n * k);  // B is [N,K] transposed
    std::vector<int32_t> bias_m(m);
    std::vector<int32_t> bias_n(n);
    std::vector<int32_t> C(m * n);
    std::vector<float> dst_bias(n);

    fill_random_u8(A.data(), m * k, rng);
    fill_random_u8(B.data(), n * k, rng);
    fill_random_i32(bias_m.data(), m, rng);
    fill_random_i32(bias_n.data(), n, rng);
    if (params.has_c) {
        fill_random_i32(C.data(), m * n, rng);
    }
    fill_random_f32(dst_bias.data(), n, rng, -10.0f, 10.0f);

    const float scale = 0.001f;
    const float clamp_min = -FLT_MAX;
    const float clamp_max = FLT_MAX;

    struct tqt_u8_u8_clamp_f32_params qparams;
    qparams.scale = scale;

    // Reference computation: i32 accumulation then dequant to f32
    std::vector<float> D_ref(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            int32_t acc = params.has_c ? C[i * n + j] : 0;
            acc += bias_m[i] + bias_n[j];
            for (size_t p = 0; p < k; ++p) {
                acc += static_cast<int32_t>(A[i * k + p]) * static_cast<int32_t>(B[j * k + p]);
            }
            float val = static_cast<float>(acc) * scale + dst_bias[j];
            val = std::max(clamp_min, std::min(clamp_max, val));
            D_ref[i * n + j] = val;
        }
    }

    // Pack A and B
    const size_t a_packed_size =
        tqt_get_a_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m, k);
    const size_t b_packed_size =
        tqt_get_b_packed_size_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n, k);
    std::vector<uint8_t> A_packed(a_packed_size);
    std::vector<uint8_t> B_packed(b_packed_size);

    tqt_run_a_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m, k, k, k, 0, A.data(),
                                                                    A_packed.data());
    // B is [N,K] transposed -> use bt_pack
    tqt_run_bt_pack_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n, k, k, k, 0, B.data(),
                                                                     B_packed.data());

    // Kernel computation using tiling loop
    std::vector<float> D_act(m * n, 0.0f);

    const size_t m_step = tqt_get_m_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv();
    const size_t n_step = tqt_get_n_step_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv();

    for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
            const size_t actual_m = std::min(m - m_idx, m_step);
            const size_t actual_n = std::min(n - n_idx, n_step);

            const size_t a_off =
                tqt_get_a_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m_idx, 0,
                                                                                         k, m);
            const size_t b_off =
                tqt_get_b_packed_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n_idx, 0,
                                                                                         k, n);
            const size_t c_off =
                tqt_get_c_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m_idx, n_idx, n);
            const size_t d_off =
                tqt_get_d_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m_idx, n_idx, n);
            const size_t bm_off =
                tqt_get_bias_m_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(m_idx);
            const size_t bn_off =
                tqt_get_bias_n_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n_idx);
            const size_t db_off =
                tqt_get_dst_bias_offset_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv(n_idx);

            const void *a_ptr = A_packed.data() + a_off;
            const void *b_ptr = B_packed.data() + b_off;
            const void *c_tile =
                params.has_c ? reinterpret_cast<const char *>(C.data()) + c_off : nullptr;
            void *d_ptr = reinterpret_cast<char *>(D_act.data()) + d_off;
            const void *bm_ptr = reinterpret_cast<const char *>(bias_m.data()) + bm_off;
            const void *bn_ptr = reinterpret_cast<const char *>(bias_n.data()) + bn_off;
            const void *db_ptr = reinterpret_cast<const char *>(dst_bias.data()) + db_off;

            clamp_test_run<tqt_run_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv, float, 15, 11,
                           12, 16, ClampSource::kColumnBias, false, 5, false>()(
                actual_m, actual_n, k, a_ptr, k, 0, b_ptr, k, 0, c_tile, n, d_ptr, n, bm_ptr,
                bn_ptr, db_ptr, clamp_min, clamp_max, &qparams);
        }
    }

    // Verify with tolerance for float conversion
    const float tolerance = 1.0f;
    const float cosine_threshold = 0.999f;
    auto result =
        verify_gemm_result<float>(D_ref.data(), D_act.data(), m * n, tolerance, cosine_threshold);
    EXPECT_TRUE(result.passed) << result.error_message << "\n  Shape: M=" << m << " N=" << n
                               << " K=" << k << "\n  Max abs error: " << result.max_abs_error
                               << "\n  Cosine similarity: " << result.cosine_similarity;
}

INSTANTIATE_TEST_SUITE_P(GemmTests, Test_gemm_mx1bias_1xnbias_clamp_f32_u8p_u8p_4x1vl_rvv,
                         clamp_u8_dimension_values(), U8GemmTestParamNameGenerator());
