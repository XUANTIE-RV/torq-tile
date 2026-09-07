//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/common/tqt_common.h"
#include "test/common/data_generator.h"
#include "test/common/reference_activation.h"
#include "test/common/verify.h"

namespace torq_tile
{
namespace test
{

// ============================================================================
// Activation Test Harness
// ============================================================================
//
// Activation micro-kernels come in two shapes, and this harness drives both over
// float / float16_t / bfloat16_t:
//
//   single-operand (relu, gelu, silu): run(m, n, a, lda, c, ldc)
//   gated          (reglu, geglu, swiglu): run(m, n, a, lda, b, ldb, c, ldc)
//
// Both work on a row-major [m, n] matrix in which every operand carries its own
// leading dimension, and both allow the output to alias an input. The shape set,
// the parameter struct, the padding guard and the window-gathering helper are
// shared; only the per-contract drivers differ, because the number of operands the
// harness has to allocate and seed is what actually changes.
//
// Properties checked on every shape, for both contracts:
//   1. The [m, n] window matches the reference within an absolute tolerance.
//   2. The output padding (columns [n, ldc) of every row) is left untouched, so a
//      kernel that overruns its row window fails instead of passing silently.
//   3. In place works. The gated contract has two directions (c aliasing a, and c
//      aliasing b), and the operand that is not aliased must come back untouched.
//
// The gated contract additionally gets a fused-layout check: the [M, 2N]
// gate-and-up buffer, addressed purely through strides, must agree bit for bit
// with two separate operand buffers. That property is what makes a separate
// split-half kernel unnecessary, so it is worth asserting directly.
// ============================================================================

/// Run signature of the single-operand activation micro-kernels (relu, gelu, silu).
using ActivationRunFunc = void (*)(size_t m, size_t n, const void *a, size_t lda, void *c,
                                   size_t ldc);

/// Run signature of the gated activation micro-kernels (reglu, geglu, swiglu).
using GatedActivationRunFunc = void (*)(size_t m, size_t n, const void *a, size_t lda,
                                        const void *b, size_t ldb, void *c, size_t ldc);

/// Which activation the reference should compute, for the single-operand drivers.
enum class ActivationKind
{
    Relu,
    Silu,
    Gelu,
};

/// Which gated activation the reference should compute.
enum class GatedActivationKind
{
    Reglu,
    Geglu,
    Swiglu,
};

/// Which operand the output buffer aliases in the gated in-place test.
enum class GatedAliasOperand
{
    A,
    B,
};

struct ActivationTestParams
{
    size_t m;
    size_t n;
    size_t lda_pad;  // lda = n + lda_pad
    size_t ldb_pad;  // ldb = n + ldb_pad; gated contract only, ignored by single-input ops
    size_t ldc_pad;  // ldc = n + ldc_pad
    std::string name;

    ActivationTestParams(size_t m_, size_t n_, size_t lda_pad_, size_t ldb_pad_, size_t ldc_pad_,
                         std::string name_)
        : m(m_),
          n(n_),
          lda_pad(lda_pad_),
          ldb_pad(ldb_pad_),
          ldc_pad(ldc_pad_),
          name(std::move(name_))
    {
    }
};

struct ActivationTestParamNameGenerator
{
    std::string operator()(const testing::TestParamInfo<ActivationTestParams> &info) const
    {
        return info.param.name;
    }
};

/// The one shape set behind both contracts.
///
/// The first block keeps every leading dimension equal to n, which exercises the
/// fully packed contract. The remaining blocks vary the paddings independently over
/// small, large, and boundary-shaped cases, including M=7/8 and N=1023/1024.
///
/// `ldb_pad` only means something under the gated contract. The B-only row exists to
/// catch a gated kernel whose packed fast path forgets to test `ldb == n`; see
/// activation_test_values() for how the single-operand contract handles it.
inline const std::vector<ActivationTestParams> &activation_shapes()
{
    static const std::vector<ActivationTestParams> shapes = {
        // Fully packed operands and output.
        ActivationTestParams(1, 1, 0, 0, 0, "M1_N1"),          //
        ActivationTestParams(2, 7, 0, 0, 0, "M2_N7"),          //
        ActivationTestParams(3, 16, 0, 0, 0, "M3_N16"),        //
        ActivationTestParams(5, 127, 0, 0, 0, "M5_N127"),      //
        ActivationTestParams(8, 256, 0, 0, 0, "M8_N256"),      //
        ActivationTestParams(17, 1023, 0, 0, 0, "M17_N1023"),  //
        ActivationTestParams(1, 4096, 0, 0, 0, "M1_N4096"),    //
        ActivationTestParams(64, 64, 0, 0, 0, "M64_N64"),      //
        ActivationTestParams(100, 100, 0, 0, 0, "M100_N100"),  //
        // Padded operands and/or output over small shapes.
        ActivationTestParams(1, 1, 3, 7, 5, "Padded_M1_N1"),       //
        ActivationTestParams(3, 100, 5, 0, 0, "PaddedA_M3_N100"),  //
        ActivationTestParams(3, 100, 0, 6, 0, "PaddedB_M3_N100"),  //
        ActivationTestParams(3, 100, 0, 0, 9, "PaddedC_M3_N100"),  //
        ActivationTestParams(7, 33, 2, 3, 4, "Padded_M7_N33"),     //
        // Padded operands and output over many-row shapes.
        ActivationTestParams(33, 17, 1, 5, 2, "Padded_M33_N17"),       //
        ActivationTestParams(40, 100, 7, 1, 3, "Padded_M40_N100"),     //
        ActivationTestParams(64, 64, 64, 1, 65, "Padded_M64_N64"),     //
        ActivationTestParams(100, 300, 11, 4, 6, "Padded_M100_N300"),  //
        // Long-row strided coverage.
        ActivationTestParams(16, 1024, 3, 7, 5, "PaddedLong_M16_N1024"),  //
        // Boundary-shaped strided coverage.
        ActivationTestParams(8, 1023, 2, 4, 2, "PaddedBoundary_M8_N1023"),  //
        ActivationTestParams(7, 1024, 2, 4, 2, "PaddedBoundary_M7_N1024"),
    };
    return shapes;
}

namespace detail
{

/// Drops the rows whose only padding is `ldb_pad`. With no B operand those rows
/// degenerate to a fully packed case, and keeping them would put a row named
/// "PaddedB_..." into a test list where nothing is actually padded.
inline std::vector<ActivationTestParams> single_operand_shapes()
{
    std::vector<ActivationTestParams> kept;
    for (const ActivationTestParams &p : activation_shapes()) {
        const bool b_only = p.ldb_pad != 0 && p.lda_pad == 0 && p.ldc_pad == 0;
        if (!b_only) {
            kept.push_back(p);
        }
    }
    return kept;
}

}  // namespace detail

/// Shape set for the single-operand contract (relu, gelu, silu).
inline auto activation_test_values()
{
    static const std::vector<ActivationTestParams> single = detail::single_operand_shapes();
    return testing::ValuesIn(single);
}

/// Shape set for the gated contract: every row, B-side padding included.
inline auto gated_activation_test_values()
{
    return testing::ValuesIn(activation_shapes());
}

namespace detail
{

/// Sentinel written into every output element before the kernel runs. Any element
/// outside the [m, n] window must still hold it afterwards.
template <typename T>
inline T activation_guard_value()
{
    return static_cast<T>(-12345.0f);
}

/// Gathers the [m, n] window out of a padded buffer so the comparison is not
/// diluted by the padding.
template <typename T>
inline std::vector<T> gather_window(const T *src, size_t m, size_t n, size_t ld)
{
    std::vector<T> packed(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            packed[i * n + j] = src[i * ld + j];
        }
    }
    return packed;
}

template <typename T>
inline void activation_reference(ActivationKind kind, size_t m, size_t n, const T *a, size_t lda,
                                 T *c, size_t ldc)
{
    if (kind == ActivationKind::Relu) {
        reference_relu<T>(m, n, a, lda, c, ldc);
    } else if (kind == ActivationKind::Silu) {
        reference_silu<T>(m, n, a, lda, c, ldc);
    } else {
        reference_gelu<T>(m, n, a, lda, c, ldc);
    }
}

template <typename T>
inline void gated_activation_reference(GatedActivationKind kind, size_t m, size_t n, const T *a,
                                       size_t lda, const T *b, size_t ldb, T *c, size_t ldc)
{
    if (kind == GatedActivationKind::Reglu) {
        reference_reglu<T>(m, n, a, lda, b, ldb, c, ldc);
    } else if (kind == GatedActivationKind::Geglu) {
        reference_geglu<T>(m, n, a, lda, b, ldb, c, ldc);
    } else {
        reference_swiglu<T>(m, n, a, lda, b, ldb, c, ldc);
    }
}

}  // namespace detail

// ============================================================================
// Single-operand contract: run(m, n, a, lda, c, ldc)
// ============================================================================

/// Drives one single-operand activation micro-kernel over one shape.
///
/// @param run                      Kernel entry point.
/// @param kind                     Which reference activation to compare against.
/// @param p                        Shape and padding (ldb_pad unused).
/// @param max_abs_error_threshold  Absolute error budget per element. Pass 0 for
///                                 bit-exact kernels such as relu.
/// @param min_cosine_similarity    Cosine similarity floor over the valid window.
template <typename T>
void run_activation_test(ActivationRunFunc run, ActivationKind kind, const ActivationTestParams &p,
                         float max_abs_error_threshold, float min_cosine_similarity)
{
    const size_t m = p.m;
    const size_t n = p.n;
    const size_t lda = n + p.lda_pad;
    const size_t ldc = n + p.ldc_pad;
    const T guard = detail::activation_guard_value<T>();

    std::vector<T> input(m * lda);
    std::vector<T> output_ref(m * ldc, guard);
    std::vector<T> output_actual(m * ldc, guard);

    // Fill the whole input extent, padding included, so a kernel that reads its
    // row padding still sees defined values rather than uninitialised memory.
    UniformRandomGenerator<T> rng(static_cast<T>(-10.0f), static_cast<T>(10.0f), 42);
    rng.fill_matrix(input.data(), m, lda);

    detail::activation_reference<T>(kind, m, n, input.data(), lda, output_ref.data(), ldc);
    run(m, n, input.data(), lda, output_actual.data(), ldc);

    auto packed_ref = detail::gather_window<T>(output_ref.data(), m, n, ldc);
    auto packed_actual = detail::gather_window<T>(output_actual.data(), m, n, ldc);

    auto result = verify_gemm_result<T>(packed_ref.data(), packed_actual.data(), m * n,
                                        max_abs_error_threshold, min_cosine_similarity);
    ASSERT_TRUE(result.passed) << result.error_message;

    // The output padding must be untouched.
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = n; j < ldc; ++j) {
            ASSERT_EQ(static_cast<float>(output_actual[i * ldc + j]), static_cast<float>(guard))
                << "output padding written at row " << i << ", column " << j;
        }
    }
}

/// Drives one single-operand activation micro-kernel in place (c == a, ldc == lda).
///
/// @param run                      Kernel entry point.
/// @param kind                     Which reference activation to compare against.
/// @param p                        Shape and padding (only lda_pad is used).
/// @param max_abs_error_threshold  Absolute error budget per element.
/// @param min_cosine_similarity    Cosine similarity floor over the valid window.
template <typename T>
void run_activation_inplace_test(ActivationRunFunc run, ActivationKind kind,
                                 const ActivationTestParams &p, float max_abs_error_threshold,
                                 float min_cosine_similarity)
{
    const size_t m = p.m;
    const size_t n = p.n;
    const size_t lda = n + p.lda_pad;

    std::vector<T> input(m * lda);
    UniformRandomGenerator<T> rng(static_cast<T>(-10.0f), static_cast<T>(10.0f), 42);
    rng.fill_matrix(input.data(), m, lda);

    // Reference is computed from a pristine copy into a separate buffer, then the
    // kernel overwrites the original.
    std::vector<T> original = input;
    std::vector<T> output_ref(m * lda);
    detail::activation_reference<T>(kind, m, n, original.data(), lda, output_ref.data(), lda);

    run(m, n, input.data(), lda, input.data(), lda);

    auto packed_ref = detail::gather_window<T>(output_ref.data(), m, n, lda);
    auto packed_actual = detail::gather_window<T>(input.data(), m, n, lda);

    auto result = verify_gemm_result<T>(packed_ref.data(), packed_actual.data(), m * n,
                                        max_abs_error_threshold, min_cosine_similarity);
    ASSERT_TRUE(result.passed) << result.error_message;

    // In place, the row padding of the input doubles as the output padding and
    // must survive untouched.
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = n; j < lda; ++j) {
            ASSERT_EQ(static_cast<float>(input[i * lda + j]),
                      static_cast<float>(original[i * lda + j]))
                << "row padding written at row " << i << ", column " << j;
        }
    }
}

/// Checks deterministic tail and near-zero values. In particular, the negative
/// tail catches a tanh helper that clamps x instead of abs(x).
template <typename T>
void run_activation_edge_values_test(ActivationRunFunc run, ActivationKind kind,
                                     float max_abs_error_threshold, float min_cosine_similarity)
{
    static constexpr float values[] = {-10.0f,   -8.0f, -6.0f, -5.0f,   -3.0f, -1.0f,
                                       -1.0e-4f, -0.0f, 0.0f,  1.0e-4f, 1.0f,  3.0f,
                                       5.0f,     6.0f,  8.0f,  10.0f};
    const size_t n = sizeof(values) / sizeof(values[0]);
    std::vector<T> input(n);
    std::vector<T> output_ref(n);
    std::vector<T> output_actual(n);
    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<T>(values[i]);
    }

    detail::activation_reference<T>(kind, 1, n, input.data(), n, output_ref.data(), n);
    run(1, n, input.data(), n, output_actual.data(), n);

    auto result = verify_gemm_result<T>(output_ref.data(), output_actual.data(), n,
                                        max_abs_error_threshold, min_cosine_similarity);
    ASSERT_TRUE(result.passed) << result.error_message;
}

// ============================================================================
// Gated contract: run(m, n, a, lda, b, ldb, c, ldc)
// ============================================================================

/// Drives one gated activation micro-kernel over one shape.
///
/// @param run                      Kernel entry point.
/// @param kind                     Gated activation used by the reference path.
/// @param p                        Shape and padding.
/// @param max_abs_error_threshold  Absolute error budget per element.
/// @param min_cosine_similarity    Cosine similarity floor over the valid window.
template <typename T>
void run_gated_activation_test(GatedActivationRunFunc run, GatedActivationKind kind,
                               const ActivationTestParams &p, float max_abs_error_threshold,
                               float min_cosine_similarity)
{
    const size_t m = p.m;
    const size_t n = p.n;
    const size_t lda = n + p.lda_pad;
    const size_t ldb = n + p.ldb_pad;
    const size_t ldc = n + p.ldc_pad;
    const T guard = detail::activation_guard_value<T>();

    std::vector<T> gate(m * lda);
    std::vector<T> linear(m * ldb);
    std::vector<T> output_ref(m * ldc, guard);
    std::vector<T> output_actual(m * ldc, guard);

    // Fill the whole extent of both operands, padding included, so a kernel that
    // reads its row padding still sees defined values rather than uninitialised
    // memory. The two generators use different seeds so the operands are not
    // correlated - identical operands would hide an a / b mix-up.
    UniformRandomGenerator<T> rng_a(static_cast<T>(-10.0f), static_cast<T>(10.0f), 42);
    UniformRandomGenerator<T> rng_b(static_cast<T>(-10.0f), static_cast<T>(10.0f), 1337);
    rng_a.fill_matrix(gate.data(), m, lda);
    rng_b.fill_matrix(linear.data(), m, ldb);

    detail::gated_activation_reference<T>(kind, m, n, gate.data(), lda, linear.data(), ldb,
                                          output_ref.data(), ldc);
    run(m, n, gate.data(), lda, linear.data(), ldb, output_actual.data(), ldc);

    auto packed_ref = detail::gather_window<T>(output_ref.data(), m, n, ldc);
    auto packed_actual = detail::gather_window<T>(output_actual.data(), m, n, ldc);

    auto result = verify_gemm_result<T>(packed_ref.data(), packed_actual.data(), m * n,
                                        max_abs_error_threshold, min_cosine_similarity);
    ASSERT_TRUE(result.passed) << result.error_message;

    // The output padding must be untouched.
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = n; j < ldc; ++j) {
            ASSERT_EQ(static_cast<float>(output_actual[i * ldc + j]), static_cast<float>(guard))
                << "output padding written at row " << i << ", column " << j;
        }
    }
}

/// Drives one gated activation micro-kernel in place, with the output aliasing one
/// of the two operands (c == a with ldc == lda, or c == b with ldc == ldb).
///
/// @param run                      Kernel entry point.
/// @param kind                     Gated activation used by the reference path.
/// @param alias                    Which operand the output buffer aliases.
/// @param p                        Shape and padding.
/// @param max_abs_error_threshold  Absolute error budget per element.
/// @param min_cosine_similarity    Cosine similarity floor over the valid window.
template <typename T>
void run_gated_activation_inplace_test(GatedActivationRunFunc run, GatedActivationKind kind,
                                       GatedAliasOperand alias, const ActivationTestParams &p,
                                       float max_abs_error_threshold, float min_cosine_similarity)
{
    const size_t m = p.m;
    const size_t n = p.n;
    const size_t lda = n + p.lda_pad;
    const size_t ldb = n + p.ldb_pad;

    std::vector<T> gate(m * lda);
    std::vector<T> linear(m * ldb);
    UniformRandomGenerator<T> rng_a(static_cast<T>(-10.0f), static_cast<T>(10.0f), 42);
    UniformRandomGenerator<T> rng_b(static_cast<T>(-10.0f), static_cast<T>(10.0f), 1337);
    rng_a.fill_matrix(gate.data(), m, lda);
    rng_b.fill_matrix(linear.data(), m, ldb);

    // The reference is computed from pristine copies into a separate buffer, then
    // the kernel overwrites whichever operand is aliased.
    const std::vector<T> gate_orig = gate;
    const std::vector<T> linear_orig = linear;
    const size_t ldc = (alias == GatedAliasOperand::A) ? lda : ldb;
    std::vector<T> output_ref(m * ldc);
    detail::gated_activation_reference<T>(kind, m, n, gate_orig.data(), lda, linear_orig.data(),
                                          ldb, output_ref.data(), ldc);

    T *out = (alias == GatedAliasOperand::A) ? gate.data() : linear.data();
    run(m, n, gate.data(), lda, linear.data(), ldb, out, ldc);

    auto packed_ref = detail::gather_window<T>(output_ref.data(), m, n, ldc);
    auto packed_actual = detail::gather_window<T>(out, m, n, ldc);

    auto result = verify_gemm_result<T>(packed_ref.data(), packed_actual.data(), m * n,
                                        max_abs_error_threshold, min_cosine_similarity);
    ASSERT_TRUE(result.passed) << result.error_message;

    // In place, the row padding of the aliased operand doubles as the output
    // padding and must survive untouched. The other operand must be untouched
    // entirely, padding and window alike.
    const std::vector<T> &aliased_orig = (alias == GatedAliasOperand::A) ? gate_orig : linear_orig;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = n; j < ldc; ++j) {
            ASSERT_EQ(static_cast<float>(out[i * ldc + j]),
                      static_cast<float>(aliased_orig[i * ldc + j]))
                << "row padding written at row " << i << ", column " << j;
        }
    }

    const std::vector<T> &other = (alias == GatedAliasOperand::A) ? linear : gate;
    const std::vector<T> &other_orig = (alias == GatedAliasOperand::A) ? linear_orig : gate_orig;
    for (size_t k = 0; k < other_orig.size(); ++k) {
        ASSERT_EQ(static_cast<float>(other[k]), static_cast<float>(other_orig[k]))
            << "non-aliased operand written at element " << k;
    }
}

/// Checks deterministic gate tails, near-zero values, and mixed-sign linear values.
template <typename T>
void run_gated_activation_edge_values_test(GatedActivationRunFunc run, GatedActivationKind kind,
                                           float max_abs_error_threshold,
                                           float min_cosine_similarity)
{
    static constexpr float gate_values[] = {-10.0f,   -8.0f, -6.0f, -5.0f,   -3.0f, -1.0f,
                                            -1.0e-4f, -0.0f, 0.0f,  1.0e-4f, 1.0f,  3.0f,
                                            5.0f,     6.0f,  8.0f,  10.0f};
    static constexpr float linear_values[] = {10.0f, -8.0f, 6.0f, -5.0f, 3.0f, -1.0f, 0.5f, -0.0f,
                                              0.0f,  -0.5f, 1.0f, -3.0f, 5.0f, -6.0f, 8.0f, -10.0f};
    const size_t n = sizeof(gate_values) / sizeof(gate_values[0]);
    std::vector<T> gate(n);
    std::vector<T> linear(n);
    std::vector<T> output_ref(n);
    std::vector<T> output_actual(n);
    for (size_t i = 0; i < n; ++i) {
        gate[i] = static_cast<T>(gate_values[i]);
        linear[i] = static_cast<T>(linear_values[i]);
    }

    detail::gated_activation_reference<T>(kind, 1, n, gate.data(), n, linear.data(), n,
                                          output_ref.data(), n);
    run(1, n, gate.data(), n, linear.data(), n, output_actual.data(), n);

    auto result = verify_gemm_result<T>(output_ref.data(), output_actual.data(), n,
                                        max_abs_error_threshold, min_cosine_similarity);
    ASSERT_TRUE(result.passed) << result.error_message;
}

/// Checks that the fused [M, 2N] gate-and-up layout, addressed purely through
/// strides, produces the same result as two separate operand buffers.
///
/// The kernel is called twice: once with a = fused, lda = 2 * n and
/// b = fused + n, ldb = 2 * n, and once with the two halves copied into their own
/// packed buffers. The two outputs must match bit for bit - both calls execute the
/// same arithmetic on the same values, so any difference is an addressing bug.
///
/// @param run  Kernel entry point.
/// @param m    Number of rows.
/// @param n    Half width: the fused buffer is [m, 2 * n] and the output [m, n].
template <typename T>
void run_gated_activation_fused_layout_test(GatedActivationRunFunc run, size_t m, size_t n)
{
    const size_t fused_ld = 2 * n;

    std::vector<T> fused(m * fused_ld);
    UniformRandomGenerator<T> rng(static_cast<T>(-10.0f), static_cast<T>(10.0f), 42);
    rng.fill_matrix(fused.data(), m, fused_ld);

    // Split the halves into their own packed buffers.
    std::vector<T> gate(m * n);
    std::vector<T> linear(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            gate[i * n + j] = fused[i * fused_ld + j];
            linear[i * n + j] = fused[i * fused_ld + n + j];
        }
    }

    std::vector<T> out_fused(m * n);
    std::vector<T> out_split(m * n);
    run(m, n, fused.data(), fused_ld, fused.data() + n, fused_ld, out_fused.data(), n);
    run(m, n, gate.data(), n, linear.data(), n, out_split.data(), n);

    for (size_t k = 0; k < m * n; ++k) {
        ASSERT_EQ(static_cast<float>(out_fused[k]), static_cast<float>(out_split[k]))
            << "fused [M, 2N] layout disagrees with split buffers at element " << k;
    }
}

}  // namespace test
}  // namespace torq_tile
