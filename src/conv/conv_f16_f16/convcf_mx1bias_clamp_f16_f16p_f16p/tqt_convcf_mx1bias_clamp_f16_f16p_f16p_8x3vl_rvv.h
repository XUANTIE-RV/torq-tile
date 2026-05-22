//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Conv micro-kernel: channel-first, Mx1 bias, clamp, packed, FP16
/// RVV 8x3vl tile implementation.
///
/// A = weight [M, K] packed, B = im2col(activation) [K, N] packed
/// where M = OC_per_group, K = IC_per_group * KD * KH * KW, N = OD * OH * OW

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value (8).
size_t tqt_get_m_step_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value (3 * vl, where vl = csrr_vlenb() / sizeof(float16_t)).
size_t tqt_get_n_step_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(void);

/// Gets the offset in bytes to the data element in the packed A matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] k_idx Column index.
/// @param[in] lda Leading dimension of A.
/// @param[in] actual_m Actual number of rows being processed.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_packed_offset_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m_idx,
                                                                            size_t k_idx,
                                                                            size_t lda,
                                                                            size_t actual_m);

/// Gets the offset in bytes to the data element in the im2col packed B matrix buffer.
///
/// @param[in] n_idx Row index.
/// @param[in] k_idx Column index.
/// @param[in] ldb Leading dimension of B.
/// @param[in] actual_n Actual number of columns being processed.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_b_im2colpack_offset_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t n_idx,
                                                                                size_t k_idx,
                                                                                size_t ldb,
                                                                                size_t actual_n);

/// Gets the offset in bytes to the data element in the C matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
///
/// @param[in] m_idx Row index.
///
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldd Leading dimension of D.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                     size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m, size_t n);

/// Gets the size in bytes of the packed A matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] k Number of columns.
///
/// @return The size in bytes of the packed A matrix buffer.
size_t tqt_get_a_packed_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m, size_t k);

/// Gets the size in bytes of the im2col packed B matrix buffer.
///
/// @param[in] n Number of rows.
/// @param[in] k Number of columns.
///
/// @return The size in bytes of the im2col packed B matrix buffer.
size_t tqt_get_b_im2colpack_size_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t n, size_t k);

/// Packs matrix A (weight) for optimal memory access.
///
/// @param[in]  m Number of rows in A.
/// @param[in]  k Number of columns in A.
/// @param[in]  lda Leading dimension of A.
/// @param[in]  lda_packed Leading dimension of packed A buffer.
/// @param[in]  k_idx Starting k index for packing.
/// @param[in]  A Input matrix A [m, k] in row-major format.
/// @param[out] A_packed Packed matrix A buffer.
void tqt_run_a_pack_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(size_t m, size_t k, size_t lda,
                                                                 size_t lda_packed, size_t k_idx,
                                                                 const void *A, void *A_packed);

/// Performs im2col transformation and packs matrix B (activation) for optimal memory access.
///
/// @param[in]  group_idx Group index for grouped convolution.
/// @param[in]  activation Input activation tensor.
/// @param[in]  ldb_packed Leading dimension of packed B buffer.
/// @param[in]  k_idx Starting k index for packing.
/// @param[out] B_packed Packed matrix B buffer (im2col transformed activation).
/// @param[in]  params Convolution parameters containing kernel size, padding, stride, etc.
void tqt_run_b_im2colpack_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(
    size_t group_idx, const void *activation, size_t ldb_packed, size_t k_idx, void *B_packed,
    const tqt_conv_params *params);

/// Runs the convolution microkernel with packed matrices, bias addition and clamp operation.
///
/// Computes: D = clamp(C + A_packed*B_packed + bias, clamp_min, clamp_max)
///
/// @param[in]  m Number of rows in A and D (OC_per_group).
/// @param[in]  n Number of rows in B and columns in D (OD * OH * OW).
/// @param[in]  k Common dimension (IC_per_group * KD * KH * KW).
/// @param[in]  A_packed Packed input matrix A (weight).
/// @param[in]  lda Leading dimension of original A.
/// @param[in]  k_idx_a K index in A buffer (for sub-buffer scenarios).
/// @param[in]  B_packed Packed input matrix B (im2col transformed activation).
/// @param[in]  ldb Leading dimension of original B.
/// @param[in]  k_idx_b K index in B buffer (for sub-buffer scenarios).
/// @param[in]  C Input matrix C [m, n] in row-major format.
/// @param[in]  ldc Leading dimension of C.
/// @param[out] D Output matrix D [m, n] in row-major format.
/// @param[in]  ldd Leading dimension of D.
/// @param[in]  bias Bias vector [m, 1], broadcast across all columns.
/// @param[in]  clamp_min Minimum value for clamping.
/// @param[in]  clamp_max Maximum value for clamping.
///
/// @note All matrices use float16_t elements.
/// @note A_packed and B_packed are pre-packed using the pack functions.
/// @note The bias vector has shape [m, 1] and is added to each column of the result.
/// @note k_idx_a and k_idx_b are used when A and B are sub-buffers with different sizes.
void tqt_run_conv_convcf_mx1bias_clamp_f16_f16p_f16p_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, float16_t clamp_min, float16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif
