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

/// Conv micro-kernel: channel-first, Mx1 bias, clamp, non-packed, FP16
/// RVV 8x3vl tile implementation.
///
/// A = weight [OC_per_group, K], B = im2col(activation) [K, N]
/// where K = IC_per_group * KD * KH * KW, N = OD * OH * OW

/// --------------------------------------------------

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value (8).
size_t tqt_get_m_step_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value (3 * vl, where vl = csrr_vlenb() / sizeof(float16_t)).
size_t tqt_get_n_step_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(void);

/// Gets the offset in bytes to the data element in the A matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] k_idx Column index.
/// @param[in] lda Leading dimension of A.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_offset_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(size_t m_idx, size_t k_idx,
                                                                   size_t lda);

/// Gets the offset in bytes to the data element in the C matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
///
/// @param[in] m_idx Row index.
///
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(size_t m_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldd Leading dimension of D.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                   size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(size_t m, size_t n);

/// Gets the size in bytes of the im2col buffer.
///
/// @param[in] k K dimension (IC_per_group * KD * KH * KW).
/// @param[in] n N dimension (OD * OH * OW).
///
/// @return The size in bytes of the im2col buffer.
size_t tqt_get_b_im2col_size_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(size_t k, size_t n);

/// Gets the offset in bytes to the data element in the im2col buffer.
///
/// @param[in] n_idx Column index.
/// @param[in] k_idx Row index.
/// @param[in] ldb_im2col Leading dimension of im2col buffer.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_b_im2col_offset_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(size_t n_idx,
                                                                          size_t k_idx,
                                                                          size_t ldb_im2col);

/// Runs the im2col operation to transform activation tensor.
///
/// Transforms the activation tensor into im2col format for convolution computation.
///
/// @param[in] group_idx Group index.
/// @param[in] activation Input activation tensor.
/// @param[out] im2col_buf Output im2col buffer.
/// @param[in] params Convolution parameters.
void tqt_run_im2col_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(size_t group_idx,
                                                               const void *activation,
                                                               void *im2col_buf,
                                                               const tqt_conv_params *params);

/// Runs the convolution microkernel with bias addition and clamp operation.
///
/// Computes: D = clamp(C + A*B + bias, clamp_min, clamp_max)
/// where A = weight [OC_per_group, K], B = im2col(activation) [K, N]
///
/// @param[in]  m Number of rows in A and D (output channels).
/// @param[in]  n Number of columns in B and D (OD * OH * OW).
/// @param[in]  k Common dimension (IC_per_group * KD * KH * KW).
/// @param[in]  A Input matrix A [m, k] (weight) in row-major format.
/// @param[in]  lda Leading dimension of A.
/// @param[in]  k_idx_a K index in A buffer.
/// @param[in]  B Input matrix B [k, n] (im2col activation) in row-major format.
/// @param[in]  ldb Leading dimension of B.
/// @param[in]  k_idx_b K index in B buffer.
/// @param[in]  C Input matrix C [m, n] in row-major format.
/// @param[in]  ldc Leading dimension of C.
/// @param[out] D Output matrix D [m, n] in row-major format.
/// @param[in]  ldd Leading dimension of D.
/// @param[in]  bias Bias vector [m, 1], broadcast across all columns.
/// @param[in]  clamp_min Minimum value for clamping.
/// @param[in]  clamp_max Maximum value for clamping.
///
/// @note All matrices are in row-major format with float16_t elements.
/// @note A is the weight matrix with shape [OC_per_group, K].
/// @note B is the im2col-transformed activation with shape [K, N].
/// @note The bias vector has shape [m, 1] and is added to each column of the result.
/// @note k_idx_a and k_idx_b are used when A and B are sub-buffers with different sizes.
void tqt_run_conv_convcf_mx1bias_clamp_f16_f16_f16_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float16_t clamp_min, float16_t clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif
