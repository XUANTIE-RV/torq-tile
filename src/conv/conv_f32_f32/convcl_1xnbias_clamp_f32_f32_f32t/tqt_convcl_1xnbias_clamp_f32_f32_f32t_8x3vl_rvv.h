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

/// Conv micro-kernel: channel-last, 1xN bias, clamp, non-packed, FP32 (B transposed)
/// RVV 8x3vl tile implementation.
///
/// A = im2col(activation) [M, K], B = weight [N, K] (transposed)
/// where M = OD * OH * OW, K = KD * KH * KW * IC_per_group, N = OC_per_group

/// --------------------------------------------------

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value (8).
size_t tqt_get_m_step_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value (3 * vl, where vl = csrr_vlenb() / sizeof(float)).
size_t tqt_get_n_step_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(void);

/// Gets the offset in bytes to the data element in the B matrix buffer.
///
/// @param[in] n_idx Column index.
/// @param[in] k_idx Row index.
/// @param[in] ldb Leading dimension of B.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_b_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t n_idx, size_t k_idx,
                                                                    size_t ldb);

/// Gets the offset in bytes to the data element in the C matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t n_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldd Leading dimension of D.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m_idx, size_t n_idx,
                                                                    size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m, size_t n);

/// Gets the size in bytes of the im2col buffer for activation matrix.
///
/// @param[in] m Number of rows in im2col buffer (M = OD * OH * OW).
/// @param[in] k Number of columns in im2col buffer (K = KD * KH * KW * IC_per_group).
///
/// @return The size in bytes of the im2col buffer.
size_t tqt_get_a_im2col_size_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m, size_t k);

/// Gets the offset in bytes to the data element in the im2col buffer.
///
/// @param[in] m_idx Row index in im2col buffer.
/// @param[in] k_idx Column index in im2col buffer.
/// @param[in] lda_im2col Leading dimension of im2col buffer.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_im2col_offset_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t m_idx,
                                                                           size_t k_idx,
                                                                           size_t lda_im2col);

/// Runs the im2col transformation for channel-last convolution.
///
/// Transforms the activation tensor from channel-last format to im2col format.
///
/// @param[in]  group_idx Group index for grouped convolution.
/// @param[in]  activation Input activation tensor in channel-last format.
/// @param[out] im2col_buf Output im2col buffer [M, K].
/// @param[in]  params Convolution parameters containing padding, stride, dilation, etc.
///
/// @note The activation tensor is expected to be in channel-last (NHWC) format.
/// @note The im2col buffer stores the transformed activation as [M, K] where
///       M = OD * OH * OW and K = KD * KH * KW * IC_per_group.
void tqt_run_im2col_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(size_t group_idx,
                                                                const void *activation,
                                                                void *im2col_buf,
                                                                const tqt_conv_params *params);

/// Runs the convolution microkernel with bias addition and clamp operation.
///
/// Computes: D = clamp(C + A*B + bias, clamp_min, clamp_max)
/// where A = im2col(activation) [M, K], B = weight [N, K] (transposed)
///
/// @param[in]  m Number of rows in A and D (M = OD * OH * OW).
/// @param[in]  n Number of columns in B and D (N = OC_per_group).
/// @param[in]  k Common dimension (K = KD * KH * KW * IC_per_group).
/// @param[in]  A Input matrix A [m, k] (im2col buffer) in row-major format.
/// @param[in]  lda Leading dimension of A.
/// @param[in]  k_idx_a K index in A buffer.
/// @param[in]  B Input matrix B [n, k] (weight) in transposed row-major format.
/// @param[in]  ldb Leading dimension of B.
/// @param[in]  k_idx_b K index in B buffer.
/// @param[in]  C Input matrix C [m, n] in row-major format.
/// @param[in]  ldc Leading dimension of C.
/// @param[out] D Output matrix D [m, n] in row-major format.
/// @param[in]  ldd Leading dimension of D.
/// @param[in]  bias Bias vector [1, n], broadcast across all rows.
/// @param[in]  clamp_min Minimum value for clamping.
/// @param[in]  clamp_max Maximum value for clamping.
///
/// @note All matrices are in row-major format with float elements.
/// @note B is in transposed format, stored as [n, k].
/// @note The bias vector has shape [1, n] and is added to each row of the result.
/// @note k_idx_a and k_idx_b are used when A and B are sub-buffers with different sizes.
void tqt_run_conv_convcl_1xnbias_clamp_f32_f32_f32t_8x3vl_rvv(
    size_t m, size_t n, size_t k, const void *A, size_t lda, size_t k_idx_a, const void *B,
    size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D, size_t ldd, const void *bias,
    float clamp_min, float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif
