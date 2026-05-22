//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"
#include "src/gemm/gemm_i8_i8/tqt_params_i8_i8.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Conv micro-kernel: channel-last, 1xN bias, requantize (int32->int8), clamp, packed, INT8
/// RVV 4x1vl tile implementation.
///
/// A = im2col(activation) [M, K] packed, B = weight [N, K] (transposed) packed
/// where M = OD * OH * OW, K = KD * KH * KW * IC_per_group, N = OC_per_group

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value (4).
size_t tqt_get_m_step_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value (vl, where vl = csrr_vlenb() / sizeof(int8)).
size_t tqt_get_n_step_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(void);

/// Gets the offset in bytes to the data element in the im2col packed A matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] k_idx Column index.
/// @param[in] lda Leading dimension of A.
/// @param[in] actual_m Actual number of rows being processed.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_a_im2colpack_offset_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t m_idx, size_t k_idx, size_t lda, size_t actual_m);

/// Gets the offset in bytes to the data element in the packed B matrix buffer.
///
/// @param[in] n_idx Row index.
/// @param[in] k_idx Column index.
/// @param[in] ldb Leading dimension of B.
/// @param[in] actual_n Actual number of columns being processed.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_b_packed_offset_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t n_idx, size_t k_idx, size_t ldb, size_t actual_n);

/// Gets the offset in bytes to the data element in the C matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldc Leading dimension of C.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_c_offset_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                     size_t n_idx,
                                                                                     size_t ldc);

/// Gets the offset in bytes to the data element in the bias vector buffer.
///
/// @param[in] n_idx Column index.
///
/// @return The offset in bytes to the bias element.
size_t tqt_get_bias_offset_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t n_idx);

/// Gets the offset in bytes to the data element in the D (output) matrix buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] n_idx Column index.
/// @param[in] ldd Leading dimension of D.
///
/// @return The offset in bytes to the data element.
size_t tqt_get_d_offset_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t m_idx,
                                                                                     size_t n_idx,
                                                                                     size_t ldd);

/// Gets the size in bytes of the D (output) matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the output matrix buffer.
size_t tqt_get_d_size_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t m,
                                                                                   size_t n);

/// Gets the size in bytes of the im2col packed A matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] k Number of columns.
///
/// @return The size in bytes of the packed A matrix buffer.
size_t tqt_get_a_im2colpack_size_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t m, size_t k);

/// Gets the size in bytes of the packed B matrix buffer.
///
/// @param[in] n Number of rows.
/// @param[in] k Number of columns.
///
/// @return The size in bytes of the packed B matrix buffer.
size_t tqt_get_b_packed_size_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(size_t n,
                                                                                          size_t k);

/// Packs transposed matrix B (weight) for optimal memory access.
///
/// @param[in]  n Number of rows in B (transposed).
/// @param[in]  k Number of columns in B (transposed).
/// @param[in]  ldb Leading dimension of B.
/// @param[in]  ldb_packed Leading dimension of packed B buffer.
/// @param[in]  k_idx Starting k index for packing.
/// @param[in]  B Input matrix B [n, k] in row-major format (transposed weight, int8).
/// @param[out] B_packed Packed matrix B buffer.
void tqt_run_bt_pack_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t n, size_t k, size_t ldb, size_t ldb_packed, size_t k_idx, const void *B, void *B_packed);

/// Performs im2col transformation and packs activation matrix for optimal memory access.
///
/// @param[in]  group_idx Group index for grouped convolution.
/// @param[in]  activation Input activation tensor (int8).
/// @param[in]  lda_packed Leading dimension of packed A buffer.
/// @param[in]  k_idx Starting k index for packing.
/// @param[out] A_packed Packed matrix A buffer (im2col transformed).
/// @param[in]  params Convolution parameters (padding, stride, dilation, etc.).
void tqt_run_a_im2colpack_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t group_idx, const void *activation, size_t lda_packed, size_t k_idx, void *A_packed,
    const tqt_conv_params *params);

/// Runs the convolution microkernel with packed matrices, bias addition, requantization and clamp.
///
/// Computes: D = clamp(requantize(C + A_packed*B_packed + bias), clamp_min, clamp_max)
///
/// @param[in]  m Number of rows in A and D (M = OD * OH * OW).
/// @param[in]  n Number of rows in B and columns in D (N = OC_per_group).
/// @param[in]  k Common dimension (K = KD * KH * KW * IC_per_group).
/// @param[in]  A_packed Packed input matrix A (im2col transformed activation, int8).
/// @param[in]  lda Leading dimension of original A.
/// @param[in]  k_idx_a K index in A buffer (for sub-buffer scenarios).
/// @param[in]  B_packed Packed input matrix B (transposed weight, int8).
/// @param[in]  ldb Leading dimension of original B.
/// @param[in]  k_idx_b K index in B buffer (for sub-buffer scenarios).
/// @param[in]  C Input matrix C [m, n] in row-major format (int32).
/// @param[in]  ldc Leading dimension of C.
/// @param[out] D Output matrix D [m, n] in row-major format (int8).
/// @param[in]  ldd Leading dimension of D.
/// @param[in]  bias Bias vector [1, n] (int32), broadcast across all rows.
/// @param[in]  clamp_min Minimum value for clamping (int8).
/// @param[in]  clamp_max Maximum value for clamping (int8).
/// @param[in]  params Quantization parameters.
///
/// @note A_packed is im2col transformed and packed activation, B_packed is transposed and packed
/// weight.
/// @note The bias vector has shape [1, n] and is added to each row of the result.
/// @note k_idx_a and k_idx_b are used when A and B are sub-buffers with different sizes.
void tqt_run_conv_convcl_1xnbias_reqi32toi8_clamp_qai8_qai8p_qsi8cxp_4x1vl_rvv(
    size_t m, size_t n, size_t k, const void *A_packed, size_t lda, size_t k_idx_a,
    const void *B_packed, size_t ldb, size_t k_idx_b, const void *C, size_t ldc, void *D,
    size_t ldd, const void *bias, int8_t clamp_min, int8_t clamp_max,
    const struct tqt_qai8_qai8_qsi8cx_params *params);

#ifdef __cplusplus
}  // extern "C"
#endif
