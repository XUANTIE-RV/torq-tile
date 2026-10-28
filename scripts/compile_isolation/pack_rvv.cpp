//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "src/common/tqt_gemm_pack_rvv.h"

// External entry points and runtime dimensions force code generation for all tails.
extern "C" void check_pack_a(size_t m, size_t k, size_t ld, size_t packed_ld, size_t k_idx,
                             const void *src, void *dst)
{
#if TQT_CHECK_BITS == 16
    tqt_gemm_pack_mxk_8x1_e16_rvv(m, k, ld, packed_ld, k_idx, src, dst);
#elif TQT_CHECK_BITS == 32
    tqt_gemm_pack_mxk_8x1_e32_rvv(m, k, ld, packed_ld, k_idx, src, dst);
#elif TQT_CHECK_BITS == 8
    tqt_gemm_pack_mxk_4x1_e8_rvv(m, k, ld, packed_ld, k_idx, src, dst);
#else
#error Unsupported packing width.
#endif
}

extern "C" void check_pack_b(size_t n, size_t k, size_t ld, size_t packed_ld, size_t k_idx,
                             const void *src, void *dst)
{
#if TQT_CHECK_BITS == 16
    tqt_gemm_pack_nxk_1x3vl_e16_rvv(n, k, ld, packed_ld, k_idx, src, dst);
    tqt_gemm_pack_nxk_1x1vl_e16_rvv(n, k, ld, packed_ld, k_idx, src, dst);
#elif TQT_CHECK_BITS == 32
    tqt_gemm_pack_nxk_1x3vl_e32_rvv(n, k, ld, packed_ld, k_idx, src, dst);
#elif TQT_CHECK_BITS == 8
    tqt_gemm_pack_nxk_1x1vl_e8_rvv(n, k, ld, packed_ld, k_idx, src, dst);
    tqt_gemm_pack_nxk_1x2vl_e8mf2_rvv(n, k, ld, packed_ld, k_idx, src, dst);
#endif
    tqt_gemm_pack_kxn_1x3vl_rvv(n, k, ld, packed_ld, k_idx, src, dst, TQT_CHECK_BITS);
    tqt_gemm_pack_kxn_1x1vl_rvv(n, k, ld, packed_ld, k_idx, src, dst, TQT_CHECK_BITS);
}
