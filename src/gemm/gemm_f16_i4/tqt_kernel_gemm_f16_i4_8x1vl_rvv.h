//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif

#include "src/common/tqt_common.h"

// FP32 accumulators: v16-v31 (eight e32m2 rows). Scratch: v0-v3 for
// nibbles/FP16 LUT values, v4-v7 for decoded weights, v8-v9 for scales,
// v10-v11 for affine offsets, and v12-v13 for LUT byte indices.
// The symmetric path widens FP16 scales and LUT entries before arithmetic.

template <size_t MR, bool Packed, bool Affine>
static inline void tqt_run_kernel_gemm_f16_i4_8x1vl_rvv(
    size_t vl, size_t K, size_t bl, const float16_t *A, size_t lda, const uint8_t *B, size_t ldb,
    const void *scale_b, size_t ldsb, const float *offset_b, size_t ldob, const float *C,
    size_t ldc, float *D, size_t ldd, const float *bias, float clamp_min, float clamp_max,
    bool apply_clamp, const float16_t *lut)
{
    static_assert(MR >= 1 && MR <= 8, "MR must be in range [1, 8]");
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl));
    if constexpr (MR > 0) {
        if (C) {
            asm volatile("vle32.v v16, (%0)" : : "r"(C + 0 * ldc) : "v16", "v17", "memory");
        } else {
            asm volatile("vmv.v.i v16, 0" : : : "v16", "v17");
        }
    }
    if constexpr (MR > 1) {
        if (C) {
            asm volatile("vle32.v v18, (%0)" : : "r"(C + 1 * ldc) : "v18", "v19", "memory");
        } else {
            asm volatile("vmv.v.i v18, 0" : : : "v18", "v19");
        }
    }
    if constexpr (MR > 2) {
        if (C) {
            asm volatile("vle32.v v20, (%0)" : : "r"(C + 2 * ldc) : "v20", "v21", "memory");
        } else {
            asm volatile("vmv.v.i v20, 0" : : : "v20", "v21");
        }
    }
    if constexpr (MR > 3) {
        if (C) {
            asm volatile("vle32.v v22, (%0)" : : "r"(C + 3 * ldc) : "v22", "v23", "memory");
        } else {
            asm volatile("vmv.v.i v22, 0" : : : "v22", "v23");
        }
    }
    if constexpr (MR > 4) {
        if (C) {
            asm volatile("vle32.v v24, (%0)" : : "r"(C + 4 * ldc) : "v24", "v25", "memory");
        } else {
            asm volatile("vmv.v.i v24, 0" : : : "v24", "v25");
        }
    }
    if constexpr (MR > 5) {
        if (C) {
            asm volatile("vle32.v v26, (%0)" : : "r"(C + 5 * ldc) : "v26", "v27", "memory");
        } else {
            asm volatile("vmv.v.i v26, 0" : : : "v26", "v27");
        }
    }
    if constexpr (MR > 6) {
        if (C) {
            asm volatile("vle32.v v28, (%0)" : : "r"(C + 6 * ldc) : "v28", "v29", "memory");
        } else {
            asm volatile("vmv.v.i v28, 0" : : : "v28", "v29");
        }
    }
    if constexpr (MR > 7) {
        if (C) {
            asm volatile("vle32.v v30, (%0)" : : "r"(C + 7 * ldc) : "v30", "v31", "memory");
        } else {
            asm volatile("vmv.v.i v30, 0" : : : "v30", "v31");
        }
    }
    for (size_t blk = 0; blk < K / bl; ++blk) {
        if constexpr (Affine) {
            asm volatile(
                "vsetvli zero, %0, e32, m2, ta, ma\n\t"
                "vlse32.v v8, (%1), %2\n\t"
                "vlse32.v v10, (%3), %4\n\t"
                :
                : "r"(vl), "r"((const float *)scale_b + blk), "r"(ldsb * sizeof(float)),
                  "r"(offset_b + blk), "r"(ldob * sizeof(float))
                : "v8", "v9", "v10", "v11", "memory");
        } else {
            asm volatile(
                "vsetvli zero, %0, e16, m1, ta, ma\n\t"
                "vlse16.v v0, (%1), %2\n\t"
                "vfwcvt.f.f.v v8, v0\n\t"
                :
                : "r"(vl), "r"((const float16_t *)scale_b + blk), "r"(ldsb * sizeof(float16_t))
                : "v0", "v8", "v9", "memory");
        }
        for (size_t ki = blk * bl; ki < (blk + 1) * bl; ki += 2) {
            const uint8_t *b = B + (Packed ? (ki / 2) * vl : ki / 2);
            const size_t stride = Packed ? 1 : ldb / 2;
            asm volatile(
                "vsetvli zero, %0, e8, mf2, ta, ma\n\t"
                "vlse8.v v0, (%1), %2\n\t"
                "vand.vi v1, v0, 15\n\t"
                "vsrl.vi v2, v0, 4\n\t"
                :
                : "r"(vl), "r"(b), "r"(stride)
                : "v0", "v1", "v2", "memory");
            if (!Affine && lut) {
                asm volatile(
                    "vsetvli zero, %0, e16, m1, ta, ma\n\t"
                    "vzext.vf2 v12, v1\n\t"
                    "vzext.vf2 v13, v2\n\t"
                    "vsll.vi v12, v12, 1\n\t"
                    "vsll.vi v13, v13, 1\n\t"
                    "vluxei16.v v0, (%1), v12\n\t"
                    "vluxei16.v v1, (%1), v13\n\t"
                    "vfwcvt.f.f.v v4, v0\n\t"
                    "vfwcvt.f.f.v v6, v1\n\t"
                    "vsetvli zero, %0, e32, m2, ta, ma\n\t"
                    :
                    : "r"(vl), "r"(lut)
                    : "v0", "v1", "v4", "v5", "v6", "v7", "v12", "v13", "memory");
            } else {
                asm volatile(
                    "vsetvli zero, %0, e32, m2, ta, ma\n\t"
                    "vzext.vf4 v4, v1\n\t"
                    "vzext.vf4 v6, v2\n\t"
                    "vfcvt.f.x.v v4, v4\n\t"
                    "vfcvt.f.x.v v6, v6\n\t"
                    :
                    : "r"(vl)
                    : "v4", "v5", "v6", "v7");
                if constexpr (!Affine) {
                    asm volatile("vfadd.vf v4, v4, %0\n\tvfadd.vf v6, v6, %0"
                                 :
                                 : "f"(-8.0f)
                                 : "v4", "v5", "v6", "v7");
                }
            }
            asm volatile("vfmul.vv v4, v4, v8\n\tvfmul.vv v6, v6, v8" : : : "v4", "v5", "v6", "v7");
            if constexpr (Affine) {
                asm volatile("vfadd.vv v4, v4, v10\n\tvfadd.vv v6, v6, v10"
                             :
                             :
                             : "v4", "v5", "v6", "v7");
            }
            if constexpr (MR > 0) {
                const float lo = (float)A[Packed ? ki * MR + 0 : 0 * lda + ki];
                const float hi = (float)A[Packed ? (ki + 1) * MR + 0 : 0 * lda + ki + 1];
                asm volatile("vfmacc.vf v16, %0, v4\n\tvfmacc.vf v16, %1, v6"
                             :
                             : "f"(lo), "f"(hi)
                             : "v16", "v17");
            }
            if constexpr (MR > 1) {
                const float lo = (float)A[Packed ? ki * MR + 1 : 1 * lda + ki];
                const float hi = (float)A[Packed ? (ki + 1) * MR + 1 : 1 * lda + ki + 1];
                asm volatile("vfmacc.vf v18, %0, v4\n\tvfmacc.vf v18, %1, v6"
                             :
                             : "f"(lo), "f"(hi)
                             : "v18", "v19");
            }
            if constexpr (MR > 2) {
                const float lo = (float)A[Packed ? ki * MR + 2 : 2 * lda + ki];
                const float hi = (float)A[Packed ? (ki + 1) * MR + 2 : 2 * lda + ki + 1];
                asm volatile("vfmacc.vf v20, %0, v4\n\tvfmacc.vf v20, %1, v6"
                             :
                             : "f"(lo), "f"(hi)
                             : "v20", "v21");
            }
            if constexpr (MR > 3) {
                const float lo = (float)A[Packed ? ki * MR + 3 : 3 * lda + ki];
                const float hi = (float)A[Packed ? (ki + 1) * MR + 3 : 3 * lda + ki + 1];
                asm volatile("vfmacc.vf v22, %0, v4\n\tvfmacc.vf v22, %1, v6"
                             :
                             : "f"(lo), "f"(hi)
                             : "v22", "v23");
            }
            if constexpr (MR > 4) {
                const float lo = (float)A[Packed ? ki * MR + 4 : 4 * lda + ki];
                const float hi = (float)A[Packed ? (ki + 1) * MR + 4 : 4 * lda + ki + 1];
                asm volatile("vfmacc.vf v24, %0, v4\n\tvfmacc.vf v24, %1, v6"
                             :
                             : "f"(lo), "f"(hi)
                             : "v24", "v25");
            }
            if constexpr (MR > 5) {
                const float lo = (float)A[Packed ? ki * MR + 5 : 5 * lda + ki];
                const float hi = (float)A[Packed ? (ki + 1) * MR + 5 : 5 * lda + ki + 1];
                asm volatile("vfmacc.vf v26, %0, v4\n\tvfmacc.vf v26, %1, v6"
                             :
                             : "f"(lo), "f"(hi)
                             : "v26", "v27");
            }
            if constexpr (MR > 6) {
                const float lo = (float)A[Packed ? ki * MR + 6 : 6 * lda + ki];
                const float hi = (float)A[Packed ? (ki + 1) * MR + 6 : 6 * lda + ki + 1];
                asm volatile("vfmacc.vf v28, %0, v4\n\tvfmacc.vf v28, %1, v6"
                             :
                             : "f"(lo), "f"(hi)
                             : "v28", "v29");
            }
            if constexpr (MR > 7) {
                const float lo = (float)A[Packed ? ki * MR + 7 : 7 * lda + ki];
                const float hi = (float)A[Packed ? (ki + 1) * MR + 7 : 7 * lda + ki + 1];
                asm volatile("vfmacc.vf v30, %0, v4\n\tvfmacc.vf v30, %1, v6"
                             :
                             : "f"(lo), "f"(hi)
                             : "v30", "v31");
            }
        }
    }
    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" : : "r"(vl));
    if (bias) {
        asm volatile("vle32.v v0, (%0)" : : "r"(bias) : "v0", "v1", "memory");
        if constexpr (MR > 0)
            asm volatile("vfadd.vv v16, v16, v0" : : : "v16", "v17");
        if constexpr (MR > 1)
            asm volatile("vfadd.vv v18, v18, v0" : : : "v18", "v19");
        if constexpr (MR > 2)
            asm volatile("vfadd.vv v20, v20, v0" : : : "v20", "v21");
        if constexpr (MR > 3)
            asm volatile("vfadd.vv v22, v22, v0" : : : "v22", "v23");
        if constexpr (MR > 4)
            asm volatile("vfadd.vv v24, v24, v0" : : : "v24", "v25");
        if constexpr (MR > 5)
            asm volatile("vfadd.vv v26, v26, v0" : : : "v26", "v27");
        if constexpr (MR > 6)
            asm volatile("vfadd.vv v28, v28, v0" : : : "v28", "v29");
        if constexpr (MR > 7)
            asm volatile("vfadd.vv v30, v30, v0" : : : "v30", "v31");
    }
    if constexpr (MR > 0) {
        if (apply_clamp) {
            asm volatile("vfmax.vf v16, v16, %0\n\tvfmin.vf v16, v16, %1"
                         :
                         : "f"(clamp_min), "f"(clamp_max)
                         : "v16", "v17");
        }
        asm volatile("vse32.v v16, (%0)" : : "r"(D + 0 * ldd) : "memory");
    }
    if constexpr (MR > 1) {
        if (apply_clamp) {
            asm volatile("vfmax.vf v18, v18, %0\n\tvfmin.vf v18, v18, %1"
                         :
                         : "f"(clamp_min), "f"(clamp_max)
                         : "v18", "v19");
        }
        asm volatile("vse32.v v18, (%0)" : : "r"(D + 1 * ldd) : "memory");
    }
    if constexpr (MR > 2) {
        if (apply_clamp) {
            asm volatile("vfmax.vf v20, v20, %0\n\tvfmin.vf v20, v20, %1"
                         :
                         : "f"(clamp_min), "f"(clamp_max)
                         : "v20", "v21");
        }
        asm volatile("vse32.v v20, (%0)" : : "r"(D + 2 * ldd) : "memory");
    }
    if constexpr (MR > 3) {
        if (apply_clamp) {
            asm volatile("vfmax.vf v22, v22, %0\n\tvfmin.vf v22, v22, %1"
                         :
                         : "f"(clamp_min), "f"(clamp_max)
                         : "v22", "v23");
        }
        asm volatile("vse32.v v22, (%0)" : : "r"(D + 3 * ldd) : "memory");
    }
    if constexpr (MR > 4) {
        if (apply_clamp) {
            asm volatile("vfmax.vf v24, v24, %0\n\tvfmin.vf v24, v24, %1"
                         :
                         : "f"(clamp_min), "f"(clamp_max)
                         : "v24", "v25");
        }
        asm volatile("vse32.v v24, (%0)" : : "r"(D + 4 * ldd) : "memory");
    }
    if constexpr (MR > 5) {
        if (apply_clamp) {
            asm volatile("vfmax.vf v26, v26, %0\n\tvfmin.vf v26, v26, %1"
                         :
                         : "f"(clamp_min), "f"(clamp_max)
                         : "v26", "v27");
        }
        asm volatile("vse32.v v26, (%0)" : : "r"(D + 5 * ldd) : "memory");
    }
    if constexpr (MR > 6) {
        if (apply_clamp) {
            asm volatile("vfmax.vf v28, v28, %0\n\tvfmin.vf v28, v28, %1"
                         :
                         : "f"(clamp_min), "f"(clamp_max)
                         : "v28", "v29");
        }
        asm volatile("vse32.v v28, (%0)" : : "r"(D + 6 * ldd) : "memory");
    }
    if constexpr (MR > 7) {
        if (apply_clamp) {
            asm volatile("vfmax.vf v30, v30, %0\n\tvfmin.vf v30, v30, %1"
                         :
                         : "f"(clamp_min), "f"(clamp_max)
                         : "v30", "v31");
        }
        asm volatile("vse32.v v30, (%0)" : : "r"(D + 7 * ldd) : "memory");
    }
}
