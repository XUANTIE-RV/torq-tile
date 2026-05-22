//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif  // Architectural features check.

#include "src/common/tqt_common.h"
#include "src/common/tqt_conv.h"

// ============================================================================
// DWConv Pack/Unpack Helpers (single batch)
// ============================================================================
//
// Packed layout: [C1, D, H, W, C0]
//   C0 = vlmax for the dtype/LMUL=1 (e.g., fp32 + VLEN=128 -> C0=4)
//   C1 = ceil(C / C0)
//   Tail block (c1 = C1-1) holds tail_vl = C - (C1-1) * C0 elements (no zero-padding).
//
// All functions process ONE batch. Caller loops over batch dimension.
// ============================================================================

// ----------------------------------------------------------------------------
// Size helpers
// ----------------------------------------------------------------------------

/// Bytes for one batch of packed input/output (NC1DHWC0): equals IC * spatial * sizeof(T).
/// Tail block is compact (no padding), so total element count == IC * spatial.
static inline size_t tqt_dwconv_packed_tensor_bytes_f32_rvv(size_t channel, size_t spatial)
{
    return channel * spatial * sizeof(float);
}

/// Bytes for packed weight: equals OC * KD*KH*KW * sizeof(T) (single instance, no batch).
static inline size_t tqt_dwconv_packed_weight_bytes_f32_rvv(size_t channel, size_t kernel_size)
{
    return channel * kernel_size * sizeof(float);
}

// ----------------------------------------------------------------------------
// pack_input_cf : NCDHW -> NC1DHWC0  (single batch)
// ----------------------------------------------------------------------------
//   src layout : [IC, ID, IH, IW]   contiguous, channel-first
//   dst layout : [C1, ID, IH, IW, C0]   tail block compact (vl = tail_vl)
static inline void tqt_dwconv_pack_input_cf_f32_rvv(const float *input, float *packed,
                                                    const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t spatial = params->input_shape[0] * params->input_shape[1] * params->input_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t C1 = (IC + C0 - 1) / C0;

    size_t dst_offset = 0;
    for (size_t c1 = 0; c1 < C1; ++c1) {
        const size_t requested = (c1 == C1 - 1) ? (IC - c1 * C0) : C0;
        const size_t vl = __riscv_vsetvl_e32m1(requested);
        const float *src_base = input + c1 * C0 * spatial;
        float *dst_base = packed + dst_offset;

        // Strided gather: pull `vl` channels (each separated by `spatial` elements)
        // for the same spatial position; store them contiguously to packed buffer.
        for (size_t pos = 0; pos < spatial; ++pos) {
            vfloat32m1_t v =
                __riscv_vlse32_v_f32m1(src_base + pos, (ptrdiff_t)(spatial * sizeof(float)), vl);
            __riscv_vse32_v_f32m1(dst_base + pos * vl, v, vl);
        }

        dst_offset += spatial * vl;
    }
}

// ----------------------------------------------------------------------------
// pack_input_cl : NDHWC -> NC1DHWC0  (single batch)
// ----------------------------------------------------------------------------
//   src layout : [ID, IH, IW, IC]   contiguous, channel-last
//   dst layout : [C1, ID, IH, IW, C0]
static inline void tqt_dwconv_pack_input_cl_f32_rvv(const float *input, float *packed,
                                                    const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t spatial = params->input_shape[0] * params->input_shape[1] * params->input_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t C1 = (IC + C0 - 1) / C0;

    size_t dst_offset = 0;
    for (size_t c1 = 0; c1 < C1; ++c1) {
        const size_t requested = (c1 == C1 - 1) ? (IC - c1 * C0) : C0;
        const size_t vl = __riscv_vsetvl_e32m1(requested);
        float *dst_base = packed + dst_offset;

        // Channels are contiguous in src; just take slice [c1*C0, c1*C0+vl) at each pos.
        for (size_t pos = 0; pos < spatial; ++pos) {
            const float *src = input + pos * IC + c1 * C0;
            vfloat32m1_t v = __riscv_vle32_v_f32m1(src, vl);
            __riscv_vse32_v_f32m1(dst_base + pos * vl, v, vl);
        }

        dst_offset += spatial * vl;
    }
}

// ----------------------------------------------------------------------------
// pack_weight_cf : [OC, 1, KD, KH, KW] -> [OC1, KD, KH, KW, OC0]
// ----------------------------------------------------------------------------
//   PyTorch dwconv weight: 2nd dim (IC_per_group) is always 1, treated as [OC, KD*KH*KW].
//   OC ordering: oc = ic + m * IC (PyTorch convention). dm encoded via OC1 = dm * IC1.
//   For dm > 1 with tail_vl < C0: each dm cycle has IC1 blocks (IC1-1 full + 1 tail);
//   block oc1 starts at oc_base = (oc1/IC1)*IC + (oc1%IC1)*C0, i.e. NOT oc1*C0.
static inline void tqt_dwconv_pack_weight_cf_f32_rvv(const float *weight, float *packed,
                                                     const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t kernel_size =
        params->kernel_shape[0] * params->kernel_shape[1] * params->kernel_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;  // dm * IC1

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e32m1(requested);
        const size_t oc_base = m_idx * IC + ic1 * C0;

        const float *src_base = weight + oc_base * kernel_size;
        float *dst_base = packed + oc_base * kernel_size;

        for (size_t k = 0; k < kernel_size; ++k) {
            vfloat32m1_t v =
                __riscv_vlse32_v_f32m1(src_base + k, (ptrdiff_t)(kernel_size * sizeof(float)), vl);
            __riscv_vse32_v_f32m1(dst_base + k * vl, v, vl);
        }
    }
}

// ----------------------------------------------------------------------------
// pack_weight_cl : [KD, KH, KW, 1, OC] -> [OC1, KD, KH, KW, OC0]
// ----------------------------------------------------------------------------
//   HWIO-style weight (TF NHWC convention). OC in innermost dim, contiguous.
static inline void tqt_dwconv_pack_weight_cl_f32_rvv(const float *weight, float *packed,
                                                     const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t kernel_size =
        params->kernel_shape[0] * params->kernel_shape[1] * params->kernel_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e32m1(requested);
        const size_t oc_base = m_idx * IC + ic1 * C0;

        float *dst_base = packed + oc_base * kernel_size;

        for (size_t k = 0; k < kernel_size; ++k) {
            const float *src = weight + k * OC + oc_base;
            vfloat32m1_t v = __riscv_vle32_v_f32m1(src, vl);
            __riscv_vse32_v_f32m1(dst_base + k * vl, v, vl);
        }
    }
}

// ----------------------------------------------------------------------------
// unpack_output_cf : NC1DHWC0 -> NCDHW  (single batch)
// ----------------------------------------------------------------------------
static inline void tqt_dwconv_unpack_output_cf_f32_rvv(const float *packed, float *output,
                                                       const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t spatial =
        params->output_shape[0] * params->output_shape[1] * params->output_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e32m1(requested);
        const size_t oc_base = m_idx * IC + ic1 * C0;

        const float *src_base = packed + oc_base * spatial;
        float *dst_base = output + oc_base * spatial;

        for (size_t pos = 0; pos < spatial; ++pos) {
            vfloat32m1_t v = __riscv_vle32_v_f32m1(src_base + pos * vl, vl);
            __riscv_vsse32_v_f32m1(dst_base + pos, (ptrdiff_t)(spatial * sizeof(float)), v, vl);
        }
    }
}

// ----------------------------------------------------------------------------
// unpack_output_cl : NC1DHWC0 -> NDHWC  (single batch)
// ----------------------------------------------------------------------------
static inline void tqt_dwconv_unpack_output_cl_f32_rvv(const float *packed, float *output,
                                                       const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t spatial =
        params->output_shape[0] * params->output_shape[1] * params->output_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e32m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e32m1(requested);
        const size_t oc_base = m_idx * IC + ic1 * C0;

        const float *src_base = packed + oc_base * spatial;

        for (size_t pos = 0; pos < spatial; ++pos) {
            float *dst = output + pos * OC + oc_base;
            vfloat32m1_t v = __riscv_vle32_v_f32m1(src_base + pos * vl, vl);
            __riscv_vse32_v_f32m1(dst, v, vl);
        }
    }
}

// ============================================================================
// FP16 variants (mirror of fp32 above; require zfh + zvfh).
// ============================================================================

#if defined(__riscv_zfh) && defined(__riscv_zvfh)

static inline size_t tqt_dwconv_packed_tensor_bytes_f16_rvv(size_t channel, size_t spatial)
{
    return channel * spatial * sizeof(float16_t);
}

static inline size_t tqt_dwconv_packed_weight_bytes_f16_rvv(size_t channel, size_t kernel_size)
{
    return channel * kernel_size * sizeof(float16_t);
}

static inline void tqt_dwconv_pack_input_cf_f16_rvv(const float16_t *input, float16_t *packed,
                                                    const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t spatial = params->input_shape[0] * params->input_shape[1] * params->input_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e16m1();
    const size_t C1 = (IC + C0 - 1) / C0;

    size_t dst_offset = 0;
    for (size_t c1 = 0; c1 < C1; ++c1) {
        const size_t requested = (c1 == C1 - 1) ? (IC - c1 * C0) : C0;
        const size_t vl = __riscv_vsetvl_e16m1(requested);
        const float16_t *src_base = input + c1 * C0 * spatial;
        float16_t *dst_base = packed + dst_offset;

        for (size_t pos = 0; pos < spatial; ++pos) {
            vfloat16m1_t v = __riscv_vlse16_v_f16m1(src_base + pos,
                                                    (ptrdiff_t)(spatial * sizeof(float16_t)), vl);
            __riscv_vse16_v_f16m1(dst_base + pos * vl, v, vl);
        }

        dst_offset += spatial * vl;
    }
}

static inline void tqt_dwconv_pack_input_cl_f16_rvv(const float16_t *input, float16_t *packed,
                                                    const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t spatial = params->input_shape[0] * params->input_shape[1] * params->input_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e16m1();
    const size_t C1 = (IC + C0 - 1) / C0;

    size_t dst_offset = 0;
    for (size_t c1 = 0; c1 < C1; ++c1) {
        const size_t requested = (c1 == C1 - 1) ? (IC - c1 * C0) : C0;
        const size_t vl = __riscv_vsetvl_e16m1(requested);
        float16_t *dst_base = packed + dst_offset;

        for (size_t pos = 0; pos < spatial; ++pos) {
            const float16_t *src = input + pos * IC + c1 * C0;
            vfloat16m1_t v = __riscv_vle16_v_f16m1(src, vl);
            __riscv_vse16_v_f16m1(dst_base + pos * vl, v, vl);
        }

        dst_offset += spatial * vl;
    }
}

static inline void tqt_dwconv_pack_weight_cf_f16_rvv(const float16_t *weight, float16_t *packed,
                                                     const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t kernel_size =
        params->kernel_shape[0] * params->kernel_shape[1] * params->kernel_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e16m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e16m1(requested);
        const size_t oc_base = m_idx * IC + ic1 * C0;

        const float16_t *src_base = weight + oc_base * kernel_size;
        float16_t *dst_base = packed + oc_base * kernel_size;

        for (size_t k = 0; k < kernel_size; ++k) {
            vfloat16m1_t v = __riscv_vlse16_v_f16m1(
                src_base + k, (ptrdiff_t)(kernel_size * sizeof(float16_t)), vl);
            __riscv_vse16_v_f16m1(dst_base + k * vl, v, vl);
        }
    }
}

static inline void tqt_dwconv_pack_weight_cl_f16_rvv(const float16_t *weight, float16_t *packed,
                                                     const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t kernel_size =
        params->kernel_shape[0] * params->kernel_shape[1] * params->kernel_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e16m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e16m1(requested);
        const size_t oc_base = m_idx * IC + ic1 * C0;

        float16_t *dst_base = packed + oc_base * kernel_size;

        for (size_t k = 0; k < kernel_size; ++k) {
            const float16_t *src = weight + k * OC + oc_base;
            vfloat16m1_t v = __riscv_vle16_v_f16m1(src, vl);
            __riscv_vse16_v_f16m1(dst_base + k * vl, v, vl);
        }
    }
}

static inline void tqt_dwconv_unpack_output_cf_f16_rvv(const float16_t *packed, float16_t *output,
                                                       const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t spatial =
        params->output_shape[0] * params->output_shape[1] * params->output_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e16m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e16m1(requested);
        const size_t oc_base = m_idx * IC + ic1 * C0;

        const float16_t *src_base = packed + oc_base * spatial;
        float16_t *dst_base = output + oc_base * spatial;

        for (size_t pos = 0; pos < spatial; ++pos) {
            vfloat16m1_t v = __riscv_vle16_v_f16m1(src_base + pos * vl, vl);
            __riscv_vsse16_v_f16m1(dst_base + pos, (ptrdiff_t)(spatial * sizeof(float16_t)), v, vl);
        }
    }
}

static inline void tqt_dwconv_unpack_output_cl_f16_rvv(const float16_t *packed, float16_t *output,
                                                       const tqt_conv_params *params)
{
    const size_t IC = params->input_channel;
    const size_t OC = params->output_channel;
    const size_t spatial =
        params->output_shape[0] * params->output_shape[1] * params->output_shape[2];
    const size_t C0 = __riscv_vsetvlmax_e16m1();
    const size_t IC1 = (IC + C0 - 1) / C0;
    const size_t tail_vl = IC - (IC1 - 1) * C0;
    const size_t OC1 = (OC / IC) * IC1;

    for (size_t oc1 = 0; oc1 < OC1; ++oc1) {
        const size_t ic1 = oc1 % IC1;
        const size_t m_idx = oc1 / IC1;
        const size_t requested = (ic1 == IC1 - 1) ? tail_vl : C0;
        const size_t vl = __riscv_vsetvl_e16m1(requested);
        const size_t oc_base = m_idx * IC + ic1 * C0;

        const float16_t *src_base = packed + oc_base * spatial;

        for (size_t pos = 0; pos < spatial; ++pos) {
            float16_t *dst = output + pos * OC + oc_base;
            vfloat16m1_t v = __riscv_vle16_v_f16m1(src_base + pos * vl, vl);
            __riscv_vse16_v_f16m1(dst, v, vl);
        }
    }
}

#endif  // __riscv_zfh && __riscv_zvfh
