<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# TORQ-Tile micro-kernel tables

This page lists every micro-kernel variant currently shipped by TORQ-Tile. The catalog is grouped by operator family, then by data-type triple. Each row describes one micro-kernel base (a unique combination of operator, bias mode, fused operations, data types, and layout suffixes), together with all `(tile, arch)` implementations that exist for that base.

If you are reading a kernel name for the first time, also see the [naming convention](naming_convention.md), which defines every token used in the tables below.

Categories:

- [GEMM](#gemm-micro-kernels)
  - [FP32](#fp32-gemm_f32_f32)
  - [W16A32](#w16a32-gemm_f32_f16-and-gemm_f32_bf16)
  - [FP16](#fp16-gemm_f16_f16)
  - [BF16](#bf16-gemm_bf16_bf16)
  - [FP64](#fp64-gemm_f64_f64)
  - [INT8](#int8-gemm_i8_i8)
  - [UINT8](#uint8-gemm_u8_u8)
  - [INT4 W4A16](#int4-w4a16-gemm_f16_i4)
  - [Quantized-activation block GEMM (RVV and Ztt)](#quantized-activation-block-gemm-rvv-and-ztt)
  - [INT4 W4A8](#int4-w4a8-gemm_i8_i4)
  - [INT2 W2A8](#int2-w2a8-gemm_i8_i2)
- [Convolution (im2col + GEMM)](#convolution-im2col--gemm-micro-kernels)
- [Implicit GEMM convolution](#implicit-gemm-convolution-micro-kernels)
- [Depthwise convolution](#depthwise-convolution-micro-kernels)
- [Elementwise](#elementwise-micro-kernels)
- [Reduction](#reduction-micro-kernels)
- [Activation](#activation-micro-kernels)

Notation used in the tables:

- `D / A / B` columns give the destination and the two source operand types, including any quantization metadata and layout suffix (`t` = transposed, `p` = packed).
- The `Tile × Arch` column lists every implementation; each entry maps to one `tqt_<KERNEL_BASE>_<tile>_<arch>.cpp` source file.
- The `Family path` column is the directory under [`src/`](../../src/) that contains the kernel base.

---

## GEMM micro-kernels

Operator: `gemm`. Primary compute is a dense matrix multiplication `D = A · B + bias`, optionally with i32→i8 requantization and a clamp on the output.

Operator: `sgemm`. BLAS-style single-precision GEMM: `D = alpha * A · B + beta * C`, no bias/clamp; alpha and beta are scalar arguments to the run function.

Operator: `dgemm`. BLAS-style double-precision GEMM: same semantics as `sgemm` but with double-precision floats.

### FP32 (`gemm_f32_f32`)

Family path: [`src/gemm/gemm_f32_f32`](../../src/gemm/gemm_f32_f32).

#### Standard GEMM (D = A · B + bias, clamp)

| Bias    | Fused | D   | A    | B    | Tile × Arch                                             |
|---------|-------|-----|------|------|---------------------------------------------------------|
| 1xnbias | clamp | f32 | f32  | f32  | `8x3vl_rvv`, `4x4vl_rvv`, `2x8vl_rvv`, `1nlenxpnlen_ztt` |
| 1xnbias | clamp | f32 | f32  | f32t | `8x3vl_rvv`, `4x4vl_rvv`, `2x8vl_rvv`, `1nlenxpnlen_ztt` |
| 1xnbias | clamp | f32 | f32  | f32p | `8x3vl_rvv`                                             |
| 1xnbias | clamp | f32 | f32p | f32p | `8x3vl_rvv`, `4x4vl_rvv`, `2x8vl_rvv`                   |
| 1xnbias | clamp | f32p | f32p | f32p | `1nlenxpnlen_ztt` |
| mx1bias | clamp | f32 | f32  | f32  | `8x3vl_rvv`                                             |
| mx1bias | clamp | f32 | f32p | f32p | `8x3vl_rvv`                                             |

#### BLAS-style sgemm (D = alpha·A·B + beta·C)

| D   | A    | B    | Tile × Arch                        |
|-----|------|------|------------------------------------|
| f32 | f32  | f32  | `8x3vl_rvv`    |
| f32 | f32  | f32t | `8x3vl_rvv`    |
| f32 | f32t | f32  | `8x3vl_rvv`    |
| f32 | f32t | f32t | `8x3vl_rvv`    |

### W16A32 (`gemm_f32_f16` and `gemm_f32_bf16`)

Family paths: [`src/gemm/gemm_f32_f16`](../../src/gemm/gemm_f32_f16) and
[`src/gemm/gemm_f32_bf16`](../../src/gemm/gemm_f32_bf16).

The names describe the public API types: A is FP32, B is F16/BF16, and D is
FP32. RVV preserves FP32 A and widens B before FP32 FMA.

| Bias    | Fused | D   | A    | B     | Tile × Arch                              |
|---------|-------|-----|------|-------|------------------------------------------|
| 1xnbias | clamp | f32 | f32  | f16t  | `8x1vl_rvv`          |
| 1xnbias | clamp | f32 | f32p | f16p  | `8x1vl_rvv`          |
| 1xnbias | clamp | f32 | f32  | bf16t | `8x1vl_rvv`          |
| 1xnbias | clamp | f32 | f32p | bf16p | `8x1vl_rvv`          |

For packed RVV, the physical A panel remains FP32. Packed buffers are
variant-, tile-, and vector-length-specific; use the matching size and offset getters.

### FP16 (`gemm_f16_f16`)

Family path: [`src/gemm/gemm_f16_f16`](../../src/gemm/gemm_f16_f16).

| Bias    | Fused | D   | A    | B    | Tile × Arch                                           |
|---------|-------|-----|------|------|-------------------------------------------------------|
| 1xnbias | clamp | f16 | f16  | f16  | `8x3vl_rvv`, `4x4vl_rvv`, `2x8vl_rvv`, `8x1vl_f32acc_rvv`, `1nlenxpnlen_ztt`, `1nlenxpnlen_f32acc_ztt` |
| 1xnbias | clamp | f16 | f16  | f16t | `8x3vl_rvv`, `4x4vl_rvv`, `2x8vl_rvv`, `8x1vl_f32acc_rvv`, `1nlenxpnlen_ztt`, `1nlenxpnlen_f32acc_ztt` |
| 1xnbias | clamp | f16p | f16p | f16p | `1nlenxpnlen_ztt`, `1nlenxpnlen_f32acc_ztt` |
| 1xnbias | clamp | f16 | f16  | f16p | `8x3vl_rvv`                        |
| 1xnbias | clamp | f16 | f16p | f16p | `8x3vl_rvv`, `4x4vl_rvv`, `2x8vl_rvv`, `8x1vl_f32acc_rvv` |
| mx1bias | clamp | f16 | f16  | f16  | `8x3vl_rvv`, `8x1vl_f32acc_rvv`                        |
| mx1bias | clamp | f16 | f16p | f16p | `8x3vl_rvv`, `8x1vl_f32acc_rvv`                        |

### BF16 (`gemm_bf16_bf16`)

Family path: [`src/gemm/gemm_bf16_bf16`](../../src/gemm/gemm_bf16_bf16).

#### D = bf16

| Bias    | Fused | D    | A     | B     | Tile × Arch                        |
|---------|-------|------|-------|-------|------------------------------------|
| 1xnbias | clamp | bf16 | bf16  | bf16  | `8x1vl_rvv`, `1nlenxpnlen_ztt`, `1nlenxpnlen_f32acc_ztt` |
| 1xnbias | clamp | bf16 | bf16  | bf16t | `8x1vl_rvv`, `1nlenxpnlen_ztt`, `1nlenxpnlen_f32acc_ztt` |
| 1xnbias | clamp | bf16p | bf16p | bf16p | `1nlenxpnlen_ztt`, `1nlenxpnlen_f32acc_ztt` |
| 1xnbias | clamp | bf16 | bf16  | bf16p | `8x1vl_rvv`    |
| 1xnbias | clamp | bf16 | bf16p | bf16p | `8x1vl_rvv` |
| mx1bias | clamp | bf16 | bf16  | bf16  | `8x1vl_rvv` |
| mx1bias | clamp | bf16 | bf16p | bf16p | `8x1vl_rvv` |

#### D = f16 (F32 accumulation, narrowing output)

| Bias    | Fused | D   | A     | B     | Tile × Arch   |
|---------|-------|-----|-------|-------|---------------|
| 1xnbias | clamp | f16 | bf16 | bf16 | `8x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | bf16 | bf16 | `8x1vl_rvv` |
| 1xnbias | clamp | f16 | bf16 | bf16t | `8x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | bf16 | bf16t | `8x1vl_rvv` |
| 1xnbias | clamp | f16 | bf16p | bf16p | `8x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | bf16p | bf16p | `8x1vl_rvv` |

#### D = f32 (widening output)

| Bias    | Fused | D   | A     | B     | Tile × Arch                        |
|---------|-------|-----|-------|-------|------------------------------------|
| 1xnbias | clamp | f32 | bf16  | bf16  | `8x1vl_rvv`                        |
| 1xnbias | clamp | f32 | bf16  | bf16t | `8x1vl_rvv`    |
| 1xnbias | clamp | f32 | bf16p | bf16p | `8x1vl_rvv`    |

### FP64 (`gemm_f64_f64`)

Family path: [`src/gemm/gemm_f64_f64`](../../src/gemm/gemm_f64_f64).

BLAS-style dgemm: `D = alpha * A · B + beta * C`, double-precision.

| D   | A    | B    | Tile × Arch  |
|-----|------|------|--------------|
| f64 | f64  | f64  | `8x3vl_rvv`  |
| f64 | f64  | f64t | `8x3vl_rvv`  |
| f64 | f64t | f64  | `8x3vl_rvv`  |
| f64 | f64t | f64t | `8x3vl_rvv`  |

### INT8 (`gemm_i8_i8`)

Family path: [`src/gemm/gemm_i8_i8`](../../src/gemm/gemm_i8_i8).

#### Plain i32 accumulator output (no requantization)

| Bias | Fused | D   | A   | B   | Tile × Arch                        |
|------|-------|-----|-----|-----|------------------------------------|
| —    | clamp | i32 | i8  | i8  | `4x1vl_rvv`    |
| —    | clamp | i32 | i8p | i8p | `4x1vl_rvv`    |

#### Requantized i8 output (reqi32toi8)

| Bias    | Fused              | D    | A       | B        | Tile × Arch                        |
|---------|--------------------|------|---------|----------|------------------------------------|
| 1xnbias | reqi32toi8 + clamp | qai8 | qai8    | qsi8cxt  | `4x1vl_rvv`    |
| 1xnbias | reqi32toi8 + clamp | qai8 | qai8    | qsi8cxp  | `4x1vl_rvv`    |
| 1xnbias | reqi32toi8 + clamp | qai8 | qai8p   | qsi8cxp  | `4x1vl_rvv`    |
| mx1bias | reqi32toi8 + clamp | qai8 | qsi8dx  | qai8     | `4x1vl_rvv`    |
| mx1bias | reqi32toi8 + clamp | qai8 | qsi8dxp | qai8p    | `4x1vl_rvv`    |

#### Dequantized fp output (no requantization, fp clamp)

| Bias    | Fused | D   | A        | B        | Tile × Arch  |
|---------|-------|-----|----------|----------|--------------|
| 1xnbias | clamp | f32 | qai8dx   | qsi8cxt  | `4x1vl_rvv`  |
| 1xnbias | clamp | f32 | qai8dxp  | qsi8cxp  | `4x1vl_rvv`  |
| 1xnbias | clamp | f16 | qai8dx | qsi8cxt | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qai8dx | qsi8cxt | `4x1vl_rvv` |
| 1xnbias | clamp | f16 | qai8dxp | qsi8cxp | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qai8dxp | qsi8cxp | `4x1vl_rvv` |

### UINT8 (`gemm_u8_u8`)

Family path: [`src/gemm/gemm_u8_u8`](../../src/gemm/gemm_u8_u8).

Unsigned 8-bit integer GEMM with dual bias (both per-M and per-N bias simultaneously). Note: `u8` denotes raw unsigned uint8 (not quantized int8 stored as unsigned).

#### D = f32 (dequantized output with clamp)

| Bias              | Fused | D   | A   | B   | Tile × Arch                        |
|-------------------|-------|-----|-----|-----|------------------------------------|
| mx1bias + 1xnbias | clamp | f32 | u8  | u8t | `4x1vl_rvv`    |
| mx1bias + 1xnbias | clamp | f32 | u8p | u8p | `4x1vl_rvv`    |

#### D = i32 (raw accumulator output, no clamp)

| Bias              | Fused | D   | A   | B   | Tile × Arch                        |
|-------------------|-------|-----|-----|-----|------------------------------------|
| mx1bias + 1xnbias | —     | i32 | u8  | u8t | `4x1vl_rvv`    |
| mx1bias + 1xnbias | —     | i32 | u8p | u8p | `4x1vl_rvv`    |

### INT4 W4A16 (`gemm_f16_i4`)

Family path: [`src/gemm/gemm_f16_i4`](../../src/gemm/gemm_f16_i4).

W4A16: FP16 activation × INT4 weight with symmetric LUT-capable (`qsi4c2`) or
independent affine (`qai4c2`) block dequantization. `8x1vl_rvv` accumulates in F32;
`8x2vl_rvv` retains F16 intermediate precision.

| Bias    | Fused | D   | A    | B       | Tile × Arch  |
|---------|-------|-----|------|---------|--------------|
| 1xnbias | clamp | f16 | f16  | qsi4c2t | `8x2vl_rvv`  |
| 1xnbias | clamp | f16 | f16p | qsi4c2p | `8x2vl_rvv`  |
| 1xnbias | clamp | f32 | f16  | qsi4c2t | `8x2vl_rvv`, `8x1vl_rvv` |
| 1xnbias | clamp | f32 | f16p | qsi4c2p | `8x2vl_rvv`, `8x1vl_rvv` |
| 1xnbias | clamp | f32 | f16  | qai4c2t | `8x1vl_rvv` |
| 1xnbias | clamp | f32 | f16p | qai4c2p | `8x1vl_rvv` |

### Quantized-Activation Block GEMM (RVV and Ztt)

These packed variants accept the activation code and metadata contract used
by llama.cpp's quantized dot paths instead of dequantized f32/f16/bf16 A.
Q4_K/Q5_K/Q6_K use Q8_K-compatible `qsi8d256`; Q8_0 uses
Q8_0-compatible `qsi8d32`. Q5_K and Q6_K provide both byte-expanded and
group-local plane-packed execution formats on RVV. Ztt supports QAU4C32 and byte-expanded
QSI8C16 through `1nlenx1nlen_ztt`; their quantization metadata matches RVV,
while packed code buffers are private to the Ztt variant and runtime geometry.

| Source | Family path | Bias | Fused | D | A | B | Tile × Arch |
|--------|-------------|------|-------|---|---|---|-------------|
| Q2_K compact | [`gemm_i8_u2`](../../src/gemm/gemm_i8_u2) | 1xnbias | clamp | f32 | qsi8d256p | qau2c16p | `4x1vl_rvv` |
| Q2_K expanded | [`gemm_i8_u4`](../../src/gemm/gemm_i8_u4) | 1xnbias | clamp | f32 | qsi8d256p | qau4c16p | `4x1vl_rvv` |
| Q3_K compact | [`gemm_i8_i3`](../../src/gemm/gemm_i8_i3) | 1xnbias | clamp | f32 | qsi8d256p | qsi3c16p | `4x1vl_rvv` |
| Q3_K expanded | [`gemm_i8_i4`](../../src/gemm/gemm_i8_i4) | 1xnbias | clamp | f32 | qsi8d256p | qsi4c16p | `4x1vl_rvv` |
| Q4_K | [`gemm_i8_u4`](../../src/gemm/gemm_i8_u4) | 1xnbias | clamp | f32 | qsi8d256p | qau4c32p | `4x1vl_rvv`, `1nlenx1nlen_ztt` |
| Q5_K | [`gemm_i8_u8`](../../src/gemm/gemm_i8_u8) | 1xnbias | clamp | f32 | qsi8d256p | qau8c32p | `4x1vl_rvv` |
| Q5_K plane-packed | [`gemm_i8_u5`](../../src/gemm/gemm_i8_u5) | 1xnbias | clamp | f32 | qsi8d256p | qau5c32p | `4x1vl_rvv` |
| Q6_K | [`gemm_i8_i8`](../../src/gemm/gemm_i8_i8) | 1xnbias | clamp | f32 | qsi8d256p | qsi8c16p | `4x1vl_rvv`, `1nlenx1nlen_ztt` |
| Q6_K plane-packed | [`gemm_i8_i6`](../../src/gemm/gemm_i8_i6) | 1xnbias | clamp | f32 | qsi8d256p | qsi6c16p | `4x1vl_rvv` |
| Q8_0 | [`gemm_i8_i8`](../../src/gemm/gemm_i8_i8) | 1xnbias | clamp | f32 | qsi8d32p | qsi8c32p | `4x1vl_rvv` |
| Q1_0 compact | [`gemm_i8_b1`](../../src/gemm/gemm_i8_b1) | 1xnbias | clamp | f32 | qsi8d32p | qsb1c128p | `4x1vl_rvv` |
| Q2_0 compact | [`gemm_i8_u2`](../../src/gemm/gemm_i8_u2) | 1xnbias | clamp | f32 | qsi8d32p | qau2z1c64p | `4x1vl_rvv` |
| Q1_0/Q2_0 expanded and Q4_0 normalized | [`gemm_i8_i4`](../../src/gemm/gemm_i8_i4) | 1xnbias | clamp | f32 | qsi8d32p | qsi4c32p | `4x1vl_rvv` |
| Q5_0 compact | [`gemm_i8_i5`](../../src/gemm/gemm_i8_i5) | 1xnbias | clamp | f32 | qsi8d32p | qsi5c32p | `4x1vl_rvv` |

### INT4 W4A8 (`gemm_i8_i4`)

Family path: [`src/gemm/gemm_i8_i4`](../../src/gemm/gemm_i8_i4).

W4A8: INT8 activation × INT4 weight, with dequantized fp output or requantized
int8 output. Sub-schemes:

- **Per-channel** (`qai8dx × qsi4cx`): per-row INT8 activation × per-channel symmetric INT4 weight.
- **Per-channel requantized int8 output** (`qai8dx × qsi4cxt -> qai8`): same inputs as per-channel, but the fp result is requantized back to INT8 (`reqi32toi8`).
- **Block-quantized symmetric** (`qai8dx/qsi8d2 × qsi4c2`): INT8 activation × K-blocked symmetric INT4 weight.
- **Block-quantized asymmetric** (`qsi8d2 × qai4c2`): K-blocked symmetric INT8 activation × K-blocked asymmetric INT4 weight.

#### W4A8 per-channel (qai8dx × qsi4cx)

| Bias    | Fused | D    | A        | B        | Tile × Arch  |
|---------|-------|------|----------|----------|--------------|
| 1xnbias | clamp | f16 | qai8dx | qsi4cxt | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qai8dx | qsi4cxt | `4x1vl_rvv` |
| 1xnbias | clamp | f16 | qai8dxp | qsi4cxp | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qai8dxp | qsi4cxp | `4x1vl_rvv` |
| 1xnbias | clamp | f32  | qai8dx   | qsi4cxt  | `4x1vl_rvv`  |
| 1xnbias | clamp | f32  | qai8dxp  | qsi4cxp  | `4x1vl_rvv`  |
| 1xnbias | clamp | bf16 | qai8dx | qsi4cxt | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | bf16 | qai8dx | qsi4cxt | `4x1vl_rvv` |
| 1xnbias | clamp | bf16 | qai8dxp | qsi4cxp | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | bf16 | qai8dxp | qsi4cxp | `4x1vl_rvv` |

#### W4A8 per-channel requantized int8 output (reqi32toi8)

| Bias    | Fused              | D    | A        | B        | Tile × Arch  |
|---------|--------------------|------|----------|----------|--------------|
| 1xnbias | reqi32toi8 + clamp | qai8 | qai8dx   | qsi4cxt  | `4x1vl_rvv`  |
| 1xnbias | reqi32toi8 + clamp | qai8 | qai8dxp  | qsi4cxp  | `4x1vl_rvv`  |

#### W4A8 block-quantized symmetric (× qsi4c2)

| Bias    | Fused | D    | A        | B        | Tile × Arch  |
|---------|-------|------|----------|----------|--------------|
| 1xnbias | clamp | f16 | qai8dx | qsi4c2t | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qai8dx | qsi4c2t | `4x1vl_rvv` |
| 1xnbias | clamp | f16 | qai8dxp | qsi4c2p | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qai8dxp | qsi4c2p | `4x1vl_rvv` |
| 1xnbias | clamp | f16 | qsi8d2 | qsi4c2t | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qsi8d2 | qsi4c2t | `4x1vl_rvv` |
| 1xnbias | clamp | f16 | qsi8d2p | qsi4c2p | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qsi8d2p | qsi4c2p | `4x1vl_rvv` |
| 1xnbias | clamp | f32  | qai8dx   | qsi4c2t  | `4x1vl_rvv`  |
| 1xnbias | clamp | f32  | qai8dxp  | qsi4c2p  | `4x1vl_rvv`  |
| 1xnbias | clamp | f32  | qsi8d2   | qsi4c2t  | `4x1vl_rvv`  |
| 1xnbias | clamp | f32  | qsi8d2p  | qsi4c2p  | `4x1vl_rvv`  |
| 1xnbias | clamp | bf16 | qai8dx | qsi4c2t | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | bf16 | qai8dx | qsi4c2t | `4x1vl_rvv` |
| 1xnbias | clamp | bf16 | qai8dxp | qsi4c2p | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | bf16 | qai8dxp | qsi4c2p | `4x1vl_rvv` |
| 1xnbias | clamp | bf16 | qsi8d2 | qsi4c2t | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | bf16 | qsi8d2 | qsi4c2t | `4x1vl_rvv` |
| 1xnbias | clamp | bf16 | qsi8d2p | qsi4c2p | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | bf16 | qsi8d2p | qsi4c2p | `4x1vl_rvv` |

#### W4A8 block-quantized asymmetric (× qai4c2)

| Bias    | Fused | D    | A       | B        | Tile × Arch  |
|---------|-------|------|---------|----------|--------------|
| 1xnbias | clamp | f16 | qsi8d2 | qai4c2t | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qsi8d2 | qai4c2t | `4x1vl_rvv` |
| 1xnbias | clamp | f16 | qsi8d2p | qai4c2p | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | f16 | qsi8d2p | qai4c2p | `4x1vl_rvv` |
| 1xnbias | clamp | f32  | qsi8d2  | qai4c2t  | `4x1vl_rvv`  |
| 1xnbias | clamp | f32  | qsi8d2p | qai4c2p  | `4x1vl_rvv`  |
| 1xnbias | clamp | bf16 | qsi8d2 | qai4c2t | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | bf16 | qsi8d2 | qai4c2t | `4x1vl_rvv` |
| 1xnbias | clamp | bf16 | qsi8d2p | qai4c2p | `4x1vl_rvv` |
| 1xnbiasf32 | clamp | bf16 | qsi8d2p | qai4c2p | `4x1vl_rvv` |

### INT2 W2A8 (`gemm_i8_i2`)

Family path: [`src/gemm/gemm_i8_i2`](../../src/gemm/gemm_i8_i2).

W2A8: INT8 activation × INT2 weight with per-channel scales and an optional four-entry INT8 LUT. F32 output uses per-row activation quantization; requantized I8 output uses per-tensor activation quantization and raw I32 bias.

| Bias    | Fused | D   | A        | B        | Tile × Arch  |
|---------|-------|-----|----------|----------|--------------|
| 1xnbias | clamp | f32 | qai8dx   | qsi2cxt  | `4x1vl_rvv`  |
| 1xnbias | clamp | f32 | qai8dxp  | qsi2cxp  | `4x1vl_rvv`  |
| 1xnbias | reqi32toi8 + clamp | qai8 | qai8 | qsi2cxt | `4x1vl_rvv` |
| 1xnbias | reqi32toi8 + clamp | qai8 | qai8p | qsi2cxp | `4x1vl_rvv` |

## Convolution (im2col + GEMM) micro-kernels

Operator: `convcf` (channel-first / CHW) and `convcl` (channel-last / HWC). The convolution is realised as an im2col transformation followed by a GEMM. Both layout variants are exposed because the optimal packing layout depends on the surrounding tensor format.

### FP32 (`conv_f32_f32`)

Family path: [`src/conv/conv_f32_f32`](../../src/conv/conv_f32_f32).

| Op     | Bias    | Fused | D   | A    | B    | Tile × Arch  |
|--------|---------|-------|-----|------|------|--------------|
| convcf | mx1bias | clamp | f32 | f32  | f32  | `8x3vl_rvv`  |
| convcf | mx1bias | clamp | f32 | f32p | f32p | `8x3vl_rvv`  |
| convcl | 1xnbias | clamp | f32 | f32  | f32t | `8x3vl_rvv`  |
| convcl | 1xnbias | clamp | f32 | f32  | f32p | `8x3vl_rvv`  |
| convcl | 1xnbias | clamp | f32 | f32p | f32p | `8x3vl_rvv`  |

### FP16 (`conv_f16_f16`)

Family path: [`src/conv/conv_f16_f16`](../../src/conv/conv_f16_f16).

| Op     | Bias    | Fused | D   | A    | B    | Tile × Arch                        |
|--------|---------|-------|-----|------|------|------------------------------------|
| convcf | mx1bias | clamp | f16 | f16  | f16  | `8x3vl_rvv`, `8x1vl_f32acc_rvv`    |
| convcf | mx1bias | clamp | f16 | f16p | f16p | `8x3vl_rvv`, `8x1vl_f32acc_rvv`    |
| convcl | 1xnbias | clamp | f16 | f16  | f16t | `8x3vl_rvv`, `8x1vl_f32acc_rvv`    |
| convcl | 1xnbias | clamp | f16 | f16  | f16p | `8x3vl_rvv`                        |
| convcl | 1xnbias | clamp | f16 | f16p | f16p | `8x3vl_rvv`, `8x1vl_f32acc_rvv`    |

### BF16 (`conv_bf16_bf16`)

Family path: [`src/conv/conv_bf16_bf16`](../../src/conv/conv_bf16_bf16). Delegates to the `gemm_bf16_bf16`
RVV backend (`8x1vl`).

| Op     | Bias    | Fused | D    | A     | B     | Tile × Arch                                             |
|--------|---------|-------|------|-------|-------|---------------------------------------------------------|
| convcf | mx1bias | clamp | bf16 | bf16  | bf16  | `8x1vl_rvv` |
| convcf | mx1bias | clamp | bf16 | bf16p | bf16p | `8x1vl_rvv` |
| convcl | 1xnbias | clamp | bf16 | bf16  | bf16t | `8x1vl_rvv` |
| convcl | 1xnbias | clamp | bf16 | bf16p | bf16p | `8x1vl_rvv` |

### INT8 (`conv_i8_i8`)

Family path: [`src/conv/conv_i8_i8`](../../src/conv/conv_i8_i8).

| Op     | Bias    | Fused              | D    | A        | B        | Tile × Arch                        |
|--------|---------|--------------------|------|----------|----------|------------------------------------|
| convcf | mx1bias | reqi32toi8 + clamp | qai8 | qsi8dx   | qai8     | `4x1vl_rvv`    |
| convcf | mx1bias | reqi32toi8 + clamp | qai8 | qsi8dxp  | qai8p    | `4x1vl_rvv`    |
| convcl | 1xnbias | reqi32toi8 + clamp | qai8 | qai8     | qsi8cxt  | `4x1vl_rvv`    |
| convcl | 1xnbias | reqi32toi8 + clamp | qai8 | qai8p    | qsi8cxp  | `4x1vl_rvv`    |

---

## Implicit GEMM convolution micro-kernels

Operator: `igemmcf` (channel-first) and `igemmcl` (channel-last).

The implicit-GEMM kernels do not materialise an im2col buffer; they walk the input tensor directly via an indirection table. The `igemmcf` variants use a swapped GEMM register layout (channels along columns, spatial along rows), which is reflected in their tile names (`3vlx8`, `1vlx4`).

### FP32 (`igemm_f32_f32`)

Family path: [`src/conv/igemm_f32_f32`](../../src/conv/igemm_f32_f32).

| Op      | Bias    | Fused | D   | A    | B    | Tile × Arch  |
|---------|---------|-------|-----|------|------|--------------|
| igemmcf | mx1bias | clamp | f32 | f32  | f32  | `3vlx8_rvv`  |
| igemmcf | mx1bias | clamp | f32 | f32p | f32  | `3vlx8_rvv`  |
| igemmcl | 1xnbias | clamp | f32 | f32  | f32t | `8x3vl_rvv`  |
| igemmcl | 1xnbias | clamp | f32 | f32  | f32p | `8x3vl_rvv`  |

### FP16 (`igemm_f16_f16`)

Family path: [`src/conv/igemm_f16_f16`](../../src/conv/igemm_f16_f16).

| Op      | Bias    | Fused | D   | A    | B    | Tile × Arch  |
|---------|---------|-------|-----|------|------|--------------|
| igemmcf | mx1bias | clamp | f16 | f16  | f16  | `3vlx8_rvv`  |
| igemmcf | mx1bias | clamp | f16 | f16p | f16  | `3vlx8_rvv`  |
| igemmcl | 1xnbias | clamp | f16 | f16  | f16t | `8x3vl_rvv`  |
| igemmcl | 1xnbias | clamp | f16 | f16  | f16p | `8x3vl_rvv`  |

### INT8 (`igemm_i8_i8`)

Family path: [`src/conv/igemm_i8_i8`](../../src/conv/igemm_i8_i8).

| Op      | Bias    | Fused              | D    | A        | B        | Tile × Arch  |
|---------|---------|--------------------|------|----------|----------|--------------|
| igemmcf | mx1bias | reqi32toi8 + clamp | qai8 | qsi8dx   | qai8     | `1vlx4_rvv`  |
| igemmcf | mx1bias | reqi32toi8 + clamp | qai8 | qsi8dxp  | qai8     | `1vlx4_rvv`  |
| igemmcl | 1xnbias | reqi32toi8 + clamp | qai8 | qai8     | qsi8cxt  | `4x1vl_rvv`  |
| igemmcl | 1xnbias | reqi32toi8 + clamp | qai8 | qai8     | qsi8cxp  | `4x1vl_rvv`  |

---

## Depthwise convolution micro-kernels

Operator: `dwconvcf` / `dwconvcl` (generic, any kernel size and stride), plus the filter-size + stride specialised variants: `dwconv*3x3s1`, `dwconv*3x3s2`, `dwconv*5x5s1`.

The specialised variants fix the filter and stride at compile time. Performance depends on the target implementation and shape.

### FP32 (`dwconv_f32_f32`)

Family path: [`src/dwconv/dwconv_f32_f32`](../../src/dwconv/dwconv_f32_f32).

| Op             | Bias    | Fused | D    | A    | B    | Tile × Arch     |
|----------------|---------|-------|------|------|------|-----------------|
| dwconvcf       | mx1bias | clamp | f32  | f32  | f32  | `1x1vl_rvv`     |
| dwconvcf       | mx1bias | clamp | f32p | f32p | f32p | `1x1vl_rvv`     |
| dwconvcf3x3s1  | mx1bias | clamp | f32p | f32p | f32p | `4x1x1vl_rvv`   |
| dwconvcf3x3s2  | mx1bias | clamp | f32p | f32p | f32p | `2x1x1vl_rvv`   |
| dwconvcf5x5s1  | mx1bias | clamp | f32p | f32p | f32p | `4x1x1vl_rvv`   |
| dwconvcl       | 1xnbias | clamp | f32  | f32  | f32t | `1x1vl_rvv`     |
| dwconvcl       | 1xnbias | clamp | f32p | f32p | f32p | `1x1vl_rvv`     |
| dwconvcl3x3s1  | 1xnbias | clamp | f32  | f32  | f32p | `4x1x1vl_rvv`   |
| dwconvcl3x3s1  | 1xnbias | clamp | f32p | f32p | f32p | `4x1x1vl_rvv`   |
| dwconvcl3x3s2  | 1xnbias | clamp | f32p | f32p | f32p | `2x1x1vl_rvv`   |
| dwconvcl5x5s1  | 1xnbias | clamp | f32p | f32p | f32p | `4x1x1vl_rvv`   |

### FP16 (`dwconv_f16_f16`)

Family path: [`src/dwconv/dwconv_f16_f16`](../../src/dwconv/dwconv_f16_f16).

| Op             | Bias    | Fused | D    | A    | B    | Tile × Arch     |
|----------------|---------|-------|------|------|------|-----------------|
| dwconvcf       | mx1bias | clamp | f16  | f16  | f16  | `1x1vl_rvv`     |
| dwconvcf       | mx1bias | clamp | f16p | f16p | f16p | `1x1vl_rvv`     |
| dwconvcf3x3s1  | mx1bias | clamp | f16p | f16p | f16p | `4x1x1vl_rvv`   |
| dwconvcf3x3s2  | mx1bias | clamp | f16p | f16p | f16p | `2x1x1vl_rvv`   |
| dwconvcf5x5s1  | mx1bias | clamp | f16p | f16p | f16p | `4x1x1vl_rvv`   |
| dwconvcl       | 1xnbias | clamp | f16  | f16  | f16t | `1x1vl_rvv`     |
| dwconvcl       | 1xnbias | clamp | f16p | f16p | f16p | `1x1vl_rvv`     |
| dwconvcl3x3s1  | 1xnbias | clamp | f16  | f16  | f16p | `4x1x1vl_rvv`   |
| dwconvcl3x3s1  | 1xnbias | clamp | f16p | f16p | f16p | `4x1x1vl_rvv`   |
| dwconvcl3x3s2  | 1xnbias | clamp | f16p | f16p | f16p | `2x1x1vl_rvv`   |
| dwconvcl5x5s1  | 1xnbias | clamp | f16p | f16p | f16p | `4x1x1vl_rvv`   |

## Elementwise micro-kernels

Operators: `<op>vv` (vector-vector, `c[i] = a[i] ∘ b[i]`) and `<op>vs` (vector-scalar, `c[i] = a[i] ∘ b`, where `b` is a scalar), for `∘ ∈ {+ (add), - (sub), * (mul), / (div)}`. The inputs and output are contiguous 1-D vectors of length `n`. Elementwise kernels carry no bias and no clamp, so their `KERNEL_BASE` is just `<op>_<D>_<A>_<B>`.

### FP32 (`add_f32_f32`)

Family path: [`src/elementwise/add_f32_f32`](../../src/elementwise/add_f32_f32).

| Op    | D   | A   | B            | Tile × Arch |
|-------|-----|-----|--------------|-------------|
| addvv | f32 | f32 | f32          | `8vl_rvv` |
| addvs | f32 | f32 | f32 (scalar) | `8vl_rvv` |

### FP32 (`sub_f32_f32`)

Family path: [`src/elementwise/sub_f32_f32`](../../src/elementwise/sub_f32_f32).

| Op    | D   | A   | B            | Tile × Arch |
|-------|-----|-----|--------------|-------------|
| subvv | f32 | f32 | f32          | `8vl_rvv` |
| subvs | f32 | f32 | f32 (scalar) | `8vl_rvv` |

### FP32 (`mul_f32_f32`)

Family path: [`src/elementwise/mul_f32_f32`](../../src/elementwise/mul_f32_f32).

| Op    | D   | A   | B            | Tile × Arch |
|-------|-----|-----|--------------|-------------|
| mulvv | f32 | f32 | f32          | `8vl_rvv` |
| mulvs | f32 | f32 | f32 (scalar) | `8vl_rvv` |

### FP32 (`div_f32_f32`)

Family path: [`src/elementwise/div_f32_f32`](../../src/elementwise/div_f32_f32).

| Op    | D   | A   | B            | Tile × Arch |
|-------|-----|-----|--------------|-------------|
| divvv | f32 | f32 | f32          | `8vl_rvv`   |
| divvs | f32 | f32 | f32 (scalar) | `8vl_rvv` |

### FP16 (`add_f16_f16`)

Family path: [`src/elementwise/add_f16_f16`](../../src/elementwise/add_f16_f16).

| Op    | D   | A   | B            | Tile × Arch |
|-------|-----|-----|--------------|-------------|
| addvv | f16 | f16 | f16          | `8vl_rvv` |
| addvs | f16 | f16 | f16 (scalar) | `8vl_rvv` |

### FP16 (`sub_f16_f16`)

Family path: [`src/elementwise/sub_f16_f16`](../../src/elementwise/sub_f16_f16).

| Op    | D   | A   | B            | Tile × Arch |
|-------|-----|-----|--------------|-------------|
| subvv | f16 | f16 | f16          | `8vl_rvv` |
| subvs | f16 | f16 | f16 (scalar) | `8vl_rvv` |

### FP16 (`mul_f16_f16`)

Family path: [`src/elementwise/mul_f16_f16`](../../src/elementwise/mul_f16_f16).

| Op    | D   | A   | B            | Tile × Arch |
|-------|-----|-----|--------------|-------------|
| mulvv | f16 | f16 | f16          | `8vl_rvv` |
| mulvs | f16 | f16 | f16 (scalar) | `8vl_rvv` |

### FP16 (`div_f16_f16`)

Family path: [`src/elementwise/div_f16_f16`](../../src/elementwise/div_f16_f16).

| Op    | D   | A   | B            | Tile × Arch |
|-------|-----|-----|--------------|-------------|
| divvv | f16 | f16 | f16          | `8vl_rvv`   |
| divvs | f16 | f16 | f16 (scalar) | `8vl_rvv` |

## Reduction micro-kernels

Row and column sum/mean are available for FP32 and FP16 on RVV. Both use FP32
accumulation; FP16 results are narrowed on store. The optional init vector seeds
a sum or acts as an output bias for mean. Floating-point reduction order is not
fixed across vector widths. See the [API reference](api_reference.md#reduction-sum--mean)
for range, rounding and reproducibility limits.

### FP32 sum (`sum_f32_f32`)

Family path: [`src/reduction/sum_f32_f32`](../../src/reduction/sum_f32_f32).

| Op     | D   | A   | init           | Tile × Arch                    |
|--------|-----|-----|----------------|--------------------------------|
| sumrow | f32 | f32 | f32 (optional) | `4x4vl_rvv` |
| sumcol | f32 | f32 | f32 (optional) | `2x8vl_rvv`     |

### FP16 sum (`sum_f16_f16`)

Family path: [`src/reduction/sum_f16_f16`](../../src/reduction/sum_f16_f16).

| Op     | D   | A   | init           | Tile × Arch                    |
|--------|-----|-----|----------------|--------------------------------|
| sumrow | f16 | f16 | f16 (optional) | `4x2vl_rvv` |
| sumcol | f16 | f16 | f16 (optional) | `2x4vl_rvv`     |

### FP32 mean (`mean_f32_f32`)

Family path: [`src/reduction/mean_f32_f32`](../../src/reduction/mean_f32_f32).

| Op      | D   | A   | init           | Tile × Arch                    |
|---------|-----|-----|----------------|--------------------------------|
| meanrow | f32 | f32 | f32 (optional) | `4x4vl_rvv` |
| meancol | f32 | f32 | f32 (optional) | `2x8vl_rvv`     |

### FP16 mean (`mean_f16_f16`)

Family path: [`src/reduction/mean_f16_f16`](../../src/reduction/mean_f16_f16).

| Op      | D   | A   | init           | Tile × Arch                    |
|---------|-----|-----|----------------|--------------------------------|
| meanrow | f16 | f16 | f16 (optional) | `4x2vl_rvv` |
| meancol | f16 | f16 | f16 (optional) | `2x4vl_rvv`     |

## Activation micro-kernels

RVV provides FP32 and FP16 ReLU, GeLU, SiLU and their gated forms ReGLU,
GeGLU and SwiGLU. Gated APIs accept independently strided gate/up buffers,
including slices of a fused `[M, 2N]` allocation.

GeLU uses the standard tanh approximation. GeLU/GeGLU use the shared rational
tanh helper; SiLU/SwiGLU use the FMA-based exp helper and require RNE rounding.
FP16 arithmetic widens to FP32 before the final store. SiLU propagates NaNs,
preserves signed zeros and maps positive infinity to infinity and negative
infinity to NaN. SwiGLU applies ordinary floating multiplication to the gate.
ReLU maps NaN and negative zero to positive zero through the native maximum.
See the [API reference](api_reference.md#activation-relu--gelu--silu--reglu--geglu--swiglu)
and each public header for approximation and special-value contracts.

### FP32 ReLU (`relu_f32_f32`)

Family path: [`src/activation/relu_f32_f32`](../../src/activation/relu_f32_f32).

| Op   | D   | A   | Tile × Arch                      |
|------|-----|-----|----------------------------------|
| relu | f32 | f32 | `1x8vl_rvv`   |

### FP16 ReLU (`relu_f16_f16`)

Family path: [`src/activation/relu_f16_f16`](../../src/activation/relu_f16_f16).

| Op   | D   | A   | Tile × Arch                      |
|------|-----|-----|----------------------------------|
| relu | f16 | f16 | `1x8vl_rvv`   |

### FP32 ReGLU (`reglu_f32_f32_f32`)

Family path: [`src/activation/reglu_f32_f32_f32`](../../src/activation/reglu_f32_f32_f32).

| Op    | D   | A   | B   | Tile × Arch                      |
|-------|-----|-----|-----|----------------------------------|
| reglu | f32 | f32 | f32 | `1x8vl_rvv`   |

### FP16 ReGLU (`reglu_f16_f16_f16`)

Family path: [`src/activation/reglu_f16_f16_f16`](../../src/activation/reglu_f16_f16_f16).

| Op    | D   | A   | B   | Tile × Arch                      |
|-------|-----|-----|-----|----------------------------------|
| reglu | f16 | f16 | f16 | `1x8vl_rvv`   |

### FP32 GeLU (`gelu_f32_f32`)

Family path: [`src/activation/gelu_f32_f32`](../../src/activation/gelu_f32_f32).

| Op   | D   | A   | Tile × Arch                      |
|------|-----|-----|----------------------------------|
| gelu | f32 | f32 | `1x4vl_rvv`   |

### FP16 GeLU (`gelu_f16_f16`)

Family path: [`src/activation/gelu_f16_f16`](../../src/activation/gelu_f16_f16).

| Op   | D   | A   | Tile × Arch                      |
|------|-----|-----|----------------------------------|
| gelu | f16 | f16 | `1x2vl_rvv`   |

### FP32 GeGLU (`geglu_f32_f32_f32`)

Family path: [`src/activation/geglu_f32_f32_f32`](../../src/activation/geglu_f32_f32_f32).

| Op    | D   | A   | B   | Tile × Arch                      |
|-------|-----|-----|-----|----------------------------------|
| geglu | f32 | f32 | f32 | `1x4vl_rvv`   |

### FP16 GeGLU (`geglu_f16_f16_f16`)

Family path: [`src/activation/geglu_f16_f16_f16`](../../src/activation/geglu_f16_f16_f16).

| Op    | D   | A   | B   | Tile × Arch                      |
|-------|-----|-----|-----|----------------------------------|
| geglu | f16 | f16 | f16 | `1x2vl_rvv`   |

### FP32 SiLU (`silu_f32_f32`)

Family path: [`src/activation/silu_f32_f32`](../../src/activation/silu_f32_f32).

| Op   | D   | A   | Tile × Arch                      |
|------|-----|-----|----------------------------------|
| silu | f32 | f32 | `1x4vl_rvv`   |

### FP16 SiLU (`silu_f16_f16`)

Family path: [`src/activation/silu_f16_f16`](../../src/activation/silu_f16_f16).

| Op   | D   | A   | Tile × Arch                      |
|------|-----|-----|----------------------------------|
| silu | f16 | f16 | `1x2vl_rvv`   |

### FP32 SwiGLU (`swiglu_f32_f32_f32`)

Family path: [`src/activation/swiglu_f32_f32_f32`](../../src/activation/swiglu_f32_f32_f32).

| Op     | D   | A   | B   | Tile × Arch                      |
|--------|-----|-----|-----|----------------------------------|
| swiglu | f32 | f32 | f32 | `1x4vl_rvv`   |

### FP16 SwiGLU (`swiglu_f16_f16_f16`)

Family path: [`src/activation/swiglu_f16_f16_f16`](../../src/activation/swiglu_f16_f16_f16).

| Op     | D   | A   | B   | Tile × Arch                      |
|--------|-----|-----|-----|----------------------------------|
| swiglu | f16 | f16 | f16 | `1x2vl_rvv`   |

## Summary

Counts describe the files shipped in this public source tree. Each implementation
is one standalone kernel `.cpp`; helper headers are excluded.

| Operation group | Families | Kernel bases | RVV | Ztt | Total |
|---|---:|---:|---:|---:|---:|
| gemm | 19 | 125 | 141 | 17 | 158 |
| conv | 7 | 30 | 34 | 0 | 34 |
| dwconv | 2 | 22 | 22 | 0 | 22 |
| elementwise | 8 | 16 | 16 | 0 | 16 |
| reduction | 4 | 8 | 8 | 0 | 8 |
| activation | 12 | 12 | 12 | 0 | 12 |
| **Total** | **52** | **213** | **233** | **17** | **250** |

## Maintenance note

This catalog is maintained by hand. It is verified against [`src/`](../../src/) — every `tqt_*.cpp` source file there should have exactly one row in the tables above (and conversely, every row should correspond to a real source file). When adding a new variant, please update both this page and the [naming convention](naming_convention.md) if the new variant introduces a previously unused token.
