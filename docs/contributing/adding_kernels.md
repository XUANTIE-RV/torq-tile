<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Adding a New Micro-Kernel Variant

This guide walks through the complete process of adding a new micro-kernel variant
to TORQ-Tile. Before starting, familiarize yourself with the
[naming convention](../reference/naming_convention.md).

## Prerequisites

- Identify the target descriptor before parsing its name. Matrix/conv use
  `<op>_[<biastype>_][<fused_ops>_]<D>[suffix]_<A>[suffix]_<B>[suffix]`; elementwise uses
  `<op>{vv|vs}_<D>_<A>_<B>`; reduction uses `<op>{row|col}_<D>_<A>`; unary activation uses
  `<op>_<D>_<A>`; gated activation uses `<op>_<D>_<A>_<B>`. See
  [naming convention](../reference/naming_convention.md) for the complete grammar.
- Know which ISA extension you are targeting (RVV or Ztt)
- Have the cross-compilation toolchain set up (see root README)

## Step-by-Step Process

### 1. Identify the Family

Start with the existing directories under the target op group. Family shape is
descriptor-specific:

- Matrix, conv, and dwconv normally group by the base A/B pair. Public D, quantization tags, and
  layout suffixes remain in KERNEL_BASE.
- Elementwise and reduction use their existing operation/type family (for example,
  `add_f32_f32` or `sum_f32_f32`).
- Activation uses the complete KERNEL_BASE as both family label and kernel-base directory. Unary
  and gated activations therefore have different D/A versus D/A/B contracts.

```
src/gemm/gemm_f16_f16/      # FP16 x FP16
src/gemm/gemm_i8_i8/        # INT8 x INT8
src/gemm/gemm_f16_i4/       # FP16 x INT4 (W4A16)
src/gemm/gemm_i8_i4/        # INT8 x INT4 (W4A8)
src/conv/conv_f32_f32/       # FP32 conv
src/dwconv/dwconv_f16_f16/   # FP16 depthwise conv
src/elementwise/add_f32_f32/ # FP32 addvv/addvs
src/reduction/sum_f32_f32/   # FP32 sumrow/sumcol
src/activation/relu_f32_f32/relu_f32_f32/             # Unary activation
src/activation/geglu_f32_f32_f32/geglu_f32_f32_f32/   # Gated activation
```

**Decision rule:** Locate an existing source with the same descriptor and operation contract,
then follow its real family path. A shared `tqt_kernel_*` header is evidence for matrix families,
not a universal requirement: reduction and activation do not use one. Only create a new family
when the descriptor-specific contract requires it.

### 2. Find a Reference Variant

Prefer an existing variant with the same descriptor, operation, dtype, and architecture. For
example, an FP16 GEMM bias variant should start from the nearest `gemm_f16_f16` base. When adding
the first implementation of an operation, use the closest implementation with the same interface
shape and execution contract (for activation, keep unary and gated contracts separate).

Copy only the files the reference contract actually uses. Every public variant has `.h`, `.cpp`,
and a KERNEL_BASE `_interface.h`; a family-level inner header is conditional and must be discovered
from the reference source's includes.

### 3. Create the Kernel-Base Directory

```
src/<op_category>/<family>/
└── <KERNEL_BASE>/
    ├── tqt_<KERNEL_BASE>_<tile>_<arch>.h
    ├── tqt_<KERNEL_BASE>_<tile>_<arch>.cpp
    └── tqt_<KERNEL_BASE>_interface.h
```

Example for a new mx1bias FP16 kernel with 8x3vl tile:

```
src/gemm/gemm_f16_f16/
└── gemm_mx1bias_clamp_f16_f16_f16/
    ├── tqt_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv.h
    ├── tqt_gemm_mx1bias_clamp_f16_f16_f16_8x3vl_rvv.cpp
    └── tqt_gemm_mx1bias_clamp_f16_f16_f16_interface.h
```

### 4. Write the Header (.h)

The header declares the public C API. Read the reference `_interface.h` first and declare exactly
its getters and actions. The following is a common GEMM example, not a template for other
descriptors:

```cpp
//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include "src/common/tqt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Step/offset/size getters
size_t tqt_get_m_step_<KERNEL_FULL>(void);
size_t tqt_get_n_step_<KERNEL_FULL>(void);
size_t tqt_get_a_offset_<KERNEL_FULL>(size_t m_idx, size_t k_idx, size_t lda);
size_t tqt_get_b_offset_<KERNEL_FULL>(size_t n_idx, size_t k_idx, size_t ldb);
size_t tqt_get_c_offset_<KERNEL_FULL>(size_t m_idx, size_t n_idx, size_t ldc);
size_t tqt_get_bias_offset_<KERNEL_FULL>(size_t n_idx);
size_t tqt_get_d_offset_<KERNEL_FULL>(size_t m_idx, size_t n_idx, size_t ldd);
size_t tqt_get_d_size_<KERNEL_FULL>(size_t m, size_t n);

// Core run function
void tqt_run_<KERNEL_FULL>(size_t m, size_t n, size_t k,
                           const void *A, size_t lda, size_t k_idx_a,
                           const void *B, size_t ldb, size_t k_idx_b,
                           const void *C, size_t ldc,
                           void *D, size_t ldd,
                           const void *bias,
                           <dtype> clamp_min, <dtype> clamp_max);

#ifdef __cplusplus
}
#endif
```

Where `<KERNEL_FULL>` = `<KERNEL_BASE>_<tile>_<arch>` (e.g.,
`gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv`).

Quantized kernels with a `params` struct replace the bias/clamp scalar arguments with
a `const void *params` pointer.

### 5. Write the Source (.cpp)

The source must follow the reference variant's descriptor-specific structure. The list below is a
common tiled matrix implementation; elementwise, reduction, and activation have different loops,
steps, helpers, and action names:

1. **Architecture guard** (required):
   ```cpp
   #if !defined(__riscv) || !defined(__riscv_v)
   #error This file must be compiled for riscv, riscv_vector.
   #endif
   ```
   For FP16 kernels, also check `__riscv_zfh` and `__riscv_zvfh`.

2. **Includes**: the kernel's own header plus only the helpers included by the reference. Do not
   invent a shared inner kernel header for a descriptor that has none.

3. **Constants and runtime geometry**: keep only compile-time values (such as a fixed
   `m_step`) at file scope. Inside getters/run functions, derive `vlmax` and variable
   steps from the common C++ `tqt_get_vlenb()`, `tqt_get_amenlen()` or
   `tqt_get_ameudsz()` helper for the backend. Reuse the reference variant's structure;
   do not add per-kernel caches or CSR reads in global initializers. The common
   helpers cache on first use in hosted builds and read on demand in freestanding.

4. **Getter implementations**: trivial wrappers returning offsets computed from strides.

5. **Template kernel dispatcher**: a `template <size_t MR, size_t NR>` function that
   orchestrates init -> load A -> load B -> K-loop (fused load+fmacc) -> epilogue ->
   bias -> conditional clamp -> store. New floating GEMM variants must compute the
   shared `tqt_clamp_is_enabled` policy once from the original public bounds,
   before any narrowing or live vector/matrix state, and pass it through private
   dispatch. Only exact `[-INFINITY, INFINITY]` skips min/max. Keep raw helpers,
   public ABI, and the existing bias/conversion/store order intact. Register a
   per-variant special-value test with an independent NaN/Inf oracle.

6. **Dispatch table**: `static constexpr` 2D array of function pointers indexed by
   `[mr_idx][nr_idx]`.

7. **Run function**: outer M-tile loop, inner N-tile loop, compute actual_m/actual_n,
   dispatch to the correct kernel via the table.

### 6. Write or Reuse the Interface Header (_interface.h)

The interface header defines function pointer typedefs and a `ukernel` struct shared
across all tile/arch variants of the same kernel-base. Derive it from an exact same-contract
reference. This is a GEMM-shaped example only:

```cpp
struct tqt_<KERNEL_BASE>_ukernel
{
    tqt_<KERNEL_BASE>_get_m_step_func_t get_m_step;
    tqt_<KERNEL_BASE>_get_n_step_func_t get_n_step;
    tqt_<KERNEL_BASE>_get_a_offset_func_t get_a_offset;
    tqt_<KERNEL_BASE>_get_b_offset_func_t get_b_offset;
    tqt_<KERNEL_BASE>_get_c_offset_func_t get_c_offset;
    tqt_<KERNEL_BASE>_get_bias_offset_func_t get_bias_offset;
    tqt_<KERNEL_BASE>_get_d_offset_func_t get_d_offset;
    tqt_<KERNEL_BASE>_get_d_size_func_t get_d_size;
    tqt_<KERNEL_BASE>_run_gemm_func_t run_gemm;
};
```

**If an `_interface.h` already exists for your kernel-base** (e.g., you are adding a new
tile size variant), reuse it as-is. Never modify an existing interface header unless
the function signatures are fundamentally incompatible.

### 7. Register in CMakeLists.txt

Add the `.cpp` file to the appropriate CMake source variable in the root `CMakeLists.txt`:

| Data Types | Arch | CMake Variable |
|-----------|------|----------------|
| f32, i8_i8, i8_i4, i8_u4, i8_u5, i8_u8, i8_i6, u8_u8, i8_i2, qsi8d256/qsi8d32 block GEMM, elementwise/reduction/activation f32 | RVV | `TORQ_TILE_RVV_SOURCES` |
| f16, f32xf16 W16A32, f16_i4, i8_i8, i8_i4, elementwise/reduction/activation f16 | RVV | `TORQ_TILE_RVV_F16_SOURCES` |
| bf16 gemm/conv, f32xbf16 W16A32, i8_i4 | RVV | `TORQ_TILE_RVV_BF16_SOURCES` |
| bf16-input GEMM with f16 output (also needs Zvfhmin or Zvfh) | RVV | `TORQ_TILE_RVV_BF16_TO_F16_SOURCES` |
| f64 | RVV | `TORQ_TILE_RVV_F64_SOURCES` |
| relu / gelu / geglu / silu / reglu / swiglu activation (RVV, no extra extension) | RVV | `TORQ_TILE_RVV_SOURCES` / `TORQ_TILE_RVV_F16_SOURCES` |
| F32 GEMM | Ztt | `TORQ_TILE_ZTT_GEMM_F32_SOURCES` |
| F16 GEMM (native and f32acc) | Ztt | `TORQ_TILE_ZTT_GEMM_F16_SOURCES` |
| BF16 GEMM (native and f32acc) | Ztt | `TORQ_TILE_ZTT_GEMM_BF16_SOURCES` |
| Q4_K integer dot | Ztt | `TORQ_TILE_ZTT_GEMM_BLOCK_INT_U4_SOURCES` |
| Expanded Q6_K integer dot | Ztt | `TORQ_TILE_ZTT_GEMM_BLOCK_INT_I8_SOURCES` |

Add the source file path following the existing family, kernel-base, and tile
grouping within the variable block.

### 8. Add Test Case

Add a test to the matching op directory (for example,
`test/gemm/test_gemm_f16_f16_rvv.cpp` or `test/activation/test_geglu_f32_f32_f32_rvv.cpp`), or
create a new file if none exists. Reuse that descriptor's real runner and fixture rather than
adapting `NonPackedGemmFuncs` universally.

Required coverage is contract-dependent:

- All descriptors: full tiles, tails, minimal/zero dimensions accepted by the public contract,
  independent strides, output padding, numerical reference, and documented alias cases.
- Matrix/conv/dwconv: applicable M/N/K, bias, fused operations, accumulator, packing, and metadata.
- Elementwise/reduction/activation: their actual axis/operand/action semantics; activation must
  distinguish unary from gated A/B contracts and cover allowed in-place aliases.
- Split-K, `k == 0` with null A/B, and non-zero K-origin packing are required only when the target
  interface and implementation support those capabilities. Record unsupported cases explicitly
  instead of treating a helper-name grep as a repository-wide compliance percentage.

For mixed-input families, keep source and weight template types independent in
the reference runner and model the selected execution precision. Allocate packed
buffers from the kernel size getters and use the documented packed layout.

See [testing.md](./testing.md) for the full testing guide.

### 9. Add Benchmark Case

Add a benchmark under `benchmark/<op-group>/bench_<descriptor-specific-name>_<arch>.cpp` (or
create a new file). Reuse the matching op runner and choose representative shapes for that
contract; GEMM commonly uses square inference dimensions, while elementwise, reduction, and
activation do not inherit an M/N/K workload automatically.

### 10. Assess Example Necessity

Check the [scenario index](../../examples/README.md) before adding an example.
A new datatype, tile or ISA normally reuses an existing calling workflow; add a
program only when a distinct workflow needs one. Follow the
[example maintenance rules](examples.md) for source organization, registration,
documentation and checks.

## File Naming Rules

| File Type | Pattern |
|-----------|---------|
| Source | `tqt_<KERNEL_BASE>_<tile>_<arch>.cpp` |
| Header | `tqt_<KERNEL_BASE>_<tile>_<arch>.h` |
| Interface | `tqt_<KERNEL_BASE>_interface.h` |
| Inner kernel | Descriptor-specific and optional; use the exact include from the reference source |
| Test | `test_<descriptor-specific-name>_<arch>.cpp` |
| Benchmark | `bench_<descriptor-specific-name>_<arch>.cpp` |
| Example dir | Calling scenario under `examples/api/`; reuse an existing topic where possible |

## Architecture Guard Templates

### RVV (basic)
```cpp
#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif
```

### RVV with FP16
```cpp
#if !defined(__riscv) || !defined(__riscv_v) || !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error This file must be compiled for riscv, riscv_vector, zfh, zvfh.
#endif
```

## Key Rules

1. **Never modify existing shared inner kernel headers** (`tqt_kernel_*_<tile>_<arch>.h`) --
   when present, they are shared across kernel-bases in a family.
2. **Never modify existing `_interface.h`** files if the function signatures are compatible
   with your new variant.
3. **Always use a reference variant** with the same descriptor and contract; prefer the same
   family when it already exists.
4. **Test edge cases**: cover partial tiles where the public contract permits them;
   for aligned-only interfaces, verify their documented preconditions instead.
5. **Maintain register allocation discipline**: document which registers are used for
   accumulators, temporaries, and source operands in the inner kernel header.

## Checklist

Before submitting:

- [ ] Header (.h) declares all getters + run function in `extern "C"` block
- [ ] Source (.cpp) has an architecture guard that rejects unsupported target flags
- [ ] Source includes the exact descriptor-specific helpers required by the reference
- [ ] Interface header exists and is compatible
- [ ] Registered in `CMakeLists.txt` under the correct variable
- [ ] Test added with aligned + edge-case dimensions
- [ ] Benchmark added with typical inference sizes
- [ ] If an example changed, its independent CMake project builds and the selected scenario passes
- [ ] SPDX license header on all new files
- [ ] `scripts/format-all.sh` passes without changes
