<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Coding Conventions

This document defines the code style and conventions for all contributions to
TORQ-Tile. Consistency across the codebase is essential for maintainability and
for enabling multiple developers to work on shared inner kernel headers.

## Language Standards

- **C**: GNU C11 (`-std=gnu11`, for standalone C files or headers consumed by C code)
- **C++**: C++17 (for all `.cpp` implementation files)

GCC/Clang extensions, including extended inline assembly, are permitted.

All public API headers use `extern "C"` linkage to ensure C compatibility.

## Formatting

TORQ-Tile uses **clang-format** with a Google-based configuration. Run the formatter
before committing:

```bash
bash scripts/format-all.sh
```

### Key Formatting Rules

| Rule | Setting |
|------|---------|
| Indent width | 4 spaces (no tabs) |
| Line limit | 100 characters (except indivisible tokens; see below) |
| Brace style | Next line for functions, classes, structs |
| Brace style | Same line for control statements (if, for, while) |
| Brace style | Short inline member functions may stay on one line (`AllowShortFunctionsOnASingleLine: Inline`) |
| Pointer alignment | Right (`int *p`, not `int* p` or `int * p`) |
| Include order | System headers, then project headers (sorted) |
| Namespace indent | None (contents at column 0) |

The 100-character limit is a formatting target. A line may exceed it only when
clang-format cannot split an indivisible token, such as an include path or a complete
identifier. Surrounding syntax that can be wrapped must still follow the configured limit.

> **Short inline member functions.** clang-format is configured with
> `AllowShortFunctionsOnASingleLine: Inline`, so a trivial constructor or accessor
> defined inside a class/struct is kept on a single line (e.g. the empty-body
> `ElementwiseTestParams(size_t n_, std::string name_ = "") : n(n_), name(std::move(name_)) {}`
> in `test/common/elementwise_test_runner.h`). This is intentional and does not violate the
> "brace on next line for functions" rule, which applies to free/out-of-line
> function definitions.

### Examples

```cpp
// Function definition: brace on next line
size_t tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv(void)
{
    return m_step;
}

// Control flow: brace on same line
for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
    const size_t actual_m = TQT_MIN(m - m_idx, m_step);
    if (actual_m == 0) {
        break;
    }
}

// Struct definition: brace on next line
struct tqt_gemm_1xnbias_clamp_f16_f16_f16_ukernel
{
    tqt_gemm_1xnbias_clamp_f16_f16_f16_get_m_step_func_t get_m_step;
    tqt_gemm_1xnbias_clamp_f16_f16_f16_run_gemm_func_t run_gemm;
};

// Pointer alignment: right-aligned
const float16_t *a_ptr = (const float16_t *)A + m_idx * lda;
float16_t *d_ptr = (float16_t *)D + m_idx * ldd + n_idx;
```

## License Header

Every source file (`.h`, `.cpp`, `.c`) must begin with the SPDX license header:

```cpp
//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//
```

For markdown and other non-source files, use the HTML comment form:

```html
<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->
```

The license header must be the very first content in the file (after any shebang line
for scripts).

## Comments

- **Language**: English only. No Chinese characters in source files.
- **Style**: Minimal comments. Code should be self-documenting through clear naming.
- **When to comment**:
  - Register allocation maps in inner kernel headers
  - Non-obvious mathematical transformations (e.g., requantization formulas)
  - Architecture-specific behavior (e.g., VLENB assumptions)
  - Doxygen-style `///` for public API functions in headers

### Comment Examples

```cpp
/// Register allocation for 8x3vl GEMM kernel:
///   ft0-ft7: A[0-7]
///   v0-v2:   B matrix columns (3 vector registers)
///   v8-v31:  Accumulator matrix D[0-7][0-2] (24 vector registers)

// Steady-state K-loop: fused load + fmacc for next iteration
for (size_t k_idx = 1; k_idx < K; k_idx++) {
    tqt_fused_load_vfmacc_axb_kernel_gemm_f16_f16_8x3vl_rvv<MR, NR>(
        a_ptr, a_stride_row, b_ptr, b_stride_col);
    a_ptr += 1;
    b_ptr += ldb;
}
```

Avoid comments that merely restate what the code does:

```cpp
// Bad: restates the code
size_t mr_idx = M - 1;  // Subtract 1 from M to get mr index

// Good: explains why
size_t mr_idx = M - 1;  // Dispatch table is 0-indexed; M is 1-based actual row count
```

## Naming Conventions

### Public Functions

All public functions use the prefix `tqt_` followed by an action and the full kernel
name:

```
tqt_<action>_<KERNEL_FULL>
```

Common getter actions include `get_m_step`, `get_n_step`, `get_oh_step`,
`get_*_offset`, and `get_*_size`. Compute actions are `run`, `run_conv`,
`run_dwconv`, `run_igemm`, and `run_im2col`. Packing actions include `run_a_pack`,
`run_b_pack`, `run_bt_pack`, `run_qparams_pack`, `run_scale_a_pack`,
`run_scale_bt_pack`, `run_a_im2colpack`, `run_b_im2colpack`, `run_pack_input`,
`run_pack_weight`, and `run_unpack_output`. The exact action set is defined by the
kernel-base `_interface.h`.

Examples:
```cpp
tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv
tqt_run_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv
tqt_run_b_pack_gemm_1xnbias_clamp_f16_f16p_f16p_8x3vl_rvv
```

### Inner Kernel Functions (Templates)

Inner kernel template functions in the shared header use the pattern:

```
tqt_<action>_kernel_<op>_<A>_<B>_<tile>_<arch>
```

Examples:
```cpp
tqt_init_zero_kernel_gemm_f16_f16_8x3vl_rvv<MR, NR>(vl)
tqt_load_a_kernel_gemm_f16_f16_8x3vl_rvv<MR>(a_ptr, stride)
tqt_fused_load_vfmacc_axb_kernel_gemm_f16_f16_8x3vl_rvv<MR, NR>(...)
tqt_clamp_kernel_gemm_f16_f16_8x3vl_rvv<MR, NR>(vl, min, max)
tqt_store_kernel_gemm_f16_f16_8x3vl_rvv<MR, NR>(vl, D, stride_row, stride_col)
```

### Static Constants

Ordinary descriptive module-level static constants in `.cpp` files use
`lower_snake_case`. Established hardware/ISA names and mathematical or algorithmic symbols
may retain `UPPER_SNAKE_CASE` when that spelling preserves their domain meaning, such as
`VLENB`, `LMUL`, `SNUM`, and `PNUM`. New descriptive constants must not use
`kPascalCase`.

```cpp
static constexpr size_t m_step = 8;
static constexpr size_t block_size = 128;
```

CSR-derived geometry is runtime data, not a module-level constant initializer.
Inside C++ getters or run functions, use the common `tqt_get_vlenb()`,
`tqt_get_amenlen()` and `tqt_get_ameudsz()` helpers under their
matching ISA guards. For example, an RVV FP16 function can compute:

```cpp
const size_t VLENB = tqt_get_vlenb();
const size_t vlmax = VLENB / sizeof(float16_t);
const size_t n_step = 3 * vlmax;
```

Do not duplicate these caches in individual kernel files or read CSRs during
module initialization. See [No Global Mutable State](#no-global-mutable-state).

### Template Parameters

Template parameters for the kernel dispatcher use uppercase:

```cpp
template <size_t MR, size_t NR>
static void gemm_kernel(size_t K, float16_t *D, ...);
```

### Type Aliases

Function pointer typedefs in interface headers follow the pattern:

```
tqt_<KERNEL_BASE>_<action>_func_t
```

### Test / Benchmark Helper Types

Internal (non-public-API) helper classes and structs in `test/` and `benchmark/` are
otherwise PascalCase, but may embed underscore-separated segments that mirror the kernel
naming grammar. These segments may encode the operation, data types, layout, quantization
scheme, tile, architecture, kernel size, or stride. This keeps the field boundaries readable
and traceable back to the kernel name:

```cpp
class Test_dgemm_f64_f64_f64_8x3vl_rvv { ... };
class Test_dwconvcf3x3s1_f32_f32_f32p { ... };
struct W4A8BQ_Qai8dx_Qsi4c2_BenchFuncs { ... };
struct W4A8BQ_Qsi8d2_Qai4c2_PackedFuncs { ... };
```

Collapsing these to pure PascalCase (`W4A8BQQai8dxQsi4c2BenchFuncs`) would erase the
field boundaries, so the underscore-segmented form is a sanctioned exception limited to
test/benchmark scaffolding — it is **not** used for public API or production code.

### General Rules

- No Hungarian notation (no `p_`, `m_`, `g_` prefixes)
- No abbreviations beyond established ones (`vl`, `vlmax`, `lda`, `ldb`, `ldc`, `ldd`)
- Macro names use ALL_CAPS. Internal implementation macros use the `TQT_` prefix (e.g.,
  `TQT_ASSUME`, `TQT_MIN`, `TQT_MAX`, `TQT_UNUSED`); externally overrideable build,
  configuration, or compiler-capability macros use the `TORQ_TILE_` prefix.
- Test/benchmark-only code-generation macros may omit the project prefix when their names
  describe the generated fixture or function set. Macros local to a `.cpp` file must be
  `#undef`-ed after use; macros declared in a test helper header must be intentional
  test-scaffolding interfaces.

## Architecture Guards

Every architecture-specific kernel `.cpp` file under `src/` must include an architecture
guard at the top, immediately after the license header. Kernel source files use `#error`
to catch misconfigurations at build time:

```cpp
#if !defined(__riscv) || !defined(__riscv_v)
#error This file must be compiled for riscv, riscv_vector.
#endif
```

Standalone architecture-specific examples may instead use the silent
empty-translation-unit pattern (`#if defined(...) ... #endif`) if they need to compile on
x86 for IDE indexing or build system validation without producing a RISC-V binary.

Test and benchmark source files may omit a per-file architecture guard when their source
registration is gated by the matching ISA capability in CMake. A source file that combines
multiple ISA families may use internal conditional sections to gate each family separately.

## Design Constraints

### No External Dependencies

TORQ-Tile micro-kernels have no third-party library dependency. Core headers may
use freestanding standard headers (`<stddef.h>`, `<stdint.h>`, `<float.h>`) and
compiler-provided RISC-V intrinsic headers. Guard hosted-only headers and calls
with `#if !TORQ_TILE_FREESTANDING`; a `.cpp` suffix does not make them available in
a freestanding build.

Use the memory and diagnostic helpers in `src/common/tqt_runtime.h` and
`src/common/tqt_common.h`. Freestanding final links still need platform-provided
`memcpy`, `memmove`, `memset`, `memcmp` and any compiler-generated helpers. Hosted
builds may also use the C/C++ runtime, including lazy-static initialization guards.
Do not inject SDK headers through `TORQ_TILE_PLATFORM_HEADER`; keep platform
integration in the caller. See the [runtime contract](../guide/integration.md).

### No Dynamic Allocation

Kernel implementations must not call `malloc`, `new`, or any dynamic allocator. All
buffers are caller-provided via pointer arguments. The library operates in
fixed-memory environments (embedded, real-time).

### No Global Mutable State

Call-specific state (pointers, dimensions, strides) is passed as function arguments.
Module-level constants must have compile-time initializers. The common C++ CSR
getters cache geometry per translation unit in hosted builds, with thread-safe
function-local static initialization on first use. Freestanding getters read the
CSR on demand and require neither a cache nor C++ guard services. Neither mode may
read CSRs at load time. Hosted caching assumes compatible hardware geometry across
all harts that use a translation unit; callers must initialize the ISA state before
calling a getter or kernel.

### No Exceptions

Do not use C++ exceptions. Error handling (if needed) is via return codes. In
practice, micro-kernels assume valid inputs and rely on `TQT_ASSUME` for optimization
hints.

## Include Paths

- Within `src/`: use paths relative to the project root
  (e.g., `#include "src/common/tqt_common.h"`)
- In the kernel's own directory: use bare filename
  (e.g., `#include "tqt_gemm_1xnbias_clamp_f16_f16_f16_8x3vl_rvv.h"`)
- Never use absolute paths or `../` relative paths

## Inline Assembly

Inner kernel headers use GCC inline assembly for performance-critical vector
operations. Follow these rules:

- Use `asm volatile(...)` for all instructions with side effects
- Declare all clobbered registers in the clobber list
- Use named constraints where possible for readability
- Define assembly macros with `#define` using ALL_CAPS names (e.g., `TQT_VZERO_FP16`)
- Undefine (`#undef`) assembly macros at the end of the header to prevent leakage

## Commit and PR Conventions

- One kernel variant per commit (or one logical unit of change)
- Commit message format: `feat(<scope>): <description>` for new kernels,
  `fix(<scope>): <description>` for corrections
- Run `scripts/format-all.sh` before committing
- Ensure all tests pass under QEMU before submitting a PR
