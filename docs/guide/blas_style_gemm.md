<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# BLAS-Style GEMM Guide (sgemm / dgemm)

This guide is for users familiar with standard BLAS who want to use TORQ-Tile's
optimized `sgemm` and `dgemm` micro-kernels on RISC-V.

## What sgemm/dgemm Provide

These kernels implement the standard BLAS general matrix multiply:

```
D = alpha * op(A) * op(B) + beta * C
```

where `op(X)` is either `X` (no transpose) or `X^T` (transpose), and `alpha`
and `beta` are scalar multipliers.

Key semantics:
- When `beta == 0`, `C` may be `nullptr` (the kernel skips the C read entirely).
- When `beta == 1` and `alpha == 1`, this reduces to `D = A*B + C`.
- D and C may alias (in-place accumulation).

## Differences from Standard GEMM Kernels

| Feature | Standard gemm (`gemm_1xnbias_clamp_*`) | BLAS sgemm/dgemm |
|---------|---------------------------------------|------------------|
| Bias    | 1xN or Mx1 bias vector               | None (use C for bias) |
| Clamp   | `clamp_min` / `clamp_max`            | None |
| Scalars | None                                  | `alpha`, `beta` as last args |
| Output  | `D = clamp(A*B + bias + C, min, max)` | `D = alpha*A*B + beta*C` |

BLAS kernels are simpler: no bias, no activation clamp, no quantization.
The `alpha`/`beta` scalars are passed as the last two arguments of the run
function.

## Available Variants

### sgemm (single-precision, f32)

All four transpose combinations are available on RVV:

| Variant | A layout | B layout | Arch | Tile |
|---------|----------|----------|------|------|
| `sgemm_f32_f32_f32` | Row-major | Row-major | RVV | 8x3vl |
| `sgemm_f32_f32_f32t` | Row-major | Transposed | RVV | 8x3vl |
| `sgemm_f32_f32t_f32` | Transposed | Row-major | RVV | 8x3vl |
| `sgemm_f32_f32t_f32t` | Transposed | Transposed | RVV | 8x3vl |

### dgemm (double-precision, f64, RVV only)

| Variant | A layout | B layout | Arch | Tile |
|---------|----------|----------|------|------|
| `dgemm_f64_f64_f64` | Row-major | Row-major | RVV | 8x3vl |
| `dgemm_f64_f64_f64t` | Row-major | Transposed | RVV | 8x3vl |
| `dgemm_f64_f64t_f64` | Transposed | Row-major | RVV | 8x3vl |
| `dgemm_f64_f64t_f64t` | Transposed | Transposed | RVV | 8x3vl |

## Transpose Naming Convention

The data-type slot for A and B carries a `t` suffix when that operand is
transposed:

- `f32` = row-major (non-transposed): stored as `[rows, cols]`
- `f32t` = transposed: stored as `[cols, rows]` in row-major order

The A type appears first (left), B type second (right):

```
sgemm_f32_<A-type>_<B-type>_<tile>_<arch>
              |         |
         A's slot   B's slot
```

Examples:
- `sgemm_f32_f32_f32t` -- A is row-major `[m, k]`, B is transposed `[n, k]`
- `sgemm_f32_f32t_f32` -- A is transposed `[k, m]`, B is row-major `[k, n]`

## API Shape

### Full function signature (NT variant, RVV)

```c
void tqt_run_sgemm_f32_f32_f32t_8x3vl_rvv(
    size_t m,        // Number of rows in output D
    size_t n,        // Number of columns in output D
    size_t k,        // Shared/contraction dimension
    const void *A,   // Input A: [m, k] row-major (float)
    size_t lda,      // Leading dimension of A (>= k)
    size_t k_idx_a,  // K offset in A buffer (0 for full matrix)
    const void *B,   // Input B: [n, k] transposed (float)
    size_t ldb,      // Leading dimension of B (>= k)
    size_t k_idx_b,  // K offset in B buffer (0 for full matrix)
    const void *C,   // Input C: [m, n] row-major (float), nullptr if beta==0
    size_t ldc,      // Leading dimension of C (>= n)
    void *D,         // Output D: [m, n] row-major (float)
    size_t ldd,      // Leading dimension of D (>= n)
    float alpha,     // Scalar multiplier for A*B product
    float beta       // Scalar multiplier for C
);
```

### Full function signature (NT variant, dgemm)

```c
void tqt_run_dgemm_f64_f64_f64t_8x3vl_rvv(
    size_t m, size_t n, size_t k,
    const void *A, size_t lda, size_t k_idx_a,
    const void *B, size_t ldb, size_t k_idx_b,
    const void *C, size_t ldc,
    void *D, size_t ldd,
    double alpha, double beta
);
```

### Getter functions

Every sgemm/dgemm variant also provides:

```c
size_t tqt_get_m_step_sgemm_f32_f32_f32t_8x3vl_rvv(void);  // returns 8
size_t tqt_get_n_step_sgemm_f32_f32_f32t_8x3vl_rvv(void);  // returns 3*vl

size_t tqt_get_a_offset_sgemm_f32_f32_f32t_8x3vl_rvv(size_t m_idx, size_t k_idx, size_t lda);
size_t tqt_get_b_offset_sgemm_f32_f32_f32t_8x3vl_rvv(size_t n_idx, size_t k_idx, size_t ldb);
size_t tqt_get_c_offset_sgemm_f32_f32_f32t_8x3vl_rvv(size_t m_idx, size_t n_idx, size_t ldc);
size_t tqt_get_d_offset_sgemm_f32_f32_f32t_8x3vl_rvv(size_t m_idx, size_t n_idx, size_t ldd);
size_t tqt_get_d_size_sgemm_f32_f32_f32t_8x3vl_rvv(size_t m, size_t n);
```

## Migration from Reference BLAS

### cblas_sgemm mapping

```c
// Standard CBLAS call:
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

// Equivalent TORQ-Tile call (NT = A row-major, B transposed):
// Note: TORQ-Tile writes to D (separate output buffer)
//       Set D = C and ldd = ldc for in-place accumulation.
tqt_run_sgemm_f32_f32_f32t_8x3vl_rvv(
    M, N, K,
    A, lda, /*k_idx_a=*/0,
    B, ldb, /*k_idx_b=*/0,
    C, ldc,
    /*D=*/C, /*ldd=*/ldc,    // in-place: overwrite C
    alpha, beta);
```

### Transpose mapping

| CBLAS TransA | CBLAS TransB | TORQ-Tile variant |
|--------------|--------------|-------------------|
| NoTrans | NoTrans | `sgemm_f32_f32_f32` |
| NoTrans | Trans   | `sgemm_f32_f32_f32t` |
| Trans   | NoTrans | `sgemm_f32_f32t_f32` |
| Trans   | Trans   | `sgemm_f32_f32t_f32t` |

### Leading dimension semantics

TORQ-Tile uses the same leading-dimension convention as CBLAS with
`CblasRowMajor`:

- For row-major A `[m, k]`: `lda >= k`
- For transposed A `[k, m]`: `lda >= m`
- For row-major B `[k, n]`: `ldb >= n`
- For transposed B `[n, k]`: `ldb >= k`

### k_idx parameters

The `k_idx_a` and `k_idx_b` parameters are for advanced K-tiling (splitting
the K loop across multiple kernel calls). For standard usage, set both to `0`.

## Tile Loop Pattern

The TORQ-Tile micro-kernels process one tile at a time. The caller must loop
over tiles:

```c
#include "tqt_sgemm_f32_f32_f32t_8x3vl_rvv.h"

void my_sgemm_nt(size_t M, size_t N, size_t K,
                 float alpha, const float *A, size_t lda,
                 const float *B, size_t ldb,
                 float beta, float *C, size_t ldc)
{
    const size_t m_step = tqt_get_m_step_sgemm_f32_f32_f32t_8x3vl_rvv();
    const size_t n_step = tqt_get_n_step_sgemm_f32_f32_f32t_8x3vl_rvv();

    for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
        for (size_t n_idx = 0; n_idx < N; n_idx += n_step) {
            size_t actual_m = (M - m_idx < m_step) ? M - m_idx : m_step;
            size_t actual_n = (N - n_idx < n_step) ? N - n_idx : n_step;

            const uint8_t *a_ptr = (const uint8_t *)A
                + tqt_get_a_offset_sgemm_f32_f32_f32t_8x3vl_rvv(m_idx, 0, lda);
            const uint8_t *b_ptr = (const uint8_t *)B
                + tqt_get_b_offset_sgemm_f32_f32_f32t_8x3vl_rvv(n_idx, 0, ldb);
            const uint8_t *c_ptr = (beta != 0.0f)
                ? (const uint8_t *)C
                  + tqt_get_c_offset_sgemm_f32_f32_f32t_8x3vl_rvv(m_idx, n_idx, ldc)
                : nullptr;
            uint8_t *d_ptr = (uint8_t *)C  // in-place
                + tqt_get_d_offset_sgemm_f32_f32_f32t_8x3vl_rvv(m_idx, n_idx, ldc);

            tqt_run_sgemm_f32_f32_f32t_8x3vl_rvv(
                actual_m, actual_n, K,
                a_ptr, lda, 0,
                b_ptr, ldb, 0,
                c_ptr, ldc,
                d_ptr, ldc,
                alpha, beta);
        }
    }
}
```

## Performance Notes

- **No packed variants**: Unlike the standard GEMM kernels, sgemm/dgemm do not
  offer `*p` packed versions. The caller is responsible for data layout. If you
  want to reuse packed weights for the special case `alpha=1, beta=0`, consider
  a compatible standard GEMM path with null C/bias and unbounded clamp. Standard
  GEMM does not take alpha/beta arguments; check its numerical contract as well.

- **RVV tile (8x3vl)**: On a VLEN=128 core, `vl = 4` floats, so
  `n_step = 12`. On VLEN=256, `n_step = 24`. The kernel handles tail M and
  tail N internally.

## Examples

The [basic GEMM scenario](../../examples/api/gemm/basic_gemm/basic_gemm.cpp) teaches
common stride, offset and interface setup. BLAS alpha/beta and transpose semantics
are described above; separate duplicate programs are not maintained for each form.
Implementation coverage remains in [RVV SGEMM tests](../../test/gemm/test_gemm_f32_f32_rvv.cpp),
and [RVV DGEMM tests](../../test/gemm/test_gemm_f64_f64_rvv.cpp).

## See Also

- [Naming Convention](../reference/naming_convention.md) -- how transpose
  suffixes map to memory layout
- [Packing Guide](packing.md) -- why sgemm/dgemm skip packing
- [Micro-kernel Tables](../reference/microkernel_tables.md) -- full list of
  all BLAS-style variants with tile sizes
