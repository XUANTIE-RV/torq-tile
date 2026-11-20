<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Testing Guide

This document describes the test framework, how to run tests, and how to add new
test cases for TORQ-Tile micro-kernels.

## Framework

TORQ-Tile uses [Google Test](https://github.com/google/googletest) (gtest) for kernel
correctness tests. Tests are cross-compiled alongside the library and executed on
target hardware or under QEMU emulation. Runtime and toolchain contracts require
additional checks beyond numerical kernel tests.

## Running tests

Start with the [RVV library configuration](../getting_started/build_options.md#configure-and-build),
then enable tests in the same directory:

```bash
TQT_SYSROOT="$(riscv64-unknown-linux-gnu-gcc -print-sysroot)"
cmake -S . -B build -DTORQ_TILE_BUILD_TEST=ON \
    -DTORQ_TILE_TEST_RUNNER="qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 -L $TQT_SYSROOT"
cmake --build build -j8
(cd build/test && ctest --output-on-failure)
```

For a native build, omit the runner. For Ztt or additional RVV datatypes,
use matching compiler flags and an execution environment supporting all selected
instructions. [Build options](../getting_started/build_options.md#tests-and-benchmarks)
also covers offline GoogleTest sources.

To build and run one suite:

```bash
cmake --build build --target test_gemm_f32_f32_rvv -j8
(cd build/test && ctest -R '^test_gemm_f32_f32_rvv$' --output-on-failure)
```

For individual GoogleTest cases, first list names, then use an exact name or a
matching filter:

```bash
qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 -L "$TQT_SYSROOT" \
    build/test/test_gemm_f32_f32_rvv --gtest_list_tests
qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 -L "$TQT_SYSROOT" \
    build/test/test_gemm_f32_f32_rvv --gtest_filter='*8x3vl*'
```

Check that the requested tests actually execute. Zero selected tests is not a
pass. QEMU validates behavior; performance requires target hardware.

## Test File Organization

Test files are organized by operation category, then by family and architecture. The simplified
layout below shows the main categories and common helper types rather than every file:

```
test/
├── CMakeLists.txt
├── common/
│   ├── *_test_runner.h         # Shared test harnesses for operation families
│   ├── reference_*.h/.cpp      # Reference implementations
│   └── data_generator.*, verify.*
├── gemm/
│   ├── test_gemm_f16_f16_rvv.cpp
│   ├── test_gemm_f32_f32_rvv.cpp
│   ├── test_gemm_i8_i8_rvv.cpp
│   └── ...
├── conv/
│   └── ...
├── dwconv/
│   └── ...
├── elementwise/
│   └── ...
├── reduction/
│   └── ...
└── activation/
    ├── test_relu_f32_f32_rvv.cpp
    ├── test_geglu_f32_f32_f32_rvv.cpp
    └── ...
```

Test grouping follows each descriptor's real family and architecture. Many matrix test files cover
all kernel-bases for an input family, while elementwise, reduction, and activation files use their
own operation/type names. For example, `test_gemm_f16_f16_rvv.cpp` tests:
- `gemm_1xnbias_clamp_f16_f16_f16` (all RVV tile variants: 8x3vl, 4x4vl, 2x8vl)
- `gemm_1xnbias_clamp_f16_f16_f16t` (transposed B)
- `gemm_1xnbias_clamp_f16_f16_f16p` (packed B)
- `gemm_1xnbias_clamp_f16_f16p_f16p` (both packed)
- `gemm_mx1bias_clamp_f16_f16_f16` (column bias)
- `gemm_mx1bias_clamp_f16_f16p_f16p` (column bias, packed)

## Test Pattern

Every correctness test follows the same high-level pattern, using descriptor-specific operands and
runner types:

1. **Generate inputs** -- for example A/B/bias/C for GEMM, one input for unary activation, or A/B
   for gated activation.
2. **Compute reference output** using the matching reference model, including the
   variant's intended accumulation and rounding behavior.
3. **Call the kernel under test** via the public `tqt_run_*` function.
4. **Compare** the kernel output against the reference with appropriate tolerance.

### Reference Implementation

Reference implementations are provided in `test/common/reference_*.h/.cpp`. Test runners call
these helpers while handling shared data preparation, kernel invocation, and verification. The
reference helpers perform the same mathematical operation as the kernel using straightforward
implementations with `double` or `float` accumulators where appropriate to minimize reference
error.

For packed-layout kernels, the test runner calls the pack helper
(`tqt_run_a_pack_*` / `tqt_run_b_pack_*` / `tqt_run_bt_pack_*`) before invoking
the kernel, mirroring the intended usage pattern.

### Tolerance guidelines

Use the selected runner and variant's numerical contract. The shared
[`verify_gemm_result`](../../test/common/verify.h) compares maximum absolute error
and cosine similarity; it does not impose one relative-error threshold per dtype.
Individual runners and dedicated tests may use different comparisons.

Choose tolerances against the reference precision, input range, reduction length,
quantization and intermediate rounding. Integer accumulation and requantization
need their own exactness/rounding checks. NaNs, infinities, signed zeros, padding
and aliases require explicit assertions rather than an aggregate similarity score.
Keep the reason for each exceptional tolerance with the affected test.

## Adding a New Test

### Adding to an Existing Test File

If you are adding a new tile variant for an existing kernel-base in a family that
already has a test file, add a new section to that file:

```cpp
// ============================================================================
// With 1xN bias, With clamp, NonPacked, B RowMajor, Tile=4x4vl
// ============================================================================

static NonPackedGemmFuncs<float16_t> make_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv_funcs()
{
    return {
        tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_n_step_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_a_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_b_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_c_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_bias_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_d_offset_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_get_d_size_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
        tqt_run_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
    };
}

class Test_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv
    : public testing::TestWithParam<GemmTestParams>
{
};

TEST_P(Test_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv, Correctness)
{
    auto funcs = make_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv_funcs();
    run_nonpacked_gemm_test<float16_t, float16_t, BLayout::kRowMajor, BiasMode::k1xN>(
        GetParam(), funcs);
}

INSTANTIATE_TEST_SUITE_P(GemmTests,
                         Test_gemm_1xnbias_clamp_f16_f16_f16_4x4vl_rvv,
                         common_test_values(),
                         GemmTestParamNameGenerator());
```

### Creating a New Test File

If you are adding a family+arch combination that has no test file yet, follow the
shape of an existing one. The steps below are written against `gemm_f64_f64_rvv`
(already in the tree) as a filled-in illustration — substitute your own family and
arch throughout:

1. Create `test/<op-group>/test_<family>_<arch>.cpp` — for this illustration,
   `test/gemm/test_gemm_f64_f64_rvv.cpp`.
2. Include the relevant kernel headers.
3. Define `common_test_values()` returning a `testing::Values(...)` set of
   `GemmTestParams`.
4. Register the new file in `test/CMakeLists.txt`:
   ```cmake
   torq_tile_add_test(test_gemm_f64_f64_rvv
       gemm/test_gemm_f64_f64_rvv.cpp
   )
   ```

### Required Test Dimensions

Every kernel variant must cover the dimensions and semantic branches exposed by its public
contract. The following table applies to matrix-like M/N/K kernels; it is not a universal schema
for elementwise, reduction, or activation:

| Category | Example Values | Purpose |
|----------|---------------|---------|
| Aligned | (32,32,32), (64,64,64) | Verify basic correctness with full tiles |
| Small M | (1,32,32), (3,32,32) | Partial M-tile handling |
| Small N | (32,1,32), (32,3,32) | Partial N-tile handling |
| Non-aligned | (17,19,23), (31,35,63) | General partial-tile coverage |
| Large | (60,61,127), (128,128,256) | Multi-tile execution |
| With bias | (32,32,32, bias=true) | Bias addition path |
| With clamp | (32,32,32, clamp=[-100,150]) | Clamp path |
| With C | (32,32,32, C=true) | C accumulator init path |
| Combined | (60,61,127, C+bias+clamp) | All paths active |
| K-tiled | (127,127,255, tile_k=128) | Split-K execution |
| Loop orders | (31,35,63, order=MKN) | Non-default loop orders |

Floating GEMM additionally requires a traceable `ClampSpecialValues` test for each
public variant. Reuse the existing factory and legal packing/scale setup with
`gemm_clamp_test_utils.h`: the dedicated scope selects a function-pointer probe,
while ordinary tests continue to call the original function directly. The probe
injects mixed NaN, positive/negative infinity, finite +20/-20 and in-range neighbors through documented
floating bias or C, checks categories and infinity signs independently, and verifies
output row padding and end guards. `ClampDimensions` parameters add runtime tile/sub-block
boundaries before packing. A shadow C chain follows the runner's actual output-to-C
pointer dependency and adjacent K origins to check NaN propagation across real split-K;
it also checks a finite final clamp. The original runner output is restored after each
probe. XML properties named `clamp_probe_<public_run>` record actual calls, all five
ranges, executed M/N/K/K-origin shapes and split-K dependency counts per variant.
Do not encode float NaN bytes into quantized data
or inject C into interfaces that require it to be null.

Test exact unbounded, finite, both one-sided infinite and maximum-finite intervals.
Finite/one-sided NaN expectations must follow the original kernel min/max order;
ordinary cosine/error checks and production predicates cannot define this oracle.
Cover legal full, tail and multiple tiles using runtime getters, every reachable
MR/NR or matrix sub-block, supported split-K and zero-K paths. Include
both FP32-output W4A16 variants whose public float bounds narrow to FP16 internally.
GEMM-backed conv must opt into `ConvClampPolicy::kSkipUnbounded` in its reference;
floating IGEMM explicitly opts into the same policy. DWCONV references also skip
only exact unbounded intervals while preserving their existing accumulation model.
Each floating IGEMM/DWCONV public variant has a dedicated `ClampSpecialValues` test
using `conv_clamp_test_utils.h`. It checks NaN/Inf categories, infinity signs, five
clamp intervals, runtime tile boundaries, bias and weight sources, and output guards.
IGEMM additionally checks grouped dispatch, nonzero-K C, and real split-K chains: a
bounded request must not clamp an intermediate segment.

For Debug and Release, record enabled ISA/source units, list actual GoogleTest
registrations, and reconcile executed variant names with the source inventory.
A missing target or unregistered variant is not a pass or a skip. Check generated
machine code around the policy, conversions, min/max and stores. QEMU validates
behavior, not native hardware performance.

For elementwise, reduction, and activation, use the matching runner and cover full/tail steps,
minimal and zero shapes accepted by the interface, independent leading dimensions, output
padding, numerical edge values, and documented aliases. Activation tests must distinguish unary
`run_<op>(m,n,a,lda,c,ldc)` from gated
`run_<op>(m,n,a,lda,b,ldb,c,ldc)` and exercise both allowed gated alias directions.

Split-K, `k == 0` with null A/B, and non-zero K-origin packing are capability-specific matrix
tests. Require them only when the target header/interface and implementation advertise the
corresponding indices, pack actions, and accumulator semantics. If a kernel explicitly rejects a
non-zero K origin, record that limitation and test the supported path; do not report a global
compliance percentage from one helper-name search.

### Testing Quantized Kernels

For INT8/INT4 quantized kernels, additional test cases are needed:

- **Scale combinations**: test with scale values spanning different magnitudes
  (0.01, 0.1, 1.0, 10.0).
- **Zero-point values**: for asymmetric quantization (`qa`), test with zero_point = 0
  and non-zero values.
- **Block quantization**: for `c2`/`d2` granularity, test with different block lengths
  (bl = 2, 4, 8, 16, 32).
- **Saturation**: verify output clamps correctly at INT8 boundaries (-128, 127).

### Testing Packed Kernels

For packed variants (`f16p`, `f32p`, etc.) whose public interface exposes packing actions:

1. Pack the input using the corresponding `tqt_run_*_pack_*` helper function.
2. Run the kernel on packed data.
3. Compare against the reference computed from unpacked data.

GEMM runners may provide `run_packed_gemm_test<>()`; other descriptors must use the exact packing
API, if any, declared by their interface.

## Test Infrastructure Details

### Descriptor-Specific Runners

Shared runners include `gemm_test_runner.h`, `conv_test_runner.h`, `dwconv_test_runner.h`,
`igemm_test_runner.h`, `elementwise_test_runner.h`, `reduction_test_runner.h`, and
`activation_test_runner.h`, plus quantized specializations. Select the runner by descriptor and
read its parameter and function-pointer structs; do not assume every test uses GEMM fields.
`activation_test_runner.h` provides `ActivationTestParams`, `run_activation_*` for unary operands,
and `run_gated_activation_*` for independent A/B operands, alias directions, edge values, and fused
layout coverage.

### GemmTestParams (GEMM Only)

```cpp
GemmTestParams(size_t m, size_t n, size_t k, std::string name_ = "")
    .set_bias(bool enable)
    .set_clamp(float min, float max)
    .set_c(bool enable)
    .set_tile(size_t tile_m, size_t tile_n, size_t tile_k)
    .set_loop_order(LoopDim d0, LoopDim d1, LoopDim d2)
```

### NonPackedGemmFuncs / PackedGemmFuncs (GEMM Only)

These structs hold all function pointers for a kernel variant, matching the
`_interface.h` ukernel struct layout. They are populated by a `make_*_funcs()` helper
at the top of each test section.

### Test Output

On failure, the test prints:
- The (m, n, k) dimensions
- The first mismatched element's index, expected value, and actual value
- The maximum error across all elements

## Debugging Test Failures

1. **Run with `--gtest_filter`** to isolate the failing test case.
2. **Reduce dimensions**: temporarily change the test to use M=1, N=1, K=1 and
   incrementally increase to find the failing condition.
3. **Check partial tiles**: most bugs occur at tile boundaries. If (32,32,32) passes
   but (33,33,33) fails, the issue is in the partial-tile dispatch.
4. **Compare register allocation**: verify the `.cpp` dispatch table matches the
   inner kernel's register layout.
5. **Check K-loop boundary**: off-by-one in the steady-state loop often causes the
   last K iteration to produce garbage.

## Runtime geometry regression

Run the RVV suites at VLEN 128, 256 and 512 when the runner supports them. Repack
buffers under each geometry and include tails, multiplier ordering, permitted
nonzero K origins and split-K boundaries. Tests must reach the intended path;
a single-segment call does not validate a nonzero K origin.

For Ztt, match compiler UDS/register-count settings to the runner and vary the
supported runtime `amenlen` values. Use the corresponding getters for sizes,
origins and alignment. Ordinary floating layouts require aligned dimensions;
packed layouts and quantized integer-dot paths have their own tail contracts.
See the [Ztt build recipe](../getting_started/build_options.md#ztt-target-configuration).
Configure CTest with the matching runner for each case and record the geometry.

## Freestanding runtime regression

A successful kernel test does not establish the runtime contract. After changes
to common helpers, environment selection or geometry getters, cover hosted and
freestanding modes in Debug and Release with the actual target toolchain.

- Compile the selected source groups and public C/C++ consumers.
- Inspect objects for optional-ISA load-time initialization and verify hosted
  geometry caches versus freestanding reads on demand.
- Link the freestanding archive with real platform memory/arithmetic helpers and
  inspect unresolved symbols; verify that hosted stdio, libm and C++ guards are
  absent from the freestanding core.
- Exercise memory helpers, assertion traps and scalar special-value behavior.
- Check installed-package consumers and final-link closure under the application's
  real compiler/linker flags.

Retain compiler settings, selected sources, commands, symbol/link evidence and
runtime results. Compile-only checks do not establish execution, concurrent first
use, device-loader compatibility or native performance. Repeat for each backend;
RVV checks do not supply Ztt evidence. See the
[freestanding build contract](../getting_started/build_options.md#freestanding-core).
