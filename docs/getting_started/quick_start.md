<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Quick start

Build and run one FP32 GEMM example on RISC-V Linux or under QEMU. This path uses
RVV 1.0 (`rv64gcv`); Ztt, FP16 and BF16 are not required.

## Prepare the tools

Run the commands from the repository root. You need:

- CMake and Make or Ninja; the library requires CMake 3.16 or newer.
- A RISC-V C11/C++17 compiler with the RVV intrinsics used by the library.
- For cross-compilation, a matching RISC-V Linux sysroot and a user-mode
  `qemu-riscv64` with RVV 1.0 support.

The cross-compilation recipe uses the following compiler names on `PATH`:

```bash
riscv64-unknown-linux-gnu-gcc --version
riscv64-unknown-linux-gnu-g++ --version
qemu-riscv64 --version
TQT_SYSROOT="$(riscv64-unknown-linux-gnu-gcc -print-sysroot)"
```

`TQT_SYSROOT` must point to the target runtime supplied with the compiler. If
`-print-sysroot` returns an empty or unsuitable path, set it explicitly.

## Build the example

The example uses `build/example-basic` to keep its CMake configuration separate
from a library build in `build`. You can choose another build directory; use the
same path in the build and CTest commands below.

```bash
cmake -S examples/api/gemm/basic_gemm -B build/example-basic \
    -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/riscv64-unknown-linux-gnu-gcc.toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d" \
    -DTORQ_TILE_EXAMPLE_RUNNER="qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 -L $TQT_SYSROOT"
cmake --build build/example-basic --target tqt_example_basic_gemm -j8
```

This independent example project builds the local library automatically, with
tests and benchmarks disabled. It does not download GoogleTest or Google Benchmark.
There is no separate library-build step.

`-march` selects the compiler's instruction set; the QEMU `-cpu` option selects
the execution environment. Here both support RVV 1.0. On a compatible RISC-V Linux
machine, omit `CMAKE_TOOLCHAIN_FILE` and `TORQ_TILE_EXAMPLE_RUNNER` to build and run
natively. Use a fresh build directory when changing toolchains.

## Run and check the result

```bash
(cd build/example-basic && ctest --output-on-failure)
```

CTest reports `100% tests passed` on success. To see the example's variant names,
matrix dimensions and maximum absolute error, use verbose output:

```bash
(cd build/example-basic && ctest -V)
```

On success, the program exits with code zero and prints `Status: PASSED`. It compares each
output against a scalar reference, requires finite values, and checks that output
row padding remains unchanged. Its tolerance is
`abs(actual - reference) <= 1e-4 * (1 + abs(reference))` for this example's inputs.
This is a functional check, not a hardware performance measurement.

CTest runs the program directly and fails on a nonzero exit, a timeout or an
explicit failure marker. It does not require Python or a `PASSED` line.

## Understand the call

Read [basic_gemm.cpp](../../examples/api/gemm/basic_gemm/basic_gemm.cpp) alongside
its output. It computes `D = clamp(A * B + C + bias)` using ordinary row-major
`A[M,K]` and `B[K,N]` storage, with padded rows to demonstrate leading dimensions.

- The caller allocates A, B, C, bias and D. The kernel does not allocate memory.
- Leading dimensions count elements; offset and size getters return bytes.
- The example calls `run_gemm` for the whole region. The RVV implementation tiles
  that region internally and handles M/N tails.
- The interface struct lets the example call several compatible tile variants.
  Runtime ISA support must be established before calling their getters or kernels.

For a complete minimal application using the same ordinary-B contract, see
[minimal integration](../guide/integration.md#minimal-integration).

## Continue

| Task | Next page |
|------|-----------|
| Choose an operation, datatype and layout | [Choosing a kernel](choosing_a_kernel.md) |
| Configure tests, another ISA, installation or freestanding mode | [Build options](build_options.md) |
| Link the library into an application | [Framework integration](../guide/integration.md) |
| Learn packing, blocking or threading | [Example index](../../examples/README.md) |
| Look up exact variants and signatures | [Kernel catalog](../reference/microkernel_tables.md), [API reference](../reference/api_reference.md) |
