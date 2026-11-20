<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# Build options

Start with [Quick start](quick_start.md) to run an example. This page covers library
configuration, tests, scripts, installation and source selection. Commands run
from the repository root.

- [Configure and build](#configure-and-build)
- [CMake options](#cmake-options)
- [Tests and benchmarks](#tests-and-benchmarks)
- [Examples](#examples)
- [Installed package](#installed-package)
- [Toolchains and target ISA](#toolchains-and-target-isa)
- [Ztt target configuration](#ztt-target-configuration)
- [Freestanding core](#freestanding-core)
- [Conditional compilation](#conditional-compilation)
  - [Source groups](#source-groups)
  - [ISA probes and capability macros](#automatic-isa-selection-and-compile-configuration)
- [Troubleshooting](#troubleshooting)

## Configure and build

The commands below use `build`. You can choose another directory; use the same path
when configuring, building, testing and installing. Use separate directories when
keeping multiple toolchain or ISA configurations. Names such as `build/rvv` and
`build/ztt` are labels; compiler settings select the target ISA.

For a baseline RVV library on a cross-compilation host:

```bash
cmake -S . -B build \
    -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-unknown-linux-gnu-gcc.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d"
cmake --build build -j8
```

Put the cross compiler's `bin/` directory on `PATH`. For a native RISC-V Linux
build, omit the toolchain file and use flags supported by that machine.
`rv64gcv` enables RVV 1.0, including the baseline FP32 paths; it does not enable
FP16, BF16 or Ztt extensions.

## CMake options

| Option | Default | Purpose |
|--------|---------|---------|
| `TORQ_TILE_BUILD_SHARED` | `OFF` | Build a shared library instead of a static archive |
| `TORQ_TILE_BUILD_TEST` | `OFF` | Build the GoogleTest suites |
| `TORQ_TILE_BUILD_BENCHMARK` | `OFF` | Build the Google Benchmark executable |
| `TORQ_TILE_TEST_RUNNER` | Empty | Command prefix for CTest, such as a QEMU invocation |
| `TORQ_TILE_FREESTANDING` | `AUTO` | Follow compiler mode, enable freestanding (`ON`), or require hosted (`OFF`) |

Set `CMAKE_BUILD_TYPE=Release` for optimized builds, `Debug` for debugging, or
`RelWithDebInfo` for optimized code with debug information. Compiler defaults
control the optimization flags. Assertions depend on `NDEBUG`; a Debug build
does not automatically define the optional `TQT_DEBUG` diagnostic macro.

## Tests and benchmarks

After configuring the library, enable tests in the same build directory:

```bash
TQT_SYSROOT="$(riscv64-unknown-linux-gnu-gcc -print-sysroot)"
cmake -S . -B build -DTORQ_TILE_BUILD_TEST=ON \
    -DTORQ_TILE_TEST_RUNNER="qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 -L $TQT_SYSROOT"
cmake --build build -j8
(cd build/test && ctest --output-on-failure)
```

Use a runner supporting every ISA enabled in this build. A native build needs no
runner. `-L` supplies QEMU's target dynamic loader and libraries; it does not change
the compiler sysroot. See [Testing](../contributing/testing.md) for suite selection,
GoogleTest filters and numerical checks.

Tests fetch GoogleTest and benchmarks fetch Google Benchmark when enabled. For an
offline build, point the relevant environment variables at existing source trees
before configuring:

```bash
export FETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest
export FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK=/path/to/benchmark
```

Each directory must contain the dependency's top-level `CMakeLists.txt`. Set only
the variables needed for your targets. The library alone fetches neither dependency.

To add benchmarks, reconfigure with `-DTORQ_TILE_BUILD_BENCHMARK=ON`, then build.
The executable is `<build>/benchmark/torq_tile_benchmark` and accepts Google Benchmark options:

```bash
cmake -S . -B build -DTORQ_TILE_BUILD_BENCHMARK=ON
cmake --build build --target torq_tile_benchmark -j8
```

Run the resulting binary on the target board for performance measurements. QEMU is
useful for execution checks but does not measure hardware performance. See
[Performance tuning](../guide/performance_tuning.md#benchmark-methodology).

## Examples

Each API leaf directory under `examples/api/` is an independent CMake project.
The [example index](../../examples/README.md#build-one-directory) gives a complete
configuration command. Examples build the local library automatically, or reuse
an installed package via `-Dtorq_tile_DIR=/path/to/prefix/lib/cmake/torq_tile`.

`TORQ_TILE_EXAMPLE_RUNNER` is the examples' execution prefix; it is separate from
`TORQ_TILE_TEST_RUNNER`. Native builds register CTest checks automatically. Cross
builds register them when a runner is provided; otherwise they only build.
CTest runs the example directly and checks its exit status and failure markers;
this path does not require Python.

## Installed package

Install the build directory configured above to a clean prefix. If you chose a
different directory or ISA configuration, use that same directory here:

```bash
cmake --build build -j8
cmake --install build --prefix /path/to/torq-tile-package
```

The package contains the library, `include/torq_tile/tqt_compile_config.h`, and
only the selected kernels' public API/interface headers and their dependencies.
Internal packing, math and kernel implementation headers are excluded. Public
pack functions remain available through the corresponding kernel API headers.
Use a clean prefix for each ISA configuration: installation does not remove
files left by an older, broader package.

CMake consumers can use the installed target:

```cmake
cmake_minimum_required(VERSION 3.16)
project(consumer LANGUAGES C CXX)
find_package(torq_tile CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE torq_tile::torq_tile)
```

Configure the consumer with `-DCMAKE_PREFIX_PATH=/path/to/torq-tile-package` and
its own toolchain/ISA flags. The prefix can be moved; the exported target carries
the include path and link dependencies, without adding backend ISA flags to the
consumer. Include public headers as `<torq_tile/...>` instead of `"src/..."`,
and use `TORQ_TILE_HAS_*` to conditionally include/register available groups.
Native FP16/BF16 declarations still require consumer compiler type support.
The application must check runtime hardware support before calling a backend.

For direct linking, add `-I<prefix>/include -L<prefix>/lib -ltorq_tile -lm`.
Use a C++ linker for the static library, including when the caller is compiled
as C, because the library uses the C++ runtime for lazy initialization. The CMake
example enables both languages so C-only callers can also select that linker.

Installation headers are generated under `<build>/install/include/torq_tile/`;
only their project include paths are rewritten. The original source headers and
direct `.cpp` integration remain unchanged and do not need a generated config.

## Toolchains and target ISA

The supplied GCC toolchain selects `riscv64-unknown-linux-gnu-gcc` and `g++` from
`PATH`. The [Clang toolchain](../../cmake/riscv64-unknown-linux-gnu-clang.toolchain.cmake)
selects `clang`/`clang++` with a RISC-V target triple and accepts
`RISCV_GCC_INSTALL_DIR` and `RISCV_SYSROOT` for target headers and runtime libraries.
Both require compiler support for the selected instructions and intrinsics;
a compiler version number alone does not establish Ztt support.

To select explicit compiler paths, configure a fresh directory without a supplied
toolchain file:

```bash
cmake -S . -B build/custom \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=/path/to/riscv-c-compiler \
    -DCMAKE_CXX_COMPILER=/path/to/riscv-cxx-compiler \
    -DCMAKE_SYSROOT=/path/to/sysroot \
    -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d"
```

CMake selects kernels using the **C++** compiler and flags. Keep the C and C++ ABI
settings compatible. For explicit Clang drivers, also set
`CMAKE_C_COMPILER_TARGET` and `CMAKE_CXX_COMPILER_TARGET` to the required target
triple and provide their GCC runtime configuration.

| Target | Example compiler flags | Requirement |
|--------|------------------------|-------------|
| Baseline RVV | `-march=rv64gcv -mabi=lp64d` | RVV 1.0 instructions and intrinsics |
| RVV FP16 | `-march=rv64gcv_zfh_zvfh -mabi=lp64d` | FP16 arithmetic and widening operations |
| RVV BF16 | `-march=rv64gcv_zvfbfwma -mabi=lp64d` | BF16 types and widening FMA support |
| Ztt | See the recipe below | Matching Ztt compiler profile and runtime |

These are configuration examples, not a complete CPU compatibility table. Consult
the selected header and the [source groups](#source-groups)
for operation-specific requirements. Compiler flags and QEMU CPU properties use different configuration interfaces.
RVV 0.7 is not compatible with RVV 1.0 kernels.

<a id="ztt-toolchain-and-runner"></a>

## Ztt target configuration

Ztt requires a compiler supporting the ISA and intrinsic API used by the sources,
and a matching runtime. Compiler flags select UDS and register counts; `amenlen`
is read at runtime. RVV are not prerequisites.

With compatible tools, this configures a UDS32/M16/A4 build in `build/ztt`, keeping
it separate from the earlier library configuration:

```bash
cmake -S . -B build/ztt \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=/path/to/riscv-c-compiler \
    -DCMAKE_CXX_COMPILER=/path/to/riscv-cxx-compiler \
    -DCMAKE_SYSROOT=/path/to/sysroot \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=rv64gc_ztt_zttuds32b_zttm16_ztta4 -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gc_ztt_zttuds32b_zttm16_ztta4 -mabi=lp64d" \
    -DTORQ_TILE_BUILD_TEST=ON \
    -DTORQ_TILE_TEST_RUNNER="/path/to/qemu-riscv64 -cpu rv64,x-ztt=true,amenlen=32,ameudsz=32,ame-num-mregs=16,ame-num-accregs=4 -L /path/to/sysroot"
cmake --build build/ztt -j8
(cd build/ztt/test && ctest --output-on-failure)
```

The QEMU properties above require the companion Ztt implementation; they are not
portable to every QEMU build. Use a runtime that implements the selected profile.
Adaptive floating kernels cover UDS8/16/32/64/128 and A4/A8/A16; UDS128 F16/BF16
requires at least eight accumulators. Integer-dot builds use M16 or M32 and
A4/A8/A16; runtime `amenlen` is a power of two from 1 through 128. Compile/link
checks for a profile do not establish simulator or hardware execution support.

Ordinary-layout floating GEMM requires M aligned to `m_step` and N/K to `n_step`;
use packed output for arbitrary logical dimensions. Quantized Ztt GEMM accepts
arbitrary M/N with 256-aligned K and ordinary F32 output. Match the compiler UDS
and runtime geometry before packing or executing; see the
[Ztt API contract](../reference/api_reference.md#ztt-gemm).

## Freestanding core

The same kernel sources support hosted and freestanding compilation. Public header
paths, C entry points, native FP16/BF16 types and packed-buffer layouts are unchanged.
Hosted builds use lazy hardware geometry caches, diagnostics and a public `m` dependency.

```bash
cmake -S . -B build/freestanding \
    -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-unknown-linux-gnu-gcc.toolchain.cmake \
    -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d" \
    -DTORQ_TILE_FREESTANDING=ON
cmake --build build/freestanding -j8
```

`AUTO` follows the actual C++ compiler mode. `ON` adds `-ffreestanding` to the
core and its ISA probes; `OFF` rejects an already freestanding toolchain rather
than overriding it. Freestanding core compilation also uses `-fno-exceptions`
and `-fno-rtti`. These options are private to the library, so a hosted test
framework or application can consume it. Multi-config builds must use one
execution environment across configurations; otherwise use separate build trees.
The generated `TORQ_TILE_BUILT_FREESTANDING=0/1` describes the binary and does not
change the consumer's environment.

Direct source integration uses the same compiler flags. The optional C/C++ macro
`TORQ_TILE_FREESTANDING` accepts only numeric `0` or `1` and must agree with
`__STDC_HOSTED__`; defining it does not replace `-ffreestanding`. Public headers
include `tqt_runtime.h` as a support dependency automatically. SDK header injection
through `TORQ_TILE_PLATFORM_HEADER` is unsupported: remove that experimental
macro and provide actual runtime services through the platform's link.

Freestanding builds omit hosted-only headers, stdio, abort and libm from the core.
`TORQ_TILE_ERROR_TRAP` defaults to `1` and cannot be set to `0` in this mode.
Debug checks remain active; `NDEBUG` still controls debug-only assertions.
RVV/Ztt geometry getters read their CSR on demand without a local-static guard;
they do not read optional CSRs at load time. Separate getter/run calls can read
the geometry again, and packing/execution must still use compatible hardware widths
and, for Ztt, UDS.

Memory builtins can produce external `memcpy`, `memmove`, `memset` or `memcmp`
calls, and the compiler may require arithmetic helpers. The platform must supply
their real implementations. `-ffreestanding` is not `-nostdlib`: startup, linker
scripts, compiler helper libraries and final symbol resolution belong to the
application/toolchain. The first supported integration paths are a static core
and direct source embedding; device shared-object loaders need platform-specific
validation. Core-only configuration does not require Threads or fetch test
dependencies. Test/benchmark targets still require their hosted dependencies.

For a freestanding toolchain whose initial CMake compiler check cannot link an
executable, set `CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` in that toolchain.
The project option cannot change checks that run during `project()`.

## Conditional compilation

### Source groups

CMake compile probes use the selected C++ compiler and target flags to select
complete source groups. The C compiler must use compatible ABI and ISA settings.

| Source variable | Public paths |
|---|---|
| `TORQ_TILE_RVV_SOURCES` | Baseline RVV FP32 and integer paths |
| `TORQ_TILE_RVV_F16_SOURCES` | FP16 and mixed paths requiring FP16 vector operations |
| `TORQ_TILE_RVV_BF16_SOURCES` | BF16 and mixed paths requiring BF16 operations |
| `TORQ_TILE_RVV_BF16_TO_F16_SOURCES` | BF16 input to FP16 output, with both conversion capabilities |
| `TORQ_TILE_RVV_F64_SOURCES` | FP64 BLAS-style GEMM |
| `TORQ_TILE_ZTT_GEMM_F32_SOURCES` | FP32 GEMM |
| `TORQ_TILE_ZTT_GEMM_F16_SOURCES` | FP16 GEMM with native or FP32 accumulation |
| `TORQ_TILE_ZTT_GEMM_BF16_SOURCES` | BF16 GEMM with native or FP32 accumulation |
| `TORQ_TILE_ZTT_GEMM_BLOCK_INT_U4_SOURCES` | Q4_K-derived integer-dot GEMM |
| `TORQ_TILE_ZTT_GEMM_BLOCK_INT_I8_SOURCES` | Expanded Q6_K integer-dot GEMM |

### Automatic ISA selection and compile configuration

Set target extensions through the compiler's `-march` option. Unsupported
`TORQ_TILE_ENABLE_*` switches are rejected. To exclude a datatype group, choose
flags that omit its required extensions, then use a fresh build directory.

The generated `torq_tile/tqt_compile_config.h` describes the matching library's
selected sources. `TORQ_TILE_HAS_RVV` and `TORQ_TILE_HAS_ZTT` report whether any
implementation of that backend is present. Per-group macros, such as
`TORQ_TILE_HAS_RVV_F16` and `TORQ_TILE_HAS_ZTT_GEMM_F32`, describe complete groups.
Always test these numeric values with `#if`, not `#ifdef`; do not predefine them.
The public headers and installed package follow the actual source selection.

These macros do not perform runtime CPU detection. The caller must select a
compatible CPU, execution profile and geometry before invoking kernels or their
getters. Compilation and successful linking do not prove runtime ISA support.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| No kernels selected, or an architecture-guard error | Cross-compiler, C++ target flags, and the selected source's required extensions |
| Example target absent | The directory's scenario requirements and the matching library's capability header |
| Cross-built examples report no CTest tests | Set `TORQ_TILE_EXAMPLE_RUNNER`, then reconfigure |
| QEMU reports an illegal instruction | Its CPU model must support the entire binary's ISA, not just the first kernel |
| QEMU cannot find the dynamic loader | `-L` must point at the matching target sysroot |
| Link errors for hosted math/runtime helpers | Use the CMake target or link through the C++ driver with the required runtime and `-lm` |

Freestanding builds have a different runtime contract: provide real platform
services instead of adding hosted libraries to conceal missing dependencies.
