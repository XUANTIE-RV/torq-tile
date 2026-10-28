<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
    SPDX-License-Identifier: Apache-2.0
-->

# TORQ-Tile

[![XuanTie](https://img.shields.io/badge/XuanTie-Homepage-8145E2.svg)](https://www.xrvm.cn/)
![RISC-V](https://img.shields.io/badge/arch-RISC--V-F2AC10.svg?logo=riscv)
![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

TORQ-Tile is an open-source library that provides optimized performance-critical routines, also known as **micro-kernels** and **tile kernels**, for artificial intelligence (AI) workloads tailored for RISC-V CPUs, with special optimizations for XuanTie processors.

These routines are tuned to exploit the capabilities of RISC-V architecture extensions including RVV (RISC-V Vector Extension) and other extensions, aiming to maximize performance on XuanTie processors.

The TORQ-Tile library has been designed for ease of adoption into C or C++ machine learning (ML) and AI frameworks. To integrate a micro-kernel directly, include its **.cpp** and **.h** files together with all local header and implementation dependencies. Convolution wrappers can also require GEMM implementation files; the standalone examples list complete source sets.

## Who is this library for?

TORQ-Tile is a library for AI/ML framework developers interested in accelerating the computation on RISC-V CPUs, particularly XuanTie processors.

## What is a micro-kernel?

A micro-kernel, or **ukernel**, can be defined as a near-minimum amount of software to accelerate a given ML operator with high performance.

Following are examples of a micro-kernel:

- Function to perform matrix multiplication
- Function to perform element-wise vector operations (add, subtract, multiply)
- Function to perform reduction operations (sum or mean across rows or columns)

*However, why are the preceding operations not called kernels or functions instead?*

**This is because the micro-kernels are designed to give the flexibility to process also a portion of the output tensor.**

> ℹ️ The API of the micro-kernel is intended to provide the flexibility to dispatch the operation among different working threads or process only a section of the output tensor. Therefore, the caller can control what to process and how.

A micro-kernel exists for different RISC-V architecture extensions and computational parameters (for example, different output tile sizes). These implementations are called **micro-kernel variants**. Variants implement the same operation with the layout, accumulation precision and numerical limits documented by their public headers.

## Key features

Some of the key features of TORQ-Tile are the following:

- No dependencies on external libraries

- No dynamic memory allocation

- No memory management​

- No scheduling

- Stateless, stable, and consistent API​

- Performance-critical compute-bound and memory-bound micro-kernels

- Specialized micro-kernels utilizing different RISC-V CPU architectural features (for example, **RVV**)

- Specialized micro-kernels for different fusion patterns

- Micro-kernel as a standalone library, consisting of **.cpp** and **.h** files

> ℹ️ The micro-kernel API is designed to be as generic as possible for integration into third-party runtimes.

## Supported instructions and extensions

- RISC-V Vector Extension (RVV 1.0)
- Attached Matrix Extension (Ztt 0.6)

## Filename convention

The `src/gemm` directory is the home for all GEMM (General Matrix Multiply) micro-kernels. The micro-kernels are grouped in separate directories based on the data types. For example, all the FP16 matrix-multiplication micro-kernels are held in the `gemm_f16_f16/` directory.

Inside the operator directory, you can find:

- *The common utilities*, which are helper functions necessary for the correct functioning of the micro-kernels. For example, packing utilities are available in `src/common/tqt_gemm_pack_rvv.h`.
- *The micro-kernels* files, which are held in separate sub-directories.

The name of the micro-kernel folder provides the description of the operation performed and the data type of the destination and source tensors. The general syntax for the micro-kernel folder is as follows:

`<op>_<dst-data-type>_<src0-data-type>_<src1-data-type>_...`

All **.cpp** and **.h** pair files in that folder are micro-kernel variants. The variants are differentiated by specifying the computational parameters (for example, the tile size like `8x3vl`), the RISC-V extension (for example, RVV). The general syntax for the micro-kernel variant is as follows:

`tqt_<op>_<data-types>_<compute-params>_<extension>.cpp/.h`

For example:
- `tqt_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv.h` - FP16 GEMM with bias and clamp, using 8x3vl tile size, optimized for RVV
- `tqt_gemm_1xnbias_clamp_f32_f32p_f32p_8x3vl_rvv.h` - FP32 packed GEMM with bias and clamp, using RVV

All functions defined in the **.h** header file of the micro-kernel variant follow the naming pattern:

`tqt_<action>_<KERNEL_FULL>(...)`

For example: `tqt_get_m_step_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv()`, `tqt_run_gemm_1xnbias_clamp_f16_f16_f16t_8x3vl_rvv(...)`.

## Supported micro-kernels

The table lists every implementation family shipped under [src/](src/).
Counts refer to implementation `.cpp` files, including tile and accumulation
variants. CMake selects the subset supported by the target compiler and flags;
the counts do not imply that every build contains every variant.

| Operation group | Family directory | RVV variants | Ztt variants |
|-----------------|------------------|-------------:|-------------:|
| gemm | [gemm_bf16_bf16](src/gemm/gemm_bf16_bf16/) | 15 | 6 |
| gemm | [gemm_f16_f16](src/gemm/gemm_f16_f16/) | 14 | 6 |
| gemm | [gemm_f16_i4](src/gemm/gemm_f16_i4/) | 8 | 0 |
| gemm | [gemm_f32_bf16](src/gemm/gemm_f32_bf16/) | 2 | 0 |
| gemm | [gemm_f32_f16](src/gemm/gemm_f32_f16/) | 2 | 0 |
| gemm | [gemm_f32_f32](src/gemm/gemm_f32_f32/) | 16 | 3 |
| gemm | [gemm_f64_f64](src/gemm/gemm_f64_f64/) | 4 | 0 |
| gemm | [gemm_i8_b1](src/gemm/gemm_i8_b1/) | 1 | 0 |
| gemm | [gemm_i8_i2](src/gemm/gemm_i8_i2/) | 4 | 0 |
| gemm | [gemm_i8_i3](src/gemm/gemm_i8_i3/) | 1 | 0 |
| gemm | [gemm_i8_i4](src/gemm/gemm_i8_i4/) | 44 | 0 |
| gemm | [gemm_i8_i5](src/gemm/gemm_i8_i5/) | 1 | 0 |
| gemm | [gemm_i8_i6](src/gemm/gemm_i8_i6/) | 1 | 0 |
| gemm | [gemm_i8_i8](src/gemm/gemm_i8_i8/) | 15 | 1 |
| gemm | [gemm_i8_u2](src/gemm/gemm_i8_u2/) | 2 | 0 |
| gemm | [gemm_i8_u4](src/gemm/gemm_i8_u4/) | 2 | 1 |
| gemm | [gemm_i8_u5](src/gemm/gemm_i8_u5/) | 1 | 0 |
| gemm | [gemm_i8_u8](src/gemm/gemm_i8_u8/) | 1 | 0 |
| gemm | [gemm_u8_u8](src/gemm/gemm_u8_u8/) | 4 | 0 |
| conv | [conv_bf16_bf16](src/conv/conv_bf16_bf16/) | 4 | 0 |
| conv | [conv_f16_f16](src/conv/conv_f16_f16/) | 5 | 0 |
| conv | [conv_f32_f32](src/conv/conv_f32_f32/) | 5 | 0 |
| conv | [conv_i8_i8](src/conv/conv_i8_i8/) | 4 | 0 |
| conv | [igemm_f16_f16](src/conv/igemm_f16_f16/) | 4 | 0 |
| conv | [igemm_f32_f32](src/conv/igemm_f32_f32/) | 4 | 0 |
| conv | [igemm_i8_i8](src/conv/igemm_i8_i8/) | 4 | 0 |
| dwconv | [dwconv_f16_f16](src/dwconv/dwconv_f16_f16/) | 11 | 0 |
| dwconv | [dwconv_f32_f32](src/dwconv/dwconv_f32_f32/) | 11 | 0 |
| elementwise | [add_f16_f16](src/elementwise/add_f16_f16/) | 2 | 0 |
| elementwise | [add_f32_f32](src/elementwise/add_f32_f32/) | 2 | 0 |
| elementwise | [div_f16_f16](src/elementwise/div_f16_f16/) | 2 | 0 |
| elementwise | [div_f32_f32](src/elementwise/div_f32_f32/) | 2 | 0 |
| elementwise | [mul_f16_f16](src/elementwise/mul_f16_f16/) | 2 | 0 |
| elementwise | [mul_f32_f32](src/elementwise/mul_f32_f32/) | 2 | 0 |
| elementwise | [sub_f16_f16](src/elementwise/sub_f16_f16/) | 2 | 0 |
| elementwise | [sub_f32_f32](src/elementwise/sub_f32_f32/) | 2 | 0 |
| reduction | [mean_f16_f16](src/reduction/mean_f16_f16/) | 2 | 0 |
| reduction | [mean_f32_f32](src/reduction/mean_f32_f32/) | 2 | 0 |
| reduction | [sum_f16_f16](src/reduction/sum_f16_f16/) | 2 | 0 |
| reduction | [sum_f32_f32](src/reduction/sum_f32_f32/) | 2 | 0 |
| activation | [geglu_f16_f16_f16](src/activation/geglu_f16_f16_f16/) | 1 | 0 |
| activation | [geglu_f32_f32_f32](src/activation/geglu_f32_f32_f32/) | 1 | 0 |
| activation | [gelu_f16_f16](src/activation/gelu_f16_f16/) | 1 | 0 |
| activation | [gelu_f32_f32](src/activation/gelu_f32_f32/) | 1 | 0 |
| activation | [reglu_f16_f16_f16](src/activation/reglu_f16_f16_f16/) | 1 | 0 |
| activation | [reglu_f32_f32_f32](src/activation/reglu_f32_f32_f32/) | 1 | 0 |
| activation | [relu_f16_f16](src/activation/relu_f16_f16/) | 1 | 0 |
| activation | [relu_f32_f32](src/activation/relu_f32_f32/) | 1 | 0 |
| activation | [silu_f16_f16](src/activation/silu_f16_f16/) | 1 | 0 |
| activation | [silu_f32_f32](src/activation/silu_f32_f32/) | 1 | 0 |
| activation | [swiglu_f16_f16_f16](src/activation/swiglu_f16_f16_f16/) | 1 | 0 |
| activation | [swiglu_f32_f32_f32](src/activation/swiglu_f32_f32_f32/) | 1 | 0 |
| **Total** | **52 families** | **226** | **17** |

GEMM families group the base input types; output type, bias precision, quantization,
layout and accumulation mode are specified by each kernel base and public header.
The RVV families include BLAS-style SGEMM/DGEMM, W16A32, W4A16, W8A8, W4A8,
W2A8 and block-quantized integer-dot paths. `1xnbiasf32` selects FP32 bias for the
applicable FP16/BF16-output kernels; it is distinct from the output-type bias
contract of `1xnbias`. W2A8 includes floating-output and INT8-output variants.

Ztt provides FP32 GEMM, FP16/BF16 GEMM with native or FP32 accumulation, and
Q4_K-derived or expanded Q6_K-derived integer-dot GEMM. The quantized kernels
consume normalized execution tensors (`qsi8d256p` with `qau4c32p` or `qsi8c16p`),
not raw model-file blocks. Their packed layouts are specific to the backend and
target geometry; they are not interchangeable with RVV buffers. Ordinary-layout
floating Ztt kernels require aligned dimensions as documented in their headers.
Packed floating variants provide packing and output-unpacking APIs for tails.

`conv` uses im2col plus GEMM, while `igemm` uses implicit GEMM. Depthwise kernels
cover generic and specialized spatial kernels; consult the variant header for
layout, stride and depth-multiplier requirements. Elementwise kernels provide
vector-vector and vector-scalar add/subtract/multiply/divide. Reduction kernels
provide row-wise and column-wise sum/mean, with FP32 accumulation for FP16.

RVV activations include ReLU, GeLU, SiLU and their gated forms ReGLU, GeGLU and
SwiGLU. GeLU/GeGLU use the tanh approximation; SiLU/SwiGLU use a polynomial exp.
FP16 activation arithmetic widens to FP32 internally. Gated APIs accept separate
strides for gate and linear operands, so a fused `[M, 2N]` buffer can be consumed
with `a = base`, `b = base + N`, and `lda = ldb = 2N`.

## Project Structure

```bash
torq-tile/
├── benchmark/        # Benchmark suites
├── cmake/            # CMake configuration files
├── examples/         # Usage examples
├── scripts/          # Formatter and compile isolation checks
├── src/              # Source code
│   ├── activation/   # Activation micro-kernel implementations
│   ├── common/       # Common utilities and headers
│   ├── conv/         # Convolution micro-kernel implementations
│   ├── dwconv/       # Depthwise convolution micro-kernel implementations
│   ├── elementwise/  # Elementwise micro-kernel implementations
│   ├── gemm/         # GEMM micro-kernel implementations
│   └── reduction/    # Reduction micro-kernel implementations
└── test/             # Test cases
```

## How to build

### Prerequisites

- CMake 3.16 or later, a build tool such as Make or Ninja, and a C11/C++17 compiler.
- A RISC-V toolchain supporting the extensions required by the selected kernels.
- For cross-compiled execution, a matching RISC-V Linux target or user-mode QEMU
  and target runtime libraries.
- For Ztt, a compiler with the `riscv_ztt.h` intrinsic API used by these sources
  and a matching AME 0.6 execution environment.

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS` | Compiler defaults | Set target ISA and ABI explicitly; kernel selection probes the actual C++ flags. |
| `TORQ_TILE_BUILD_SHARED` | `OFF` | Build a shared library instead of a static archive. |
| `TORQ_TILE_BUILD_TEST` | `OFF` | Build Google Test executables. |
| `TORQ_TILE_BUILD_BENCHMARK` | `OFF` | Build the Google Benchmark executable. |
| `TORQ_TILE_FREESTANDING` | `AUTO` | `AUTO` detects the compiler mode, `ON` adds freestanding core flags, and `OFF` requires hosted mode. |
| `TORQ_TILE_TEST_RUNNER` | Empty | Prefix for CTest commands, such as a QEMU invocation. |
| `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` | Unset | Local Google Test sources, supplied as a CMake cache variable or environment variable. |
| `FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK` | Unset | Local Google Benchmark sources, supplied in the same way. |

There are no manual backend enable switches. CMake compiles ISA probes and selects
compatible source groups. `torq_tile/tqt_compile_config.h` records which groups
were selected in that build; these macros do not detect runtime CPU support.
Check them with `#if`, rather than `#ifdef`. An unsupported target with no buildable
kernels is rejected.

### RVV library

For a native RISC-V build, omit `CMAKE_TOOLCHAIN_FILE`. For a cross build, put
`riscv64-unknown-linux-gnu-gcc` and `riscv64-unknown-linux-gnu-g++` on PATH:

```shell
cmake -S . -B build/rvv \
    -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-unknown-linux-gnu-gcc.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=rv64gcv_zfh_zvfh -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv_zfh_zvfh -mabi=lp64d"
cmake --build build/rvv --parallel
```

Use `rv64gcv` for baseline RVV, add `zfh_zvfh` for FP16, and add `zvfbfwma` for
BF16. BF16-input GEMM with FP16 output additionally needs `zvfhmin` or `zvfh`
vector conversions. To select all current RVV families, use
`-march=rv64gcv_zfh_zvfh_zvfbfwma -mabi=lp64d` with a compatible compiler.

### Build and run tests

Configure the same RVV build with tests enabled and a runner matching those flags:

```shell
cmake -S . -B build/rvv \
    -DTORQ_TILE_BUILD_TEST=ON \
    -DTORQ_TILE_TEST_RUNNER="qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64,zfh=true,zvfh=true -L /path/to/sysroot"
cmake --build build/rvv --parallel
ctest --test-dir build/rvv/test --output-on-failure --no-tests=error
```

If the build also enables BF16, the runner must support and enable both scalar
and vector BF16 conversions. For compatible QEMU builds, add
`zfbfmin=true,zvfbfmin=true,zvfbfwma=true` to the CPU properties. For native execution, leave `TORQ_TILE_TEST_RUNNER` empty. Enabling
`TORQ_TILE_BUILD_BENCHMARK=ON` builds `build/rvv/benchmark/torq_tile_benchmark`;
run it through the same runner, or directly on a matching RISC-V target.
Emulator timing is not a native-hardware performance measurement.

### Ztt library and tests

Use explicit compiler paths to select a compatible Ztt toolchain. This example
uses UDS32, 16 matrix registers and 4 accumulators; the runner properties must
match the compiler target profile:

```shell
cmake -S . -B build/ztt \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=/path/to/ztt-toolchain/bin/clang \
    -DCMAKE_CXX_COMPILER=/path/to/ztt-toolchain/bin/clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=rv64gc_ztt_zttuds32b_zttm16_ztta4 -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gc_ztt_zttuds32b_zttm16_ztta4 -mabi=lp64d" \
    -DTORQ_TILE_BUILD_TEST=ON \
    -DTORQ_TILE_TEST_RUNNER="/path/to/ztt-qemu -cpu rv64,x-ztt=true,amenlen=32,ameudsz=32,ame-num-mregs=16,ame-num-accregs=4 -L /path/to/ztt-toolchain/sysroot"
cmake --build build/ztt --parallel
ctest --test-dir build/ztt/test --output-on-failure --no-tests=error
```

A compatible GCC toolchain can also be selected through `CMAKE_C_COMPILER` and
`CMAKE_CXX_COMPILER`. The build probes the actual implementations, including
packing and epilogues. Check `TORQ_TILE_HAS_ZTT_GEMM_F32`,
`TORQ_TILE_HAS_ZTT_GEMM_F16`, `TORQ_TILE_HAS_ZTT_GEMM_BF16`,
`TORQ_TILE_HAS_ZTT_GEMM_BLOCK_INT_U4` and `TORQ_TILE_HAS_ZTT_GEMM_BLOCK_INT_I8`
in the generated capability header to confirm the selected coverage.

### Standalone examples

Each directory in [examples/](examples/) has its own CMake build and lists the
implementation sources required by that example. For example:

```shell
cmake -S examples/relu_f32_f32_rvv -B build/example-relu \
    -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/riscv64-unknown-linux-gnu-gcc.toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=rv64gcv -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d"
cmake --build build/example-relu --parallel
qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 build/example-relu/relu_f32_f32_rvv
```

Ztt examples use the same CMake compiler and ISA settings as the Ztt library.
They include a packed FP32 example in
[gemm_1xnbias_clamp_f32p_f32p_f32p_ztt](examples/gemm_1xnbias_clamp_f32p_f32p_f32p_ztt/).
The example CMake files request static linking, so a matching static target
runtime must be available in the toolchain.

### Offline dependencies

Tests use Google Test v1.17.0; benchmarks use Google Benchmark v1.9.5. CMake
fetches only the enabled dependencies. To reuse local source checkouts, add:

```shell
cmake -S . -B build/rvv \
    -DTORQ_TILE_BUILD_TEST=ON -DTORQ_TILE_BUILD_BENCHMARK=ON \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK=/path/to/benchmark
```

Use the configured cross-build directory above, or include its compiler and ISA
settings when creating a fresh build directory. Environment variables with the
same names are also accepted.

### Installation and freestanding integration

```shell
cmake --install build/rvv --prefix /path/to/install
```

The package installs the selected public headers and their local dependencies,
the capability header, the library and a relocatable CMake package. Consumers
can use `find_package(torq_tile CONFIG REQUIRED)` and link `torq_tile::torq_tile`.

For a core-only freestanding build, set `TORQ_TILE_FREESTANDING=ON` and leave tests
and benchmarks disabled. The platform supplies any required memory functions and
compiler runtime helpers at link time; this library does not provide a device
runtime. Consumer compilation must match its own hosted/freestanding environment.

The retained [compile isolation checks](scripts/compile_isolation/README.md) cover
source selection, headers, installation and bounded runtime contracts. Invoke
them directly with `python3 scripts/compile_isolation/check.py`.

## Release

### Cadence

Releases will be done regularly. All releases can be found in the release section.

### Version

The release version conforms to Semantic Versioning.

> ⚠️ Please note that API modifications, including function name changes, and feature enhancements may occur without advance notice.

## Support

Please raise a GitLab/GitHub Issue for technical support.

## Frequently Asked Questions (FAQ)

### Is x86 or Arm supported?

No, TORQ-Tile is a library of micro-kernels optimized for the RISC-V architecture, specifically XuanTie processors. The micro-kernels use specific RISC-V instruction set architecture features which prevent compilation for other architectures like x86 or Arm. Note that this applies to all micro-kernels including the packing and quantization micro-kernels.

### What ML operators are supported?

TORQ-Tile does not provide traditional ML operators; it implements a set of optimized micro-kernels which can be used as building blocks to implement ML operators. Please refer to the sections [What is a micro-kernel](#what-is-a-micro-kernel), [Supported micro-kernels](#supported-micro-kernels) for more information.

### What CPUs does TORQ-Tile support?

Any CPU that implements the RISC-V architecture extensions listed in section [Supported instructions and extensions](#supported-instructions-and-extensions) is supported by TORQ-Tile. The library is specifically optimized for XuanTie processors but can run on any RISC-V CPU with the required extensions. However, every micro-kernel does not exist in versions for all architecture extensions. Each micro-kernel advertises the required architecture extensions through the micro-kernel name.

Most operating systems provide a mechanism to list the architecture extensions supported by the runtime environment, e.g. `cat /proc/cpuinfo` on Linux.

### What is the difference between TORQ-Tile and other RISC-V AI libraries?

TORQ-Tile focuses on providing low-level, highly optimized micro-kernels that can be easily integrated into existing AI/ML frameworks. Unlike full-featured libraries that provide complete operator implementations and runtime management, TORQ-Tile offers building blocks that give framework developers fine-grained control over computation, memory management, and threading.

**TORQ-Tile is designed for frameworks where the runtime, memory manager, thread management, and fusion mechanisms are already available**.

### Can the micro-kernels be multi-threaded?

**Yes, they can**. The micro-kernel can be dispatched among multiple threads using the thread management available in the target AI/ML framework.

*The micro-kernel does not use any internal threading mechanism*. However, the micro-kernel's API is designed to allow the computation to be carried out only on specific areas of the output tensor. Therefore, this mechanism is sufficient to split the workload on parallel threads.

Callers may partition independent output regions across threads, subject to each
variant's offset, alignment and scratch-buffer requirements. Keep writable
output and scratch regions disjoint between simultaneous calls.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
