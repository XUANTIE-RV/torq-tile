<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.

    SPDX-License-Identifier: Apache-2.0
-->

# TORQ-Tile documentation

For the library overview, see the top-level [README](../README.md). Choose a reading
path below, or use the full index that follows.

| Goal | Reading path |
|------|--------------|
| Run the first example | [Quick start](getting_started/quick_start.md) → [Examples](../examples/README.md) |
| Integrate into an application | [Choose a kernel](getting_started/choosing_a_kernel.md) → [Integration](guide/integration.md) → [API reference](reference/api_reference.md) |
| Add or change a kernel | [Architecture](contributing/architecture.md) → [Adding kernels](contributing/adding_kernels.md) → [Testing](contributing/testing.md) |

## Getting started

New to TORQ-Tile? Start here.

| Document | Description |
|----------|-------------|
| [Quick start](getting_started/quick_start.md) | Build and run an FP32 RVV example with CMake. |
| [Build options](getting_started/build_options.md) | CMake options, toolchains, scripts, installation and source selection. |
| [Choosing a kernel](getting_started/choosing_a_kernel.md) | Select by operation, precision, ISA, layout and tile. |

## Guides

| Document | Description |
|----------|-------------|
| [Packing](guide/packing.md) | The packing system: when to use `t` vs `p`, pack helpers, buffer sizing, and workflow. |
| [Quantization](guide/quantization.md) | Integer and low-precision floating formats, scales, zero points and execution layouts. |
| [BLAS-style GEMM](guide/blas_style_gemm.md) | sgemm/dgemm alpha/beta semantics, transpose combinations and reference BLAS migration. |
| [Framework integration](guide/integration.md) | How to embed torq-tile into an ML inference framework (NCNN, TVM, ONNX Runtime, etc.). |
| [Performance tuning](guide/performance_tuning.md) | Tile selection strategy, packed vs transposed tradeoffs, benchmark methodology. |
| [Depthwise convolution](guide/depthwise_conv.md) | Generic vs specialized dwconv kernels, layout choice, packing, threading patterns. |
| [Implicit GEMM](guide/implicit_gemm.md) | Convolution directly from input geometry, without caller-allocated im2col. |

## Reference

Look-up material — definitions, catalogs, API patterns.

| Document | Description |
|----------|-------------|
| [Naming convention](reference/naming_convention.md) | Name grammar, layout tags, datatype fields and tile dimensions. |
| [Micro-kernel tables](reference/microkernel_tables.md) | Complete catalog of every kernel source file shipped in the library. |
| [API reference](reference/api_reference.md) | Getters, run signatures, interface structs and numerical contracts. |

## Contributing

For developers who want to add kernels, fix bugs, or improve the library.

| Document | Description |
|----------|-------------|
| [Adding kernels](contributing/adding_kernels.md) | Step-by-step guide: file generation, CMake registration, test/bench/example wiring. |
| [Maintaining examples](contributing/examples.md) | Scenario selection, standalone project registration, and correctness checks. |
| [Testing](contributing/testing.md) | Test framework, reference implementations, tolerance rules, running with QEMU. |
| [Coding conventions](contributing/coding_conventions.md) | Code style, clang-format settings, SPDX headers, architecture guards. |
| [Architecture](contributing/architecture.md) | Internal design deep dive: dispatch tables, template kernels, family convention, pack format. |

---

## Repository directory overview

| Directory | Purpose |
|-----------|---------|
| `src/` | Library sources: `common/` (shared headers), `gemm/`, `conv/`, `dwconv/`, `elementwise/`, `reduction/`, `activation/` (micro-kernel implementations). |
| `test/` | Google Test suites, one file per `<op>_<types>_<arch>` family. |
| `benchmark/` | Google Benchmark suites, same naming convention as tests. |
| `examples/` | [Scenario-based API examples and independent integration consumers](../examples/README.md). |
| `scripts/` | The shared source formatter (`format-all.sh`). |
| `cmake/` | Toolchain files and CMake modules. |
| `docs/` | This documentation directory. |
