<!--
    SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
    SPDX-License-Identifier: Apache-2.0
-->

# Contributing to TORQ-Tile

Thank you for your interest in contributing to TORQ-Tile! Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository and clone your fork locally
2. Set up the development environment (see [README](./README.md#how-to-build) for build instructions)
3. Install pre-commit hooks: `pip install pre-commit && pre-commit install`
4. Create a new branch, make your changes, and submit a merge request

## Code Style

- **Formatting**: We use `clang-format-18` (config in `.clang-format`). The pre-commit hook formats files automatically on each commit
- **Standards**: C11 for `.c` files, C++17 for `.cpp` files
- **Micro-kernel constraints**: No external dependencies, no dynamic memory allocation, no global/static mutable state
- **License header**: All new files must include the SPDX header:

```c
/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
```

## Adding a New Micro-Kernel

Follow the existing naming conventions under `src/`:

- **Directory**: `<op>_<dst-type>_<src0-type>_<src1-type>/` (e.g., `gemm_f16_f16_f16/`)
- **File**: `tqt_<op>_<data-types>_<compute-params>_<extension>.cpp/.h`
- **Suffixes**: `t` = transposed, `p` = packed, `rvv` = RISC-V Vector

**Checklist:**

- [ ] Follow file and function naming conventions
- [ ] Add the SPDX license header
- [ ] Register source files in `CMakeLists.txt` under the appropriate `TORQ_TILE_*_SOURCES` list
- [ ] Add corresponding test cases
- [ ] Verify formatting with `clang-format-18`

## Testing

Tests use [Google Test](https://github.com/google/googletest). See [README](./README.md#build-and-run-tests) for build and run instructions. When writing tests:

- Place test files in the `test/` directory
- Cover correctness for various matrix sizes and edge cases
- Verify numerical accuracy against reference implementations

## Submitting Changes

- **Branch naming**: `feature/<desc>`, `fix/<desc>`, `docs/<desc>`, `refactor/<desc>`
- **Commit messages**: Use imperative mood (e.g., "Add FP16 RVV micro-kernel"), keep subject under 72 characters
- **MR guidelines**: Describe your changes, reference related issues, keep MRs focused on a single concern, ensure CI passes

## Reporting Issues

Please include: problem description, steps to reproduce, expected vs. actual behavior, environment info (OS, compiler, RISC-V CPU model, architecture extensions), and relevant logs.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](./LICENSE).
