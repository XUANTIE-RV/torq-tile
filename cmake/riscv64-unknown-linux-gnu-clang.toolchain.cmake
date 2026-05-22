#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

# ============================================================================
# Clang Cross-Compilation Toolchain for RISC-V 64-bit Linux
# ============================================================================
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-unknown-linux-gnu-clang.toolchain.cmake \
#         -DCMAKE_C_FLAGS="<arch-flags>" \
#         -DCMAKE_CXX_FLAGS="<arch-flags>" \
#         -S . -B build/
#
# Optional variables (set before or via -D):
#   RISCV_GCC_INSTALL_DIR  - Path to the GCC toolchain installation
#                            (auto-detected from riscv64-unknown-linux-gnu-gcc if not set)
#   RISCV_SYSROOT          - Path to the target sysroot
#                            (auto-detected from GCC toolchain if not set)
#

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_CROSSCOMPILING   TRUE)

set(RISCV_TARGET_TRIPLE riscv64-unknown-linux-gnu)

# Use clang/clang++ as the compiler
set(CMAKE_C_COMPILER   clang)
set(CMAKE_CXX_COMPILER clang++)

# Set the target triple for cross-compilation
set(CMAKE_C_COMPILER_TARGET   ${RISCV_TARGET_TRIPLE})
set(CMAKE_CXX_COMPILER_TARGET ${RISCV_TARGET_TRIPLE})

# Auto-detect GCC toolchain path if not provided
if(NOT RISCV_GCC_INSTALL_DIR)
    execute_process(
        COMMAND which ${RISCV_TARGET_TRIPLE}-gcc
        OUTPUT_VARIABLE _gcc_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_gcc_path)
        # Resolve symlinks and get the real path
        get_filename_component(_gcc_realpath "${_gcc_path}" REALPATH)
        # Go up two levels: bin/riscv64-unknown-linux-gnu-gcc -> install_dir
        get_filename_component(RISCV_GCC_INSTALL_DIR "${_gcc_realpath}" DIRECTORY)
        get_filename_component(RISCV_GCC_INSTALL_DIR "${RISCV_GCC_INSTALL_DIR}" DIRECTORY)
        message(STATUS "Auto-detected GCC install dir: ${RISCV_GCC_INSTALL_DIR}")
    endif()
endif()

# Auto-detect sysroot if not provided
if(NOT RISCV_SYSROOT)
    if(RISCV_GCC_INSTALL_DIR AND EXISTS "${RISCV_GCC_INSTALL_DIR}/sysroot")
        set(RISCV_SYSROOT "${RISCV_GCC_INSTALL_DIR}/sysroot")
    elseif(RISCV_GCC_INSTALL_DIR AND EXISTS "${RISCV_GCC_INSTALL_DIR}/${RISCV_TARGET_TRIPLE}/sysroot")
        set(RISCV_SYSROOT "${RISCV_GCC_INSTALL_DIR}/${RISCV_TARGET_TRIPLE}/sysroot")
    endif()
endif()

# Apply --gcc-toolchain and --sysroot flags
if(RISCV_GCC_INSTALL_DIR)
    set(CMAKE_C_FLAGS_INIT   "--gcc-toolchain=${RISCV_GCC_INSTALL_DIR}")
    set(CMAKE_CXX_FLAGS_INIT "--gcc-toolchain=${RISCV_GCC_INSTALL_DIR}")
endif()

if(RISCV_SYSROOT)
    set(CMAKE_SYSROOT ${RISCV_SYSROOT})
    message(STATUS "Using sysroot: ${RISCV_SYSROOT}")
endif()

# Use GNU linker from the GCC toolchain (lld may not support all RISC-V relocations)
set(CMAKE_EXE_LINKER_FLAGS_INIT    "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld")

# Use LLVM tools for archiving and other utilities
find_program(CMAKE_AR      llvm-ar)
find_program(CMAKE_RANLIB  llvm-ranlib)
find_program(CMAKE_STRIP   llvm-strip)
find_program(CMAKE_OBJDUMP llvm-objdump)
find_program(CMAKE_NM      llvm-nm)
