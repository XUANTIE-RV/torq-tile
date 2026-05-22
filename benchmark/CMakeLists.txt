#
# SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
#

# ============================================================================
# Benchmark Sources Collection
# ============================================================================
set(TORQ_TILE_BENCHMARK_SOURCES
    main.cpp
)

# RVV benchmarks (auto-detected via TORQ_TILE_ENABLE_RVV)
if(TORQ_TILE_ENABLE_RVV)
    list(APPEND TORQ_TILE_BENCHMARK_SOURCES
        gemm/bench_gemm_f32_f32_rvv.cpp
        gemm/bench_gemm_i8_i8_rvv.cpp
        gemm/bench_gemm_i8_i4_rvv.cpp
        conv/bench_conv_f32_f32_rvv.cpp
        conv/bench_conv_i8_i8_rvv.cpp
        conv/bench_igemm_f32_f32_rvv.cpp
        conv/bench_igemm_i8_i8_rvv.cpp
        dwconv/bench_dwconv_f32_f32_rvv.cpp
    )
endif()

# RVV FP16 benchmarks (auto-detected via TORQ_TILE_ENABLE_RVV_FP16)
if(TORQ_TILE_ENABLE_RVV_FP16)
    list(APPEND TORQ_TILE_BENCHMARK_SOURCES
        gemm/bench_gemm_f16_f16_rvv.cpp
        gemm/bench_gemm_f16_i4_rvv.cpp
        conv/bench_conv_f16_f16_rvv.cpp
        conv/bench_igemm_f16_f16_rvv.cpp
        dwconv/bench_dwconv_f16_f16_rvv.cpp
    )
endif()

# RVV BF16 benchmarks (auto-detected via TORQ_TILE_ENABLE_RVV_BF16)
if(TORQ_TILE_ENABLE_RVV_BF16)
    list(APPEND TORQ_TILE_BENCHMARK_SOURCES
        gemm/bench_gemm_bf16_bf16_rvv.cpp
    )
endif()

# ============================================================================
# Benchmark Executable
# ============================================================================
add_executable(torq_tile_benchmark ${TORQ_TILE_BENCHMARK_SOURCES})

target_include_directories(torq_tile_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(torq_tile_benchmark PRIVATE
    torq_tile
    benchmark::benchmark
)

# Pass architecture feature flags as compile definitions
if(TORQ_TILE_ENABLE_RVV)
    target_compile_definitions(torq_tile_benchmark PRIVATE TORQ_TILE_ENABLE_RVV)
endif()
if(TORQ_TILE_ENABLE_RVV_FP16)
    target_compile_definitions(torq_tile_benchmark PRIVATE TORQ_TILE_ENABLE_RVV_FP16)
endif()
if(TORQ_TILE_ENABLE_RVV_BF16)
    target_compile_definitions(torq_tile_benchmark PRIVATE TORQ_TILE_ENABLE_RVV_BF16)
endif()

# Static library needs --whole-archive to ensure all symbols are visible
if(NOT TORQ_TILE_BUILD_SHARED)
    target_link_options(torq_tile_benchmark PRIVATE
        -Wl,--whole-archive
        $<TARGET_FILE:torq_tile>
        -Wl,--no-whole-archive
    )
endif()
