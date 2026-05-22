//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <benchmark/benchmark.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "benchmark/common/benchmark_common.h"
#include "benchmark/common/conv_benchmark_runner.h"

// ============================================================================
// Forward declarations: each bench_gemm_*.cpp provides a registration function
// Conditionally declared based on architecture feature flags
// ============================================================================

namespace torq_tile
{
namespace benchmark
{

#ifdef TORQ_TILE_ENABLE_RVV
void register_gemm_f32_f32_rvv_benchmarks();
void register_gemm_i8_i8_rvv_benchmarks();
void register_gemm_i8_i4_rvv_benchmarks();
void register_conv_f32_f32_rvv_benchmarks();
void register_conv_i8_i8_rvv_benchmarks();
void register_igemmcf_f32_f32_rvv_benchmarks();
void register_igemmcl_f32_f32_rvv_benchmarks();
void register_igemm_i8_i8_rvv_benchmarks();
void register_dwconv_f32_f32_rvv_benchmarks();
#endif

#ifdef TORQ_TILE_ENABLE_RVV_FP16
void register_gemm_f16_f16_rvv_benchmarks();
void register_gemm_f16_i4_rvv_benchmarks();
void register_conv_f16_f16_rvv_benchmarks();
void register_igemmcf_f16_f16_rvv_benchmarks();
void register_igemmcl_f16_f16_rvv_benchmarks();
void register_dwconv_f16_f16_rvv_benchmarks();
#endif

#ifdef TORQ_TILE_ENABLE_RVV_BF16
void register_gemm_bf16_bf16_rvv_benchmarks();
#endif

inline void register_gemm_benchmarks()
{
#ifdef TORQ_TILE_ENABLE_RVV
    register_gemm_f32_f32_rvv_benchmarks();
    register_gemm_i8_i8_rvv_benchmarks();
    register_gemm_i8_i4_rvv_benchmarks();
#endif
#ifdef TORQ_TILE_ENABLE_RVV_FP16
    register_gemm_f16_f16_rvv_benchmarks();
    register_gemm_f16_i4_rvv_benchmarks();
#endif
#ifdef TORQ_TILE_ENABLE_RVV_BF16
    register_gemm_bf16_bf16_rvv_benchmarks();
#endif
}

inline void register_conv_benchmarks()
{
#ifdef TORQ_TILE_ENABLE_RVV
    register_conv_f32_f32_rvv_benchmarks();
    register_conv_i8_i8_rvv_benchmarks();
    register_igemmcf_f32_f32_rvv_benchmarks();
    register_igemmcl_f32_f32_rvv_benchmarks();
    register_igemm_i8_i8_rvv_benchmarks();
    register_dwconv_f32_f32_rvv_benchmarks();
#endif
#ifdef TORQ_TILE_ENABLE_RVV_FP16
    register_conv_f16_f16_rvv_benchmarks();
    register_igemmcf_f16_f16_rvv_benchmarks();
    register_igemmcl_f16_f16_rvv_benchmarks();
    register_dwconv_f16_f16_rvv_benchmarks();
#endif
}

}  // namespace benchmark
}  // namespace torq_tile

// ============================================================================
// CLI argument parsing
// ============================================================================

static void print_usage(const char *program_name)
{
    std::fprintf(stderr,
                 "Usage: %s [-m M] [-n N] [-k K] [Google Benchmark flags...]\n"
                 "\n"
                 "Options:\n"
                 "  -m M    Set M dimension (default: 128)\n"
                 "  -n N    Set N dimension (default: 128)\n"
                 "  -k K    Set K dimension (default: 128)\n"
                 "\n"
                 "All other flags are passed to Google Benchmark.\n"
                 "Example:\n"
                 "  %s -m 256 -n 256 -k 512 --benchmark_filter=\"f32\"\n",
                 program_name, program_name);
}

static void parse_custom_args(int &argc, char **argv)
{
    auto &shape = torq_tile::benchmark::global_gemm_shape();

    // Collect indices of consumed arguments for removal
    std::vector<int> consumed_indices;

    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            shape.m = static_cast<size_t>(std::atol(argv[i + 1]));
            consumed_indices.push_back(i);
            consumed_indices.push_back(i + 1);
            ++i;
        } else if ((std::strcmp(argv[i], "-n") == 0) && i + 1 < argc) {
            shape.n = static_cast<size_t>(std::atol(argv[i + 1]));
            consumed_indices.push_back(i);
            consumed_indices.push_back(i + 1);
            ++i;
        } else if ((std::strcmp(argv[i], "-k") == 0) && i + 1 < argc) {
            shape.k = static_cast<size_t>(std::atol(argv[i + 1]));
            consumed_indices.push_back(i);
            consumed_indices.push_back(i + 1);
            ++i;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            // Don't consume --help, let Google Benchmark also handle it
        }
    }

    // Remove consumed arguments by shifting remaining ones
    if (!consumed_indices.empty()) {
        int write_idx = 1;
        for (int read_idx = 1; read_idx < argc; ++read_idx) {
            bool is_consumed = false;
            for (int ci : consumed_indices) {
                if (read_idx == ci) {
                    is_consumed = true;
                    break;
                }
            }
            if (!is_consumed) {
                argv[write_idx++] = argv[read_idx];
            }
        }
        argc = write_idx;
    }
}

// ============================================================================
// Main entry point
// ============================================================================

int main(int argc, char **argv)
{
    // Parse custom -m/-n/-k arguments before Google Benchmark initialization
    parse_custom_args(argc, argv);

    const auto &shape = torq_tile::benchmark::global_gemm_shape();
    const auto &conv_shape = torq_tile::benchmark::global_conv_shape();
    std::fprintf(stdout, "TORQ-Tile Benchmark\n");
    std::fprintf(stdout, "  GEMM shape: M=%zu, N=%zu, K=%zu\n", shape.m, shape.n, shape.k);
    std::fprintf(stdout,
                 "  Conv shape: IC=%zu, OC=%zu, IH=%zu, IW=%zu, KH=%zu, KW=%zu, "
                 "stride=%zu, pad=%zu\n",
                 conv_shape.ic, conv_shape.oc, conv_shape.ih, conv_shape.iw, conv_shape.kh,
                 conv_shape.kw, conv_shape.stride_h, conv_shape.pad_h);
    std::fprintf(stdout, "\n");

    // Register all benchmarks
    torq_tile::benchmark::register_gemm_benchmarks();
    torq_tile::benchmark::register_conv_benchmarks();

    // Initialize and run Google Benchmark
    ::benchmark::Initialize(&argc, argv);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}
