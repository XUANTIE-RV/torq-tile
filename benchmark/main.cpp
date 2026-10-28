//
// SPDX-FileCopyrightText: Copyright 2024-2026 C-SKY Microsystems Co., Ltd.
//
// SPDX-License-Identifier: Apache-2.0
//

#include <benchmark/benchmark.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "benchmark/common/benchmark_common.h"
#include "benchmark/common/conv_benchmark_runner.h"
#include "torq_tile/tqt_compile_config.h"

// ============================================================================
// Forward declarations: each bench_*.cpp provides a registration function
// Conditionally declared based on architecture feature flags
// ============================================================================

namespace torq_tile
{
namespace benchmark
{

#if TORQ_TILE_HAS_ZTT_GEMM_BLOCK_INT_U4
void register_gemm_i8_u4_ztt_benchmarks();
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_BLOCK_INT_I8
void register_gemm_i8_i8_ztt_benchmarks();
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_F32
void register_gemm_f32_f32_ztt_benchmarks();
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_BF16
void register_gemm_bf16_bf16_ztt_benchmarks();
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_F16
void register_gemm_f16_f16_ztt_benchmarks();
#endif

#if TORQ_TILE_HAS_RVV
void register_gemm_f32_f32_rvv_benchmarks();
void register_gemm_i8_i8_rvv_benchmarks();
void register_gemm_i8_i4_rvv_benchmarks();
void register_gemm_i8_b1_rvv_benchmarks();
void register_gemm_i8_i5_rvv_benchmarks();
void register_gemm_i8_i3_rvv_benchmarks();
void register_gemm_i8_u2_rvv_benchmarks();
void register_gemm_i8_u4_rvv_benchmarks();
void register_gemm_i8_u5_rvv_benchmarks();
void register_gemm_i8_u8_rvv_benchmarks();
void register_gemm_i8_i6_rvv_benchmarks();
void register_gemm_i8_i2_rvv_benchmarks();
void register_gemm_u8_u8_rvv_benchmarks();
void register_conv_f32_f32_rvv_benchmarks();
void register_conv_i8_i8_rvv_benchmarks();
void register_igemmcf_f32_f32_rvv_benchmarks();
void register_igemmcl_f32_f32_rvv_benchmarks();
void register_igemm_i8_i8_rvv_benchmarks();
void register_dwconv_f32_f32_rvv_benchmarks();
void register_add_f32_f32_rvv_benchmarks();
void register_sub_f32_f32_rvv_benchmarks();
void register_mul_f32_f32_rvv_benchmarks();
void register_div_f32_f32_rvv_benchmarks();
void register_sum_f32_f32_rvv_benchmarks();
void register_mean_f32_f32_rvv_benchmarks();
void register_relu_f32_f32_rvv_benchmarks();
void register_gelu_f32_f32_rvv_benchmarks();
void register_silu_f32_f32_rvv_benchmarks();
void register_reglu_f32_f32_f32_rvv_benchmarks();
void register_geglu_f32_f32_f32_rvv_benchmarks();
void register_swiglu_f32_f32_f32_rvv_benchmarks();
#endif

#if TORQ_TILE_HAS_RVV_F16
void register_gemm_f32_f16_rvv_benchmarks();
void register_gemm_f16_f16_rvv_benchmarks();
void register_gemm_f16_i4_rvv_benchmarks();
void register_conv_f16_f16_rvv_benchmarks();
void register_igemmcf_f16_f16_rvv_benchmarks();
void register_igemmcl_f16_f16_rvv_benchmarks();
void register_dwconv_f16_f16_rvv_benchmarks();
void register_add_f16_f16_rvv_benchmarks();
void register_sub_f16_f16_rvv_benchmarks();
void register_mul_f16_f16_rvv_benchmarks();
void register_div_f16_f16_rvv_benchmarks();
void register_sum_f16_f16_rvv_benchmarks();
void register_mean_f16_f16_rvv_benchmarks();
void register_relu_f16_f16_rvv_benchmarks();
void register_gelu_f16_f16_rvv_benchmarks();
void register_silu_f16_f16_rvv_benchmarks();
void register_reglu_f16_f16_f16_rvv_benchmarks();
void register_geglu_f16_f16_f16_rvv_benchmarks();
void register_swiglu_f16_f16_f16_rvv_benchmarks();
#endif

#if TORQ_TILE_HAS_RVV_BF16
void register_gemm_f32_bf16_rvv_benchmarks();
void register_gemm_bf16_bf16_rvv_benchmarks();
void register_conv_bf16_bf16_rvv_benchmarks();
#endif

#if TORQ_TILE_HAS_RVV_F64
void register_gemm_f64_f64_rvv_benchmarks();
#endif

inline void register_gemm_benchmarks()
{
#if TORQ_TILE_HAS_ZTT_GEMM_BLOCK_INT_U4
    register_gemm_i8_u4_ztt_benchmarks();
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_BLOCK_INT_I8
    register_gemm_i8_i8_ztt_benchmarks();
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_F32
    register_gemm_f32_f32_ztt_benchmarks();
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_BF16
    register_gemm_bf16_bf16_ztt_benchmarks();
#endif
#if TORQ_TILE_HAS_ZTT_GEMM_F16
    register_gemm_f16_f16_ztt_benchmarks();
#endif

#if TORQ_TILE_HAS_RVV
    register_gemm_f32_f32_rvv_benchmarks();
    register_gemm_i8_i8_rvv_benchmarks();
    register_gemm_i8_i4_rvv_benchmarks();
    register_gemm_i8_b1_rvv_benchmarks();
    register_gemm_i8_i5_rvv_benchmarks();
    register_gemm_i8_i3_rvv_benchmarks();
    register_gemm_i8_u2_rvv_benchmarks();
    register_gemm_i8_u4_rvv_benchmarks();
    register_gemm_i8_u5_rvv_benchmarks();
    register_gemm_i8_u8_rvv_benchmarks();
    register_gemm_i8_i6_rvv_benchmarks();
    register_gemm_i8_i2_rvv_benchmarks();
    register_gemm_u8_u8_rvv_benchmarks();
#endif
#if TORQ_TILE_HAS_RVV_F16
    register_gemm_f32_f16_rvv_benchmarks();
    register_gemm_f16_f16_rvv_benchmarks();
    register_gemm_f16_i4_rvv_benchmarks();
#endif
#if TORQ_TILE_HAS_RVV_BF16
    register_gemm_f32_bf16_rvv_benchmarks();
    register_gemm_bf16_bf16_rvv_benchmarks();
#endif
#if TORQ_TILE_HAS_RVV_F64
    register_gemm_f64_f64_rvv_benchmarks();
#endif
}

inline void register_conv_benchmarks()
{
#if TORQ_TILE_HAS_RVV
    register_conv_f32_f32_rvv_benchmarks();
    register_conv_i8_i8_rvv_benchmarks();
    register_igemmcf_f32_f32_rvv_benchmarks();
    register_igemmcl_f32_f32_rvv_benchmarks();
    register_igemm_i8_i8_rvv_benchmarks();
    register_dwconv_f32_f32_rvv_benchmarks();
#endif
#if TORQ_TILE_HAS_RVV_F16
    register_conv_f16_f16_rvv_benchmarks();
    register_igemmcf_f16_f16_rvv_benchmarks();
    register_igemmcl_f16_f16_rvv_benchmarks();
    register_dwconv_f16_f16_rvv_benchmarks();
#endif
#if TORQ_TILE_HAS_RVV_BF16
    register_conv_bf16_bf16_rvv_benchmarks();
#endif
}

inline void register_elementwise_benchmarks()
{
#if TORQ_TILE_HAS_RVV
    register_add_f32_f32_rvv_benchmarks();
    register_sub_f32_f32_rvv_benchmarks();
    register_mul_f32_f32_rvv_benchmarks();
    register_div_f32_f32_rvv_benchmarks();
#endif
#if TORQ_TILE_HAS_RVV_F16
    register_add_f16_f16_rvv_benchmarks();
    register_sub_f16_f16_rvv_benchmarks();
    register_mul_f16_f16_rvv_benchmarks();
    register_div_f16_f16_rvv_benchmarks();
#endif
}

inline void register_reduction_benchmarks()
{
#if TORQ_TILE_HAS_RVV
    register_sum_f32_f32_rvv_benchmarks();
    register_mean_f32_f32_rvv_benchmarks();
#endif
#if TORQ_TILE_HAS_RVV_F16
    register_sum_f16_f16_rvv_benchmarks();
    register_mean_f16_f16_rvv_benchmarks();
#endif
}

inline void register_activation_benchmarks()
{
#if TORQ_TILE_HAS_RVV
    register_relu_f32_f32_rvv_benchmarks();
    register_gelu_f32_f32_rvv_benchmarks();
    register_silu_f32_f32_rvv_benchmarks();
    register_reglu_f32_f32_f32_rvv_benchmarks();
    register_geglu_f32_f32_f32_rvv_benchmarks();
    register_swiglu_f32_f32_f32_rvv_benchmarks();
#endif
#if TORQ_TILE_HAS_RVV_F16
    register_relu_f16_f16_rvv_benchmarks();
    register_gelu_f16_f16_rvv_benchmarks();
    register_silu_f16_f16_rvv_benchmarks();
    register_reglu_f16_f16_f16_rvv_benchmarks();
    register_geglu_f16_f16_f16_rvv_benchmarks();
    register_swiglu_f16_f16_f16_rvv_benchmarks();
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
                 "M/N/K are shared by the GEMM, activation, reduction and elementwise\n"
                 "benchmarks, and each op reads only the dimensions it needs: GEMM uses\n"
                 "M/N/K, activation and reduction use M/N, elementwise uses N. The conv and\n"
                 "dwconv benchmarks ignore M/N/K and use their own conv shape.\n"
                 "  --conv-ic/ih/iw/kh/kw VALUE  Override one conv dimension.\n"
                 "  --conv-pad-h/w VALUE        Override one padding offset.\n"
                 "1D dwconv: --conv-ih 1 --conv-kh 1 --conv-kw 4 --conv-pad-h 0\n"
                 "\n"
                 "All other flags are passed to Google Benchmark.\n"
                 "Example:\n"
                 "  %s -m 256 -n 256 -k 512 --benchmark_filter=\"f32\"\n",
                 program_name, program_name);
}

static void parse_custom_args(int &argc, char **argv)
{
    auto &shape = torq_tile::benchmark::global_gemm_shape();
    auto &conv = torq_tile::benchmark::global_conv_shape();

    // Collect indices of consumed arguments for removal
    std::vector<int> consumed_indices;

    for (int i = 1; i < argc; ++i) {
        size_t *dimension = nullptr;
        bool padding = false;
        if (std::strcmp(argv[i], "--conv-ic") == 0)
            dimension = &conv.ic;
        if (std::strcmp(argv[i], "--conv-ih") == 0)
            dimension = &conv.ih;
        if (std::strcmp(argv[i], "--conv-iw") == 0)
            dimension = &conv.iw;
        if (std::strcmp(argv[i], "--conv-kh") == 0)
            dimension = &conv.kh;
        if (std::strcmp(argv[i], "--conv-kw") == 0)
            dimension = &conv.kw;
        if (std::strcmp(argv[i], "--conv-pad-h") == 0) {
            dimension = &conv.pad_h;
            padding = true;
        }
        if (std::strcmp(argv[i], "--conv-pad-w") == 0) {
            dimension = &conv.pad_w;
            padding = true;
        }
        if (dimension) {
            char *end = nullptr;
            errno = 0;
            const unsigned long long value =
                i + 1 < argc ? std::strtoull(argv[i + 1], &end, 10) : 0;
            if (i + 1 == argc || argv[i + 1][0] == '-' || !end || end == argv[i + 1] || *end ||
                errno || (!padding && value == 0) || value > std::numeric_limits<size_t>::max()) {
                std::fprintf(stderr, "Invalid value for %s\n", argv[i]);
                std::exit(1);
            }
            *dimension = size_t(value);
            consumed_indices.push_back(i);
            consumed_indices.push_back(++i);
        } else if ((std::strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
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

    const size_t max_extent = std::numeric_limits<size_t>::max();
    if (conv.pad_h > (max_extent - conv.ih) / 2 || conv.pad_w > (max_extent - conv.iw) / 2 ||
        conv.ih + 2 * conv.pad_h < conv.kh || conv.iw + 2 * conv.pad_w < conv.kw) {
        std::fprintf(stderr, "Invalid padded conv extent\n");
        std::exit(1);
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
    // Parse custom dimensions before Google Benchmark initialization
    parse_custom_args(argc, argv);

    const auto &shape = torq_tile::benchmark::global_gemm_shape();
    const auto &conv_shape = torq_tile::benchmark::global_conv_shape();
    std::fprintf(stdout, "TORQ-Tile Benchmark\n");
    std::fprintf(stdout, "  Shared M/N/K shape: M=%zu, N=%zu, K=%zu\n", shape.m, shape.n, shape.k);
    std::fprintf(stdout,
                 "    (GEMM uses M/N/K; activation and reduction use M/N; elementwise uses N)\n");
    std::fprintf(stdout,
                 "  Conv shape: IC=%zu, OC=%zu, IH=%zu, IW=%zu, KH=%zu, KW=%zu, "
                 "stride_h=%zu, stride_w=%zu, pad_h=%zu, pad_w=%zu\n",
                 conv_shape.ic, conv_shape.oc, conv_shape.ih, conv_shape.iw, conv_shape.kh,
                 conv_shape.kw, conv_shape.stride_h, conv_shape.stride_w, conv_shape.pad_h,
                 conv_shape.pad_w);
    std::fprintf(stdout, "\n");

    // Register all benchmarks
    torq_tile::benchmark::register_gemm_benchmarks();
    torq_tile::benchmark::register_conv_benchmarks();
    torq_tile::benchmark::register_elementwise_benchmarks();
    torq_tile::benchmark::register_reduction_benchmarks();
    torq_tile::benchmark::register_activation_benchmarks();

    // Initialize and run Google Benchmark
    ::benchmark::Initialize(&argc, argv);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}
