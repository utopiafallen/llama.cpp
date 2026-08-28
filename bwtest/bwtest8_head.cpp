// Real-kernel DRAM probe: runs the PRODUCTION eSIMD DMMV kernel (copied verbatim
// from dmmv.cpp + esimd.hpp) against SoA and interleaved weight buffers.
// usage: bwtest8.exe [nrows] [ncols] [passes] [quant]  quant: 0=Q6_K 1=Q5_K 2=Q8_0
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <utility>

#define QK_K 256
#define QK8_0 32
#define K_SCALE_SIZE 12

enum ggml_type {
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
};

namespace ggml_sycl_esimd {

#ifndef GGML_SYCL_DMMV_ESIMD_WG_SIZE
constexpr int GGML_SYCL_DMMV_ESIMD_WG_SIZE = 4;
#endif

template <ggml_type T> struct esimd_reorder_q_traits;

// === verbatim from esimd.hpp (helpers) ===
