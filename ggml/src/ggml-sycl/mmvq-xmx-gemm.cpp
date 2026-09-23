//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

// Small-M GEMM for the reordered MMVQ path (M in 2..15), computed on the XMX
// matrix engine. The MMVQ dp4a loop is issue-bound ~38% above the DRAM floor at
// these shapes (see b70-decode-perf skill); this kernel streams the same weight
// bytes once, dequants k-chunks to f16 LDS and accumulates with
// joint_matrix. Tile geometry is fattn-xmx-decode's proven shape on this device:
// A [16][16] from global f16 (L2-resident, FA-proven pattern), B [16][CK] staged
// feature-major in LDS (ldm CK), C [16][16] f32, 16-wide k steps. Activations
// are padded to 16 rows once per op in an f16 scratch, so no Q8_1 requant
// happens here.
//
// Chunk width CK (GGML_SYCL_XMXG_CK, 64/128/256) trades barrier frequency and
// LDS against in-flight bytes; total per-WG LDS is kept at or under the ~8KB
// occupancy cliff (CK=256 reuses the dead B tile as C staging).
//
// Note: this device rejects A tiles with Cols=64 (Rows 8 and 16); only the
// 16-wide XMX steps work.
//
// Gate: GGML_SYCL_XMX_GEMM (default 0), reordered tensor, single device,
// contiguous f32 src1/dst, K % CK == 0, N % 16 == 0.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "common.hpp"
#include "ggml-backend-impl.h"
#include "mmvq-xmx-gemm.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <type_traits>

namespace mx = sycl::ext::oneapi::experimental::matrix;
using mx::use;
using mx::layout;
namespace syclex = sycl::ext::oneapi::experimental;

using xmp_g_h = sycl::multi_ptr<sycl::half, sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
using xmp_l_h = sycl::multi_ptr<sycl::half, sycl::access::address_space::local_space,  sycl::access::decorated::legacy>;
using xmp_l_f = sycl::multi_ptr<float,  sycl::access::address_space::local_space,  sycl::access::decorated::legacy>;

// Off by default; the A/B at 152K decides whether it ships.
static int g_ggml_sycl_xmx_gemm() {
    static int v = ggml_sycl_get_env("GGML_SYCL_XMX_GEMM", 0);
    return v;
}

// One-shot debug dump on the first op of each enabled type.
static int g_ggml_sycl_xmx_dbg() {
    static int v = ggml_sycl_get_env("GGML_SYCL_XMX_DBG", 0);
    return v;
}

// Per-op wall timing (serializes the stream; debug only).
static int g_ggml_sycl_xmx_gemm_timing() {
    static int v = ggml_sycl_get_env("GGML_SYCL_XMXG_TIMING", 0);
    return v;
}

// Split the K reduction across work-groups (f32 partials + combine pass).
// 0 = auto (8 for K >= 16K, 4 for K >= 4K, else 1); explicit 1/2/4/8.
static int g_ggml_sycl_xmxg_split() {
    static int v = ggml_sycl_get_env("GGML_SYCL_XMXG_SPLIT", 0);
    return v;
}

// K values per dequant/stage chunk. 64 (original) / 128 / 256: wider chunks
// halve the barrier pairs and raise in-flight bytes per stage, at the cost of
// LDS (CK=256 sits exactly on the ~8KB cliff with C staging overlapped).
static int g_ggml_sycl_xmxg_ck() {
    static int v = ggml_sycl_get_env("GGML_SYCL_XMXG_CK", 64);
    return v;
}

// Cost-component bisection knobs (debug only): NODQ writes a constant B tile,
// NOMAD skips the loads/mads (keeps barriers + dequant staging).
static int g_ggml_sycl_xmxg_nodq() {
    static int v = ggml_sycl_get_env("GGML_SYCL_XMXG_NODQ", 0);
    return v;
}
static int g_ggml_sycl_xmxg_nomad() {
    static int v = ggml_sycl_get_env("GGML_SYCL_XMXG_NOMAD", 0);
    return v;
}

// Per-type enable mask for bisection, e.g. GGML_SYCL_XMXG_TYPES=6 runs only
// q6_K through XMX (8=q8_0, 6=q6_k, 5=q5_k); default all three.
static const int * xmxg_type_mask() {
    static int mask[3] = { 1, 1, 1 };
    static bool done = false;
    if (!done) {
        done = true;
        const char * s = getenv("GGML_SYCL_XMXG_TYPES");
        if (s && *s) {
            mask[0] = mask[1] = mask[2] = 0;
            for (const char * p = s; *p; p++) {
                if (*p == '8') mask[0] = 1;
                if (*p == '6') mask[1] = 1;
                if (*p == '5') mask[2] = 1;
            }
        }
    }
    return mask;
}

enum { XMXG_Q8_0 = 0, XMXG_Q6_K = 1, XMXG_Q5_K = 2 };

// XMX A rows; M=2..15 pads to 16 (device-proven shape; 8-row tiles unsupported).
constexpr int XMXG_M = 16;
// K step per XMX instruction (16-wide is the only supported N on this device)
constexpr int XMXG_KB = 16;

// Dequant 16 consecutive weight values (k0 % 16 == 0) of output row `row`
// into half[16]. The reorder functions split each type's block planes over the
// whole tensor (all ql, then all qh, then scales, then d); block index
// bi = row * (K / qk) + col / qk in row-major order, and the in-block byte map
// is the standard ggml layout (verified against the CPU generic dot kernels).
template <int WTYPE>
static inline void xmxg_dequant_row(const char * __restrict__ vx, const int N, const int K,
                                    const int row, const int k0, sycl::half * __restrict__ out16) {
    if constexpr (WTYPE == XMXG_Q8_0) {
        // [qs: N*K bytes][d: N*K/32 halves]
        const size_t nb = (size_t) N * K / 32;
        const int bi    = row * (K / 32) + k0 / 32;
        const int8_t * qs = (const int8_t *) vx + bi*32 + k0%32;
        const float d = (float) *(const sycl::half *)(vx + nb*32 + (size_t) bi*2);
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            out16[k] = (sycl::half) ((float) qs[k] * d);
        }
    } else if constexpr (WTYPE == XMXG_Q6_K) {
        // [ql: nb*128][qh: nb*64][scales: nb*16][d: nb*2]; 256 values per block,
        // two 128-value sub-halves; within a sub-half value w = 32g + t uses qh byte
        // t (2 bits at 2*g) and ql byte (g odd ? t + 32 : t), low nibble for g < 2.
        const size_t nb  = (size_t) N * K / 256;
        const int bi     = row * (K / 256) + k0 / 256;
        const int w0     = k0 % 256;
        const size_t qh_base  = (size_t) nb*128;
        const size_t sc_base  = qh_base + (size_t) nb*64;
        const size_t d_base   = sc_base + (size_t) nb*16;
        const int8_t s_b = *(const int8_t *)(vx + sc_base + (size_t) bi*16 + w0/16);
        const float sf = (float) s_b;
        const float df = (float) *(const sycl::half *)(vx + d_base + (size_t) bi*2);
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            const int w = w0 + k;
            const int h = w / 128, t = (w % 128) % 32, g = (w % 128) / 32;
            const int ql_b = *(const uint8_t *)(vx + (size_t) bi*128 + h*64 + ((g & 1) ? t + 32 : t));
            const int qh_b = *(const uint8_t *)(vx + qh_base + (size_t) bi*64 + h*32 + t);
            const int nib  = (g < 2) ? (ql_b & 0xF) : (ql_b >> 4);
            const int qh2  = (qh_b >> (2*g)) & 3;
            const int v6   = (int8_t) ((nib | (qh2 << 4)) - 32);
            out16[k] = (sycl::half) ((float) v6 * sf * df);
        }
    } else { // XMXG_Q5_K
        // [qs: nb*128][qh: nb*32][scales: nb*12][dm: nb*4]; 256 values per block
        // in four 64-value quarters q = w/64; value t + 32*upper in a quarter uses
        // qs byte t (nibble by upper) and one qh bit (2*q + upper). The 12 scale
        // bytes pack 8 scales + 8 mins of 6 bits each (decode per CPU dot kernel).
        const size_t nb  = (size_t) N * K / 256;
        const int bi     = row * (K / 256) + k0 / 256;
        const int w0     = k0 % 256;
        const size_t qh_base = (size_t) nb*128;
        const size_t sc_base = qh_base + (size_t) nb*32;
        // 12-byte scale word is not u32-aligned for odd bi: assemble from bytes
        const uint8_t * sc_b = (const uint8_t *)(vx + sc_base + (size_t) bi*12);
        const auto bu32 = [](const uint8_t * p) {
            return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
        };
        uint32_t ut0 = bu32(sc_b), ut1 = bu32(sc_b + 4), ut2 = bu32(sc_b + 8);
        const uint32_t kmask1 = 0x3F3F3F3Fu, kmask2 = 0x0F0F0F0Fu, kmask3 = 0x03030303u;
        // packed decode (per the CPU dot kernel): dec[0..1] = scales 0..7, dec[2..3] = mins 0..7
        uint32_t dec[4];
        {
            uint32_t ut3 = ((ut2 >> 4) & kmask2) | (((ut1 >> 6) & kmask3) << 4);
            const uint32_t uaux  = ut1 & kmask1;
            dec[0] = ut0 & kmask1;
            dec[1] = (ut2 & kmask2) | (((ut0 >> 6) & kmask3) << 4);
            dec[2] = uaux;
            dec[3] = ut3;
        }
        const uint8_t * scales6 = (const uint8_t *) dec;       // [8]
        const uint8_t * mins6   = (const uint8_t *) (dec + 2); // [8]
        // dm plane holds {d, dmin} per block (canonical half2 order)
        const sycl::half * dm2   = (const sycl::half *)(vx + sc_base + (size_t) nb*12 + (size_t) bi*4);
        const float df    = (float) dm2[0];
        const float dminf = (float) dm2[1];
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            const int w = w0 + k;
            const int q = w / 64, t = (w % 64) % 32, upper = (w % 64) / 32;
            const int g32 = w / 32;
            const uint8_t qs_b = *(const uint8_t *)(vx + (size_t) bi*128 + q*32 + t);
            // canonical q5_K: the qh pointer does not advance across chunks;
            // the chunk only moves the bit position (2*q + upper) within byte t
            const uint8_t qh_b = *(const uint8_t *)(vx + qh_base + (size_t) bi*32 + t);
            const int nib5 = upper ? (qs_b >> 4) : (qs_b & 0xF);
            const int bit  = (qh_b >> (2*q + upper)) & 1;
            const int v5   = nib5 + (bit ? 16 : 0);
            // per the CPU dequant: d*sc*v5 - dmin*m (dmin is a block constant)
            out16[k] = (sycl::half) ((float) v5 * (float) scales6[g32] * df - (float) mins6[g32] * dminf);
        }
    }
}

// Dequant one CK-wide k chunk (k0 % CK == 0, CK in {64,128,256}) of output row
// `row` into half[CK]. The chunk spans at most one 256-value block (or CK/32
// q8_0 blocks) since CK | 256; plane bases, the scale word and d are hoisted
// out of the subtile loop. Byte math is identical to xmxg_dequant_row
// (CPU-verified).
template <int WTYPE, int CK>
static inline void xmxg_dequant_chunk(const char * __restrict__ vx, const int N, const int K,
                                      const int row, const int k0, sycl::half * __restrict__ out) {
    if constexpr (WTYPE == XMXG_Q8_0) {
        constexpr int NS = CK / 32; // 32-value blocks in the chunk
        const size_t nb = (size_t) N * K / 32;
        const int bi0   = row * (K / 32) + k0 / 32;
        float df[NS];
        #pragma unroll
        for (int j = 0; j < NS; j++) {
            df[j] = (float) *(const sycl::half *)(vx + nb*32 + (size_t) (bi0 + j)*2);
        }
        const uint32_t * qp = (const uint32_t *) (vx + (size_t) bi0*32); // CK bytes
        #pragma unroll
        for (int s = 0; s < CK/16; s++) {
            const float d = df[s >> 1];
            uint32_t qw[4];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                qw[b] = qp[s*4 + b];
            }
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                const uint32_t by = (qw[k >> 2] >> ((k & 3)*8)) & 0xFF;
                out[s*16 + k] = (sycl::half) ((float) (int8_t) by * d);
            }
        }
    } else if constexpr (WTYPE == XMXG_Q6_K) {
        // [ql: nb*128][qh: nb*64][scales: nb*16][d: nb*2]; 256 values per block,
        // two 128-value sub-halves; within a sub-half value w = 32g + t uses qh byte
        // t (2 bits at 2*g) and ql byte (g odd ? t + 32 : t), low nibble for g < 2.
        const size_t nb  = (size_t) N * K / 256;
        const int bi     = row * (K / 256) + k0 / 256;
        const int w0c    = k0 % 256; // 0 or 128 for CK >= 128, any 64-multiple for 64
        const size_t qh_base = (size_t) nb*128;
        const size_t sc_base = qh_base + (size_t) nb*64;
        // 16B scale word + block d hoisted; the chunk touches bytes
        // w0c/16 .. w0c/16 + CK/16 - 1 of it, always inside one aligned word
        uint32_t sw[4];
        const uint32_t * sp = (const uint32_t *) (vx + sc_base + (size_t) bi*16);
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            sw[s] = sp[s];
        }
        const float df = (float) *(const sycl::half *)(vx + sc_base + (size_t) nb*16 + (size_t) bi*2);
        const uint8_t * ql_c = (const uint8_t *) (vx + (size_t) bi*128);
        const uint8_t * qh_c = (const uint8_t *) (vx + qh_base + (size_t) bi*64);
        #pragma unroll
        for (int s = 0; s < CK/16; s++) {
            const int w0   = w0c + s*16;
            const int g    = (w0 % 128) / 32;
            const int si   = w0 / 16; // scale byte within the hoisted word, 0..15
            const uint8_t sb = (uint8_t) (sw[si >> 2] >> ((si & 3)*8));
            const float sf = (float) (int8_t) sb * df;
            const uint8_t * qp = ql_c + (w0 / 128)*64 + (g & 1)*32 + (w0 % 32);
            const uint8_t * hp = qh_c + (w0 / 128)*32 + (w0 % 32);
            uint32_t qw[4], hw[4];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                qw[b] = *(const uint32_t *)(qp + b*4);
                hw[b] = *(const uint32_t *)(hp + b*4);
            }
            const int nibsh = (g < 2) ? 0 : 4;
            const int qhsh  = 2*g;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                const uint32_t ql_b = (qw[k >> 2] >> ((k & 3)*8)) & 0xFF;
                const uint32_t qh_b = (hw[k >> 2] >> ((k & 3)*8)) & 0xFF;
                const int v6 = (int) (((ql_b >> nibsh) & 0xF) | ((((qh_b >> qhsh) & 3) << 4))) - 32;
                out[s*16 + k] = (sycl::half) (v6 * sf);
            }
        }
    } else { // XMXG_Q5_K
        // [qs: nb*128][qh: nb*32][scales: nb*12][dm: nb*4]; 256 values per block
        // in four 64-value quarters q = w/64; value t + 32*upper in a quarter uses
        // qs byte t (nibble by upper) and one qh bit (2*q + upper). The 12 scale
        // bytes pack 8 scales + 8 mins of 6 bits each (decode per CPU dot kernel).
        const size_t nb  = (size_t) N * K / 256;
        const int bi     = row * (K / 256) + k0 / 256;
        const int w0c    = k0 % 256;
        const size_t qh_base = (size_t) nb*128;
        const size_t sc_base = qh_base + (size_t) nb*32;
        // 12-byte scale word is not u32-aligned for odd bi: assemble from bytes
        const uint8_t * sc_b = (const uint8_t *)(vx + sc_base + (size_t) bi*12);
        const auto bu32 = [](const uint8_t * p) {
            return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
        };
        const uint32_t ut0 = bu32(sc_b), ut1 = bu32(sc_b + 4), ut2 = bu32(sc_b + 8);
        const uint32_t kmask1 = 0x3F3F3F3Fu, kmask2 = 0x0F0F0F0Fu, kmask3 = 0x03030303u;
        // packed decode (per the CPU dot kernel): dec[0..1] = scales 0..7, dec[2..3] = mins 0..7
        uint32_t dec[4];
        {
            const uint32_t ut3 = ((ut2 >> 4) & kmask2) | (((ut1 >> 6) & kmask3) << 4);
            const uint32_t uaux  = ut1 & kmask1;
            dec[0] = ut0 & kmask1;
            dec[1] = (ut2 & kmask2) | (((ut0 >> 6) & kmask3) << 4);
            dec[2] = uaux;
            dec[3] = ut3;
        }
        const uint8_t * scales6 = (const uint8_t *) dec;       // [8]
        const uint8_t * mins6   = (const uint8_t *) (dec + 2); // [8]
        // dm plane holds {d, dmin} per block (canonical half2 order)
        const sycl::half * dm2   = (const sycl::half *)(vx + sc_base + (size_t) nb*12 + (size_t) bi*4);
        const float df    = (float) dm2[0];
        const float dminf = (float) dm2[1];
        const uint8_t * qs_c = (const uint8_t *) (vx + (size_t) bi*128);
        const uint8_t * qh_c = (const uint8_t *) (vx + qh_base + (size_t) bi*32);
        #pragma unroll
        for (int s = 0; s < CK/16; s++) {
            const int w0    = w0c + s*16;
            const int q     = w0 / 64;
            const int upper = (w0 % 64) / 32;
            const int g32   = w0 / 32;
            const float sd  = (float) scales6[g32] * df;
            const float dm  = (float) mins6[g32] * dminf;
            const uint8_t * qp = qs_c + q*32 + (w0 % 32);
            const uint8_t * hp = qh_c + (w0 % 32);
            uint32_t qw[4], hw[4];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                qw[b] = *(const uint32_t *)(qp + b*4);
                hw[b] = *(const uint32_t *)(hp + b*4);
            }
            const int bitsh = 2*q + upper;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                const uint32_t qs_b = (qw[k >> 2] >> ((k & 3)*8)) & 0xFF;
                const uint32_t qh_b = (hw[k >> 2] >> ((k & 3)*8)) & 0xFF;
                const int v5 = (int) (((qs_b >> (upper*4)) & 0xF) + (((qh_b >> bitsh) & 1) ? 16 : 0));
                out[s*16 + k] = (sycl::half) (v5 * sd - dm);
            }
        }
    }
}

template <int CNT>
static inline void xmxg_cpy(uint8_t * __restrict__ d, const uint8_t * __restrict__ s) {
    #pragma unroll
    for (int j = 0; j < CNT; j++) {
        d[j] = s[j];
    }
}

// Full-row debug support: one block's raw SoA bytes as a flat 176/210/34-byte
// image (per type), for an independent CPU full-K decode of the first op.
template <int WTYPE>
static inline int xmxg_blk_bytes() {
    if constexpr (WTYPE == XMXG_Q8_0) return 34;
    else if constexpr (WTYPE == XMXG_Q6_K) return 210; // ql128 qh64 sc16 d2
    else return 176;                                   // ql128 qh32 sc12 dm4
}

template <int WTYPE>
static inline uint8_t xmxg_blk_byte(const char * __restrict__ vx, const int N, const int K,
                                    const int bi, const int off) {
    if constexpr (WTYPE == XMXG_Q8_0) {
        const size_t nb = (size_t) N * K / 32;
        if (off < 32) return (uint8_t) vx[(size_t) bi*32 + off];
        return (uint8_t) vx[nb*32 + (size_t) bi*2 + (off - 32)];
    } else if constexpr (WTYPE == XMXG_Q6_K) {
        const size_t nb = (size_t) N * K / 256;
        if (off < 128) return (uint8_t) vx[(size_t) bi*128 + off];
        if (off < 192) return (uint8_t) vx[nb*128 + (size_t) bi*64 + off - 128];
        if (off < 208) return (uint8_t) vx[nb*128 + nb*64 + (size_t) bi*16 + off - 192];
        return (uint8_t) vx[nb*128 + nb*64 + nb*16 + (size_t) bi*2 + off - 208];
    } else {
        const size_t nb = (size_t) N * K / 256;
        if (off < 128) return (uint8_t) vx[(size_t) bi*128 + off];
        if (off < 160) return (uint8_t) vx[nb*128 + (size_t) bi*32 + off - 128];
        if (off < 172) return (uint8_t) vx[nb*128 + nb*32 + (size_t) bi*12 + off - 160];
        return (uint8_t) vx[nb*128 + nb*32 + nb*12 + (size_t) bi*4 + off - 172];
    }
}

// One-shot debug dump (GGML_SYCL_XMX_DBG): group 0 of the first op of each
// type records raw SoA bytes (rows 0..3, block 0), their dequant output, the
// padded activations and the C-tile corners so a CPU recompute can localize
// any error. Raw layout per type (within the 48-byte row slot):
//   q8_0: qs[32] d[2]        q6_K: ql[16] qh[16] sc[1] d[2]
//   q5_K: qs[16] qh[16] scword[12] dm[4]
struct xmxg_dbg_buf {
    int     type;        // wtype enum
    uint8_t raw[4][48];
    float   v[4][16];    // kernel dequant of rows 0..3, k 0..15
    float   act[2][16];  // padded activations rows 0,1, k 0..15
    float   c_dbg[2][4]; // C after the K_eff=16 pass, m 0..1 x f 0..3
    float   c_full[2][4];// C after the full-K pass
};

// dst[feat][m] = sum_k act[m][k] * W[feat][k]; one work-group (one 16-lane
// sub_group) owns a 16-feature tile, lane f dequants feature row feat0+f.
// B staging follows fattn's proven convention: element (k,f) at phys[f*16+k],
// ldm 16. A comes from global f16 (fattn Q16g pattern).
template <int WTYPE, int CK>
static void xmxg_main(const char * __restrict__ vx,
                      const sycl::half * __restrict__ act,
                      float * __restrict__ dst_f,
                      float * __restrict__ part,
                      const int N, const int K, const int M,
                      const int K_eff, const bool dbg, xmxg_dbg_buf * __restrict__ db,
                      const int nsplit, const bool nodq, const bool nomad,
                      const sycl::nd_item<1> & it) {
    const uint32_t lane  = it.get_local_id(0);
    const uint32_t wg    = it.get_group(0);
    const uint32_t nfeat = (uint32_t) N / 16;
    const uint32_t split = nsplit > 1 ? wg / nfeat : 0;
    const uint32_t ftile = nsplit > 1 ? wg - split*nfeat : wg;
    const uint32_t feat0 = ftile * 16;
    const uint32_t row   = feat0 + lane;
    sycl::sub_group sg = it.get_sub_group();

    const int ks   = (int) split * (K_eff / nsplit);
    const int kend = ks + K_eff / nsplit;

    // B tile [16 feat][CK k] f16 feature-major + C staging [16][16] f32 (1KB).
    // CK=64/128: 3KB/5KB total, under the ~8KB per-WG LDS cliff. CK=256: B is 8KB
    // and the C staging reuses the dead B tile after the last mads. A loads
    // straight from global f16 (act rows are [16][K] contiguous halves = fattn
    // Q16g pattern, ldm K); at these shapes the act matrix is L2-resident.
    constexpr int CBYTES = (CK == 256) ? 0 : XMXG_M*16*4;
    syclex::work_group_static<char[16*CK*2 + CBYTES]> lsm;
    sycl::half * B64  = (sycl::half *) &lsm;
    float    * Cbuf   = (float *) ((char *) &lsm + 16*CK*2);

    mx::joint_matrix<sycl::sub_group, sycl::half, use::a, XMXG_M, XMXG_KB, layout::row_major> A_jm;
    mx::joint_matrix<sycl::sub_group, sycl::half, use::b, XMXG_KB, 16, layout::col_major> B_jm;
    mx::joint_matrix<sycl::sub_group, float,    use::accumulator, XMXG_M, 16> C_jm;
    mx::joint_matrix_fill(sg, C_jm, 0.0f);

    for (int k0 = ks; k0 < kend; k0 += CK) {
        const int ck = std::min(CK, kend - k0);
        sg.barrier(); // protect against the previous B load still reading LDS
        if (nodq) {
            #pragma unroll
            for (int j = 0; j < CK; j++) {
                B64[lane*CK + j] = (sycl::half) 0.5f;
            }
        } else if (ck == CK) {
            xmxg_dequant_chunk<WTYPE, CK>(vx, N, K, (int) row, k0, B64 + lane*CK);
        } else {
            #pragma unroll
            for (int s = 0; s < CK / XMXG_KB; s++) {
                const int k = s * XMXG_KB;
                if (k < ck) {
                    xmxg_dequant_row<WTYPE>(vx, N, K, (int) row, k0 + k, B64 + lane*CK + k);
                }
            }
        }
        sg.barrier();
        for (int s = 0; s < CK / XMXG_KB; s++) {
            if (s * XMXG_KB >= ck) {
                break;
            }
            if (nomad) {
                continue;
            }
            // A: act rows [16][K] contiguous halves, slice at col k0+s*16, ldm K
            mx::joint_matrix_load(sg, A_jm, xmp_g_h((sycl::half *) (act + k0 + s*XMXG_KB)), (int) K);
            mx::joint_matrix_load(sg, B_jm, xmp_l_h(B64 + s*XMXG_KB), CK);
            mx::joint_matrix_mad(sg, C_jm, A_jm, B_jm, C_jm);
        }
    }

    // Epilogue: stage C in LDS. Single-split writes dst [N][M] directly; with
    // K splits every WG stores its tile to the f32 partials buffer and a
    // combine pass sums the slices (ggml contiguous [N][M] is column-major
    // over ne[1]: out[n][m] at n + m*N).
    sg.barrier();
    mx::joint_matrix_store(sg, C_jm, xmp_l_f(Cbuf), 16, layout::row_major);
    sg.barrier();
    if (nsplit == 1) {
        for (int m = 0; m < XMXG_M; m++) {
            if (m < M) {
                dst_f[(size_t) (feat0 + lane) + (size_t) m * N] = Cbuf[m*16 + lane];
            }
        }
    } else {
        float * p = part + ((size_t) split*nfeat + ftile)*XMXG_M*16;
        #pragma unroll
        for (int m = 0; m < XMXG_M; m++) {
            p[m*16 + lane] = Cbuf[m*16 + lane];
        }
    }

    if (dbg && feat0 == 0) {
        if (lane < 4) {
            uint8_t * rw = db->raw[lane];
            if constexpr (WTYPE == XMXG_Q8_0) {
                const size_t nb = (size_t) N * K / 32;
                const int bi    = (int) lane * (K / 32);
                xmxg_cpy<32>(rw, (const uint8_t *) (vx + bi*32));
                xmxg_cpy<2>(rw + 32, (const uint8_t *) (vx + nb*32 + (size_t) bi*2));
            } else if constexpr (WTYPE == XMXG_Q6_K) {
                const size_t nb = (size_t) N * K / 256;
                const int bi    = (int) lane * (K / 256);
                xmxg_cpy<16>(rw,        (const uint8_t *) (vx + bi*128));
                xmxg_cpy<16>(rw + 16,   (const uint8_t *) (vx + nb*128 + (size_t) bi*64));
                xmxg_cpy<1>(rw + 32,    (const uint8_t *) (vx + nb*128 + nb*64 + (size_t) bi*16));
                xmxg_cpy<2>(rw + 33,    (const uint8_t *) (vx + nb*128 + nb*64 + nb*16 + (size_t) bi*2));
            } else {
                const size_t nb = (size_t) N * K / 256;
                const int bi    = (int) lane * (K / 256);
                xmxg_cpy<16>(rw,        (const uint8_t *) (vx + bi*128));
                xmxg_cpy<16>(rw + 16,   (const uint8_t *) (vx + nb*128 + (size_t) bi*32));
                xmxg_cpy<12>(rw + 32,   (const uint8_t *) (vx + nb*128 + nb*32 + (size_t) bi*12));
                xmxg_cpy<4>(rw + 44,    (const uint8_t *) (vx + nb*128 + nb*32 + nb*12 + (size_t) bi*4));
            }
        }
        if (lane == 0) {
            db->type = WTYPE;
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                for (int k = 0; k < 16; k++) {
                    db->act[m][k] = (float) act[(size_t) m*K + k];
                }
            }
            float (* cp)[4] = (K_eff == 16) ? db->c_dbg : db->c_full;
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                for (int f = 0; f < 4; f++) {
                    cp[m][f] = Cbuf[m*16 + f];
                }
            }
        }
        if (lane < 4) {
            // B16 now holds the last k0 block; re-dequant block 0 to match raw
            sycl::half tmp[16];
            xmxg_dequant_row<WTYPE>(vx, N, K, (int) row, 0, tmp);
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                db->v[lane][k] = (float) tmp[k];
            }
        }
    }
}

// src1 f32 [K][M] -> act f16 [XMXG_M][K], rows >= M zero-padded.
static void xmxg_act_pad(const float * __restrict__ src1, sycl::half * __restrict__ act,
                         const int K, const int M, sycl::item<1> it) {
    const size_t i = it.get_linear_id();
    const int k = (int) (i % (size_t) K), m = (int) (i / (size_t) K);
    act[m*K + k] = (m < M) ? (sycl::half) src1[k + (size_t) m*K] : (sycl::half) 0.0f;
}

// Dispatch on the runtime weight type inside the kernel; one instantiation per
// (WTYPE, CK). Mirrors the pre-CK three-way branch in the launch lambda.
template <int CK>
static inline void xmxg_run(const int wtype, const char * __restrict__ vx,
                            const sycl::half * __restrict__ act, float * __restrict__ dst_f,
                            float * __restrict__ part, const int N, const int K, const int M,
                            const int K_eff, const bool dbg, xmxg_dbg_buf * __restrict__ db,
                            const int nsplit, const bool nodq, const bool nomad,
                            const sycl::nd_item<1> & it) {
    if (wtype == XMXG_Q8_0) {
        xmxg_main<XMXG_Q8_0, CK>(vx, act, dst_f, part, N, K, M, K_eff, dbg, db, nsplit, nodq, nomad, it);
    } else if (wtype == XMXG_Q6_K) {
        xmxg_main<XMXG_Q6_K, CK>(vx, act, dst_f, part, N, K, M, K_eff, dbg, db, nsplit, nodq, nomad, it);
    } else {
        xmxg_main<XMXG_Q5_K, CK>(vx, act, dst_f, part, N, K, M, K_eff, dbg, db, nsplit, nodq, nomad, it);
    }
}

bool ggml_sycl_op_mul_mat_xmx_gemm(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                   const ggml_tensor * src1, ggml_tensor * dst) {
    if (g_ggml_sycl_xmx_gemm() == 0) {
        return false;
    }
    if (!ggml_sycl_info().devices[ctx.device].has_xmx) {
        return false;
    }
    // split buffers carry per-device sub-allocations; the static helper in
    // ggml-sycl.cpp is not visible here, compare the buffer type name directly
    if (src0->buffer && std::strcmp(src0->buffer->buft->iface.get_name(src0->buffer->buft),
                                    GGML_SYCL_NAME "_Split") == 0) {
        return false;
    }

    int wtype = -1;
    switch (src0->type) {
        case GGML_TYPE_Q8_0: wtype = XMXG_Q8_0; break;
        case GGML_TYPE_Q6_K: wtype = XMXG_Q6_K; break;
        case GGML_TYPE_Q5_K: wtype = XMXG_Q5_K; break;
        default: return false;
    }

    const int N = (int) src0->ne[1]; // output features (weight rows)
    const int K = (int) src0->ne[0]; // reduction dim
    const int M = (int) src1->ne[1];
    if (M < 2 || M > XMXG_M) {
        return false;
    }
    if (K % XMXG_KB != 0 || N % 16 != 0) {
        return false;
    }
    const int ck = g_ggml_sycl_xmxg_ck();
    if (ck != 64 && ck != 128 && ck != 256) {
        return false;
    }
    if (K % ck != 0) {
        return false;
    }
    // block sizes: q8_0 in 32s, k-quants in 256s
    const int kb = wtype == XMXG_Q8_0 ? 32 : 256;
    if (K % kb != 0) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst->ne[0] != N || dst->ne[1] != M) {
        return false;
    }

    if (xmxg_type_mask()[wtype] == 0) {
        return false;
    }

    if (g_ggml_sycl_xmx_dbg()) {
        fprintf(stderr, "[XMXG] xmx gemm: type=%s N=%d K=%d M=%d\n",
                ggml_type_name(src0->type), N, K, M);
    }

    dpct::queue_ptr stream = ctx.stream();
    ggml_sycl_pool & pool  = ctx.pool();

    const char  * vx     = (const char *) src0->data;
    const float * src1_f = (const float *) src1->data;
    float       * dst_f  = (float *) dst->data;

    ggml_sycl_pool_alloc<sycl::half> act16(pool);
    act16.alloc((size_t) XMXG_M * K);
    sycl::half * act_p = act16.ptr;

    if (g_ggml_sycl_xmx_gemm_timing()) {
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait())); // drain any backlog first
    }
    const auto t0 = std::chrono::high_resolution_clock::now();
    stream->parallel_for(sycl::range<1>((size_t) XMXG_M * K),
        [=](sycl::item<1> it) {
            xmxg_act_pad(src1_f, act_p, K, M, it);
        });

    int nsplit = g_ggml_sycl_xmxg_split();
    if (nsplit == 0) { // auto
        nsplit = K >= 16384 ? 8 : (K >= 4096 ? 4 : 1);
    }
    while (nsplit > 1 && K % ((size_t) ck * nsplit) != 0) {
        nsplit /= 2;
    }

    ggml_sycl_pool_alloc<float> part(pool);
    float * part_p = nullptr;
    if (nsplit > 1) {
        part.alloc((size_t) nsplit*(N/16)*XMXG_M*16);
        part_p = part.ptr;
    }

    const bool nodq  = g_ggml_sycl_xmxg_nodq() != 0;
    const bool nomad = g_ggml_sycl_xmxg_nomad() != 0;
    static bool dbg_done[3] = { false, false, false };
    const bool dbg = g_ggml_sycl_xmx_dbg() != 0 && !dbg_done[wtype];
    if (dbg) {
        dbg_done[wtype] = true;
    }

    const bool need_wait = dbg || g_ggml_sycl_xmx_gemm_timing() != 0;

    const sycl::range<1> nd_global((size_t) N * nsplit); // (N/16)*nsplit WGs of 16 lanes
    const sycl::range<1> nd_local(16);
    ggml_sycl_pool_alloc<char> dbgbuf(pool);
    xmxg_dbg_buf * dbd = nullptr;
    if (dbg) {
        dbgbuf.alloc(sizeof(xmxg_dbg_buf));
        dbd = (xmxg_dbg_buf *) dbgbuf.ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dbd, 0, sizeof(xmxg_dbg_buf))));
    }

    auto launch = [&](auto ck_c) {
        constexpr int CK = decltype(ck_c)::value;
        if (dbg) {
            // pass 1: K_eff=16, unsplit, so c_dbg matches a 16-term CPU dot
            stream->parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) N), nd_local),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    xmxg_run<CK>(wtype, vx, act_p, dst_f, nullptr, N, K, M, 16, true, dbd, 1, false, false, it);
                });
            SYCL_CHECK(0);
        }
        stream->parallel_for(sycl::nd_range<1>(nd_global, nd_local),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                xmxg_run<CK>(wtype, vx, act_p, dst_f, part_p, N, K, M, K, dbg && nsplit == 1, dbd, nsplit, nodq, nomad, it);
            });
    };
    if (ck == 64) {
        launch(std::integral_constant<int, 64>{});
    } else if (ck == 128) {
        launch(std::integral_constant<int, 128>{});
    } else {
        launch(std::integral_constant<int, 256>{});
    }

    if (nsplit > 1) {
        stream->parallel_for(sycl::range<1>((size_t) N*XMXG_M),
            [=](sycl::item<1> it) {
                const int i = (int) it.get_linear_id();
                const int n = i % N, m = i / N;
                if (m >= M) {
                    return;
                }
                float acc = 0.0f;
                for (int s = 0; s < nsplit; s++) {
                    acc += part_p[((size_t) s*(N/16) + (n/16))*XMXG_M*16 + m*16 + (n%16)];
                }
                dst_f[n + (size_t) m*N] = acc;
            });
    }
    if (need_wait) {
        SYCL_CHECK(0);
    }

    if (g_ggml_sycl_xmx_gemm_timing()) {
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        const auto t1 = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "[XMXG-T] type=%s N=%d K=%d M=%d CK=%d nsplit=%d wall=%.3f ms\n",
                ggml_type_name(src0->type), N, K, M, ck, nsplit,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    if (dbg) {
        // Full-K cross-check: act rows 0..1 for all K plus raw block bytes of
        // weight rows 0..3; the CPU re-decodes and dots against c_full.
        const int kb   = wtype == XMXG_Q8_0 ? 32 : 256;
        const int nblk = K / kb;
        const int blkB = wtype == XMXG_Q8_0 ? xmxg_blk_bytes<XMXG_Q8_0>()
                       : wtype == XMXG_Q6_K ? xmxg_blk_bytes<XMXG_Q6_K>()
                                            : xmxg_blk_bytes<XMXG_Q5_K>();
        static char hbig[384*1024];
        ggml_sycl_pool_alloc<float> big(pool);
        big.alloc((size_t) 2*K + (size_t) 4*nblk*blkB);
        float * bg = big.ptr;
        stream->parallel_for(sycl::range<1>((size_t) 2*K),
            [=](sycl::item<1> it) {
                const int i = (int) it.get_linear_id();
                const int k = i % K, m = i / K;
                bg[i] = (float) act_p[(size_t) m*K + k];
            });
        stream->parallel_for(sycl::range<1>((size_t) 4*nblk*blkB),
            [=](sycl::item<1> it) {
                const int i = (int) it.get_linear_id();
                const int bi = i / blkB, off = i % blkB;
                uint8_t b;
                if (wtype == XMXG_Q8_0) {
                    b = xmxg_blk_byte<XMXG_Q8_0>(vx, N, K, bi, off);
                } else if (wtype == XMXG_Q6_K) {
                    b = xmxg_blk_byte<XMXG_Q6_K>(vx, N, K, bi, off);
                } else {
                    b = xmxg_blk_byte<XMXG_Q5_K>(vx, N, K, bi, off);
                }
                bg[(size_t) 2*K + i] = (float) b;
            });
        SYCL_CHECK(0);
        const size_t ntot = (size_t) 2*K + (size_t) 4*nblk*blkB;
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(hbig, bg, ntot*4).wait()));
        const float * af = (const float *) hbig;
        for (int m = 0; m < 2; m++) {
            fprintf(stderr, "[XMXG-DBG] AF%d:", m);
            for (int k = 0; k < K; k++) {
                fprintf(stderr, " %.6g", af[(size_t) m*K + k]);
            }
            fprintf(stderr, "\n");
        }
        const float * rfp = af + 2*(size_t) K;
        for (int r = 0; r < 4; r++) {
            fprintf(stderr, "[XMXG-DBG] RF%d:", r);
            for (int b = 0; b < nblk*blkB; b++) {
                fprintf(stderr, " %02x", (int) rfp[(size_t) r*nblk*blkB + b]);
            }
            fprintf(stderr, "\n");
        }
    }

    if (dbg) {
        char hbuf[768];
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memcpy(hbuf, dbd, sizeof(xmxg_dbg_buf)).wait()));
        const xmxg_dbg_buf & db = *(const xmxg_dbg_buf *) hbuf;
        fprintf(stderr, "[XMXG-DBG] type=%s N=%d K=%d M=%d\n", ggml_type_name(src0->type), (int) N, (int) K, (int) M);
        for (int f = 0; f < 4; f++) {
            fprintf(stderr, "[XMXG-DBG] row %d raw:", f);
            for (int b = 0; b < 48; b++) {
                fprintf(stderr, " %02x", db.raw[f][b]);
            }
            fprintf(stderr, "\n[XMXG-DBG]   v:");
            for (int k = 0; k < 16; k++) {
                fprintf(stderr, " %.6g", db.v[f][k]);
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "[XMXG-DBG] act0:");
        for (int k = 0; k < 16; k++) {
            fprintf(stderr, " %.6g", db.act[0][k]);
        }
        fprintf(stderr, "\n[XMXG-DBG] act1:");
        for (int k = 0; k < 16; k++) {
            fprintf(stderr, " %.6g", db.act[1][k]);
        }
        fprintf(stderr, "\n[XMXG-DBG] c_dbg m0:");
        for (int f = 0; f < 4; f++) {
            fprintf(stderr, " %.6g", db.c_dbg[0][f]);
        }
        fprintf(stderr, " m1:");
        for (int f = 0; f < 4; f++) {
            fprintf(stderr, " %.6g", db.c_dbg[1][f]);
        }
        fprintf(stderr, "\n[XMXG-DBG] c_full m0:");
        for (int f = 0; f < 4; f++) {
            fprintf(stderr, " %.6g", db.c_full[0][f]);
        }
        fprintf(stderr, " m1:");
        for (int f = 0; f < 4; f++) {
            fprintf(stderr, " %.6g", db.c_full[1][f]);
        }
        fprintf(stderr, "\n");
    }
    return true;
}
