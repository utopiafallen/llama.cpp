//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "common.hpp"
#include "fattn.hpp"
#include "fattn-xmx-decode.hpp"

#include <float.h>
#include <cmath>
#include <algorithm>
#include <optional>

namespace mx = sycl::ext::oneapi::experimental::matrix;
using mx::use;
using mx::layout;
namespace syclex = sycl::ext::oneapi::experimental;

// KV positions per work-group split: 128 for decode, 64 for MTP verify (measured optimum at Q=2)
#define XMX_DECODE_SPLIT 128
#define XMX_VERIFY_SPLIT 64

using xmp_g_f = sycl::multi_ptr<float,  sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
using xmp_l_h = sycl::multi_ptr<sycl::half, sycl::access::address_space::local_space,  sycl::access::decorated::legacy>;
using xmp_l_f = sycl::multi_ptr<float,  sycl::access::address_space::local_space,  sycl::access::decorated::legacy>;
using xmp_g_i8  = sycl::multi_ptr<int8_t,  sycl::access::address_space::global_space, sycl::access::decorated::legacy>;

// One work-group (one 16-lane sub_group) handles one KV split and one KV head.
// MQ query positions are batched into M XMX rows: row r = (q = r/GQA, qi = r%GQA), packed
// densely. M >= GQA*MQ always holds (decode M = gqa, verify M = 16), so the storage row
// stride is M and full-tile joint_matrix_store writes never cross block bounds.
// D is fixed at 256 by the dispatch gate; SPLIT/MQ are compile-time.
// QK^T goes through dual-plane int8 XMX (K = 32 = one SoA 32-dim block, no zero padding):
// Q = s_row*(qhi + qlo/16), ~15-bit effective Q, in the two int8 A8g planes; K stays
// bit-exact to the SoA cache. Per 32-dim block: hi/lo int32 mads per tile, and the f32
// Csc acc is applied directly from the int32 C8/C8l accs (mixed f32-acc / int32-src
// joint_matrix_apply - no int32->f32 copy). The row scale sQg is applied on the single
// per-chunk C round-trip. Only error vs f16 QK: the Q quantization (~1e-3 logit,
// invisible at production sampling temperature).
// NOTE: joint_matrix_apply lambdas must take elements BY REFERENCE and mutate in place
// (x += ...); the return value is discarded (sycl_ext_oneapi_matrix spec). A by-value
// lambda that returns is a silent no-op.
template <int M, int GQA, int SPLIT, int MQ>
static void xmx_decode_main(
        const sycl::half * __restrict__ mask,
        void * __restrict__ partial_O, // f32 partial
        float * __restrict__ partial_m,
        float * __restrict__ partial_l,
        const char  * __restrict__ K_q8,
        const char  * __restrict__ V_q8,
        const int8_t * __restrict__ A8g, // [n_wg_heads][2 tiles][8 blk][8 rows][32] int8 Q codes
        const float  * __restrict__ sQg, // [n_wg_heads][16 rows][8 blk] f32 Q block scales
        const int n_kv, const int n_kv_heads, const int n_q_heads,
        const int k_pos_stride_b, const int v_pos_stride_b,
        const int mask_head_stride, const int mask_ne1,
        const float scale, const int gqa_r, const int n_hg, // real GQA ratio and head-groups per kv head (1 = dense)
        const sycl::nd_item<3> & it) {
    constexpr int D    = 256;
    constexpr int ROWS = GQA * MQ; // live rows
    const int split   = it.get_group(0);
    const int kv_head = it.get_group(1);
    const int lane    = it.get_local_id(2);
    sycl::sub_group sg = it.get_sub_group();

    // head-group split: one WG handles GQA of the gqa_r sibling heads (group n_hg > 1)
    const int kv_real = (n_hg > 1) ? kv_head / n_hg : kv_head;
    const int hg      = (n_hg > 1) ? kv_head % n_hg : 0;
    const int qh0     = kv_real * gqa_r + hg * GQA; // first q-head index of this WG
    const int n_wg_heads = n_kv_heads * n_hg;

    const int pos_base = split * SPLIT;
    const int pos_end  = std::min(pos_base + SPLIT, n_kv);

    // scores/P16 in LDS, plus the 16x16 staging tile and the int8 C store area. The C
    // store (ldm=16 int32/f32 acc stores = 2KB/WG) is kept small: crossing the ~8KB/WG
    // LDS cliff costs ~+50% FA (b70-decode-perf skill).
    constexpr int LDS_BYTES = M*SPLIT*4 + M*SPLIT*2 + 16*16*2 + 2*8*16*4;
    syclex::work_group_static<char[LDS_BYTES]> lsm;
    float    * scores   = (float *)&lsm;                    // [M][SPLIT]
    sycl::half * P16    = (sycl::half *)(scores + M*SPLIT); // [M][SPLIT]
    sycl::half * tile_buf = (sycl::half *)(P16 + M*SPLIT);  // [16][16] staging
    float    * c_lf     = (float *)(tile_buf + 256);        // [2][8][16] ldm=16

    // int8 QK operands (M=8 tiles; K = 32 = one SoA 32-dim block, no zero padding)
    mx::joint_matrix<sycl::sub_group, int8_t, use::a, 8, 32, layout::row_major> A8;
    mx::joint_matrix<sycl::sub_group, int8_t, use::a, 8, 32, layout::row_major> A8l; // lo plane
    mx::joint_matrix<sycl::sub_group, int8_t, use::b, 32, 16, layout::col_major> B8;
    mx::joint_matrix<sycl::sub_group, int32_t, use::accumulator, 8, 16> C8[2];
    mx::joint_matrix<sycl::sub_group, float, use::accumulator, 8, 16> Csc[2]; // cross-block f32 sum
    // lo-plane int32 acc (C8 doubles as the hi-plane acc; after the int32 combine
    // C8 = 16*C_hi + C_lo, exact)
    mx::joint_matrix<sycl::sub_group, int32_t, use::accumulator, 8, 16> C8l[2];
    // f16 PV operands (P probabilities x dequantized V)
    mx::joint_matrix<sycl::sub_group, sycl::half, use::a, M, 16, layout::row_major> A_jm;
    mx::joint_matrix<sycl::sub_group, sycl::half, use::b, 16, 16, layout::row_major> B_v;
    mx::joint_matrix<sycl::sub_group, float, use::accumulator, M, 16> O_jm;

    // 2. QK^T: scores = Q @ K^T, per 16-pos chunk
    for (int c = 0; c < SPLIT/16; c++) {
        const char * k_row = K_q8 + (pos_base + c*16 + lane)*k_pos_stride_b;
        // per-head segment: [D qs][D/32 half scales]; scales start at seg_start + D
        sycl::vec<sycl::half, 8> k_scv = *(const sycl::vec<sycl::half, 8> *)(k_row + kv_real*(D + 16) + D);
        const char * k_blk = K_q8 + (pos_base + c*16)*k_pos_stride_b;
        mx::joint_matrix_fill(sg, Csc[0], 0.0f);
        mx::joint_matrix_fill(sg, Csc[1], 0.0f);
        for (int blk = 0; blk < 8; blk++) {
            const int dim0 = blk * 32;
            const float sk   = (float) k_scv[blk];
            const float sk16 = (float) k_scv[blk] * 0.0625f;
            // B is tile-invariant (both tiles share the same K tile): load once per block
            mx::joint_matrix_load(sg, B8, xmp_g_i8((int8_t *)(k_blk + kv_real*(D + 16) + dim0)), k_pos_stride_b);
            #pragma unroll
            for (int tile = 0; tile < 2; tile++) {
                int8_t * ah = (int8_t *)(A8g + (size_t) kv_head*8192 + (tile*8 + blk)*256);
                mx::joint_matrix_load(sg, A8, xmp_g_i8(ah), 32);
                mx::joint_matrix_load(sg, A8l, xmp_g_i8(ah + 4096), 32);
                mx::joint_matrix_fill(sg, C8[tile], 0);
                mx::joint_matrix_fill(sg, C8l[tile], 0);
                mx::joint_matrix_mad(sg, C8[tile], A8, B8, C8[tile]);
                mx::joint_matrix_mad(sg, C8l[tile], A8l, B8, C8l[tile]);
                // no C8f copies: read the int32 accs straight, convert in the lambda
                mx::joint_matrix_apply(sg, Csc[tile], C8[tile],  [sk](float &x, int32_t &y) { x += (float) y * sk; });
                mx::joint_matrix_apply(sg, Csc[tile], C8l[tile], [sk16](float &x, int32_t &y) { x += (float) y * sk16; });
            }
        }
        mx::joint_matrix_store(sg, Csc[0], xmp_l_f(c_lf), 16, layout::row_major);
        mx::joint_matrix_store(sg, Csc[1], xmp_l_f(c_lf + 128), 16, layout::row_major);
        sg.barrier();
        #pragma unroll
        for (int r = 0; r < M; r++) {
            const float v = sQg[(size_t) kv_head*16 + r]
                          * (float) c_lf[(r >> 3)*128 + (r & 7)*16 + lane];
            scores[r*SPLIT + c*16 + lane] = v;
        }
        sg.barrier();
        // scale + causal + mask -> -FLT_MAX (each element handled by one lane)
        for (int idx = lane; idx < M*16; idx += 16) {
            const int r = idx / 16, p = idx % 16;
            float & sref = scores[r*SPLIT + c*16 + p];
            if constexpr (MQ > 1) {
                if (r >= ROWS) { sref = -FLT_MAX; continue; }
            }
            const int pos = pos_base + c*16 + p;
            const int q   = (MQ > 1 ? r / GQA : 0);
            float s = sref * scale;
            if (pos >= std::min(pos_end, n_kv - MQ + q + 1)) {
                s = -FLT_MAX;
            } else if (mask != nullptr) {
                const int qi = (MQ > 1 ? r % GQA : r);
                const int qh = qh0 + qi;
                const int mi = mask_ne1 > 1 ? (MQ > 1 ? q : qh) : 0;
                const float msk = (float) mask[pos + mi*mask_head_stride];
                s = msk < 0.0f ? -FLT_MAX : s + msk;
            }
            sref = s;
        }
        sg.barrier();
    }

    // 3. softmax over the split, per query row (unnormalized; cross-split scale in combine)
    for (int r = 0; r < ROWS; r++) {
        float pm = -FLT_MAX;
        for (int pos = lane; pos < SPLIT; pos += 16) pm = std::max(pm, scores[r*SPLIT + pos]);
        const float m = sycl::reduce_over_group(sg, pm, sycl::maximum<float>());
        float ps = 0.0f;
        for (int pos = lane; pos < SPLIT; pos += 16) {
            const float e = exp2f((scores[r*SPLIT + pos] - m) * 1.4426950408889634f);
            P16[r*SPLIT + pos] = (sycl::half) e;
            ps += e;
        }
        const float l = sycl::reduce_over_group(sg, ps, sycl::plus<float>());
        if (lane == 0) {
            // one M-row block per (split, virtual kv head)
            const size_t base = ((size_t) split * n_wg_heads + kv_head) * M;
            partial_m[base + r] = m;
            partial_l[base + r] = l;
        }
    }
    sg.barrier();

    // 4. PV: O = P @ V, per 16-dim chunk
    for (int dc = 0; dc < D/16; dc++) {
        mx::joint_matrix_fill(sg, O_jm, 0.0f);
        const int dim0 = dc * 16;
        for (int c = 0; c < SPLIT/16; c++) {
            const char * v_row = V_q8 + (pos_base + c*16 + lane)*v_pos_stride_b;
            const int8_t * qs = (const int8_t *)(v_row + kv_real*(D + 16) + dim0);
            const sycl::vec<int8_t, 16> qv = *(const sycl::vec<int8_t, 16> *)qs;
            const sycl::vec<sycl::half, 8> v_scv = *(const sycl::vec<sycl::half, 8> *)(v_row + kv_real*(D + 16) + D);
            const float s = (float) v_scv[dc/2];
            #pragma unroll
            for (int d = 0; d < 16; d++) {
                tile_buf[lane*16 + d] = (sycl::half)((float)qv[d] * s);
            }
            sg.barrier();
            mx::joint_matrix_load(sg, B_v, xmp_l_h(tile_buf), 16);
            mx::joint_matrix_load(sg, A_jm, xmp_l_h(P16 + c*16), SPLIT);
            mx::joint_matrix_mad(sg, O_jm, A_jm, B_v, O_jm);
        }
        mx::joint_matrix_store(sg, O_jm,
            xmp_g_f((float *) partial_O + ((size_t) split * n_wg_heads + kv_head) * M * D + dc*16),
            D, layout::row_major);
    }
}

// Combine the per-split partials (flash-decoding merge) into the final output, in two phases:
// phase A merges a chunk of splits per (row, chunk) into the inter buffers, phase B merges
// the chunks into out. Scalar phase A reaches only ~200 GB/s on the partials stream @143K;
// the chunked vec4 merge lifts the in-flight bytes per instruction on the same 128KB-strided
// per-split reads to ~420 GB/s (FA node -19%, wall -8.4% @143K MQ=5; vec8 measured equal).
// Same f32 reassociation as a scalar merge; below 16 splits it degenerates to a single chunk.
// Per-split storage is local: p_stride rows, row rr = q*rows_per_q + (h % gqa), with
// rows_per_q = gqa_a (the tile always fits the head-group's rows).
static void xmx_decode_combine2a(
        const void * __restrict__ partial_O, // f32 partials
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        float * __restrict__ inter_O,      // [n_chunks][n_q_rows][D]
        float * __restrict__ inter_m,      // [n_chunks][n_q_rows]
        float * __restrict__ inter_l,      // [n_chunks][n_q_rows]
        const int n_q_heads, const int n_kv_heads, const int MQ, const int n_hg,
        const int D, const int n_splits, const int p_stride, const int n_chunks,
        const sycl::nd_item<3> & it) {
    const int n_rows = n_q_heads * MQ;
    const int q_head = it.get_group(0) % n_rows;
    const int chunk  = it.get_group(0) / n_rows;
    const int dim0   = it.get_group(1)*64 + it.get_local_id(2)*4;
    if (dim0 >= D) {
        return;
    }
    const int gqa   = n_q_heads / n_kv_heads;
    const int gqa_a = (n_hg > 1) ? gqa / n_hg : gqa;
    const int h  = q_head % n_q_heads;
    const int kv = h / gqa;
    const int hg = (h % gqa) / gqa_a;
    const int rr = (q_head / n_q_heads) * gqa_a + (h % gqa_a);
    const size_t blk = ((size_t) kv * n_hg + hg) * p_stride;
    const float * m_s = partial_m + blk + rr;
    const float * l_s = partial_l + blk + rr;
    const float * O_s = (const float *) partial_O + (blk + rr) * D;
    const size_t sp = (size_t) n_kv_heads * n_hg * p_stride;
    const int csz = (n_splits + n_chunks - 1) / n_chunks;
    const int s0 = chunk * csz;
    const int s1 = std::min(s0 + csz, n_splits);
    float Mc = -FLT_MAX;
    for (int s = s0; s < s1; s++) {
        Mc = std::max(Mc, m_s[s * sp]);
    }
    sycl::vec<float, 4> num_v = sycl::vec<float, 4>(0.0f);
    float den = 0.0f;
    for (int s = s0; s < s1; s++) {
        const float w = exp2f((m_s[s * sp] - Mc) * 1.4426950408889634f);
        den += w * l_s[s * sp];
        const sycl::vec<float, 4> o = *(const sycl::vec<float, 4> *)(O_s + s * sp * D + dim0);
        for (int v = 0; v < 4; v++) {
            num_v[v] += w * o[v];
        }
    }
    inter_m[chunk*n_rows + q_head] = Mc;
    inter_l[chunk*n_rows + q_head] = den;
    float * io = inter_O + ((size_t) chunk*n_rows + q_head)*D + dim0;
    for (int v = 0; v < 4; v++) {
        if (dim0 + v < D) {
            io[v] = num_v[v];
        }
    }
}

static void xmx_decode_combine2b(
        const float * __restrict__ inter_O,      // [n_chunks][n_q_rows][D]
        const float * __restrict__ inter_m,      // [n_chunks][n_q_rows]
        const float * __restrict__ inter_l,      // [n_chunks][n_q_rows]
        float * __restrict__ out,
        const int n_q_heads, const int MQ,
        const int D, const int n_chunks,
        const int out_head_stride, const int out_pos_stride, const sycl::nd_item<3> & it) {
    // grid: (n_q_rows, D/16, 16)
    const int q_head = it.get_group(0);
    const int dim    = it.get_group(1)*16 + it.get_local_id(2);
    if (dim >= D) {
        return;
    }
    const int h = q_head % n_q_heads;
    // inter layout is [chunk][row] with n_rows rows per chunk (see combine2a)
    const int n_rows = n_q_heads * MQ;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; c++) {
        M = std::max(M, inter_m[c*n_rows + q_head]);
    }
    float num = 0.0f, den = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        const float w = exp2f((inter_m[c*n_rows + q_head] - M) * 1.4426950408889634f);
        den += w * inter_l[c*n_rows + q_head];
        num += w * inter_O[((size_t) c*n_rows + q_head)*D + dim];
    }
    out[(size_t) (q_head / n_q_heads) * (MQ > 1 ? out_pos_stride : 0) + (size_t) h * out_head_stride + dim] = num / den;
}

static int xmx_decode_verify() {
    // XMX MTP verify (Q=2..5) on by default; GGML_SYCL_FA_XMX_VERIFY=0 forces the tile fallback.
    static int v = ggml_sycl_get_env("GGML_SYCL_FA_XMX_VERIFY", 1);
    return v;
}

bool ggml_sycl_flash_attn_ext_xmx_decode_supported(int device, const ggml_tensor * dst) {
    if (!ggml_sycl_info().devices[device].has_xmx) {
        return false;
    }
    if (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!Q || !K || !V) {
        return false;
    }
    const int n_q_pos = (int) Q->ne[1]; // decode: 1, MTP verify: 2, deep verify: 3..5
    if (n_q_pos < 1 || n_q_pos > 5) {
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) { // single batch
        return false;
    }
    if (Q->type != GGML_TYPE_F32) {
        return false;
    }
    if (K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (!ggml_sycl_is_q8_0_soa(K) || !ggml_sycl_is_q8_0_soa(V)) {
        return false;
    }
    if (dst->src[4] != nullptr) { // sinks
        return false;
    }
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        return false;
    }
    if (K->ne[0] != 256) {
        return false;
    }
    const int gqa = (int) (Q->ne[2] / K->ne[2]);
    if (!(gqa >= 1 && gqa <= 8)) {
        return false;
    }
    if (n_q_pos >= 2 && xmx_decode_verify() == 0) {
        return false;
    }
    if (n_q_pos >= 3 && (gqa < 3 || gqa % 3 != 0)) {
        return false;
    }
    return true;
}

void ggml_sycl_flash_attn_ext_xmx_decode(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int D            = (int) K->ne[0];
    const int n_kv         = (int) K->ne[1];
    const int n_kv_heads   = (int) K->ne[2];
    const int n_q_heads    = (int) Q->ne[2];
    const int MQ           = (int) Q->ne[1]; // query positions (1 decode, 2 MTP verify)
    const int gqa          = n_q_heads / n_kv_heads;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    dpct::queue_ptr stream = ctx.stream();
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    ggml_sycl_pool & pool = ctx.pool();
    const int n_q_rows = n_q_heads * MQ;

    const float * Q_h = (const float *) Q->data;
    const char * K_q8_p = (const char *) K->data;
    const char * V_q8_p = (const char *) V->data;

    const int k_pos_stride_b  = (int) K->nb[1];
    const int v_pos_stride_b  = (int) V->nb[1];

    const sycl::half * m_h = mask ? (const sycl::half *) mask->data : nullptr;
    float * out_f = (float *) dst->data;

    const int q_pos_stride    = (int) (Q->nb[1] / sizeof(float));
    const int q_head_stride   = (int) (Q->nb[2] / sizeof(float));
    const int mask_head_stride = mask ? (int) (mask->nb[1] / sizeof(sycl::half)) : 0;
    const int mask_ne1         = mask ? (int) (mask->ne[1]) : 1;
    const int out_head_stride = (int) (dst->nb[1] / sizeof(float));
    const int out_pos_stride  = (int) (dst->nb[2] / sizeof(float));

    // head-group split for deep verify (MQ 3..5, gqa % 3 == 0): one M=16 tile per 3-head group
    const int n_hg_l = (MQ >= 3) ? gqa / 3 : 1;

    const sycl::range<3> wg_local(1, 1, 16);

    // int8 Q-code scratch per (virtual) kv head: [b][2 tiles][8 blk][8 rows][32] int8
    // + per (row, blk) f32 scale. Row->(q, qh) mapping mirrors the dispatch shape.
    ggml_sycl_pool_alloc<int8_t> i8a_scratch(pool);
    ggml_sycl_pool_alloc<float> sQ_scratch(pool);
    const int i8_GQA = (MQ >= 3) ? 3 : gqa;
    const size_t n_b = (size_t) n_kv_heads * n_hg_l;
    i8a_scratch.alloc(n_b * 2*8*8*32 * 2); // two planes: hi + lo
    sQ_scratch.alloc(n_b * 16*8);
    int8_t * A8_p = i8a_scratch.ptr;
    float * sQ_p = sQ_scratch.ptr;

        // Dual-plane row-scale Q: q = s*(qhi + qlo/16), one s per row. qhi = nearint(q/s)
        // (|.| <= 127), qlo = nearint(16*(q - s*qhi)/s) (|.| <= 8, no overflow). Two int8
        // planes: A8_p[b][plane][tile][blk][row][32]; sQ_p[b*16 + r] = s.
        stream->parallel_for(sycl::range<1>(n_b * 16),
            [=](sycl::item<1> it2) {
                const int i = (int) it2.get_linear_id();
                const int r = i % 16, b = i / 16;
                const int kv2 = b / n_hg_l, hg2 = b % n_hg_l;
                const int qh_base = kv2*gqa + hg2*i8_GQA;
                const int q  = r / i8_GQA;
                const int qi = r % i8_GQA;
                const bool live = r < i8_GQA*MQ;
                const float * qv = Q_h + q*q_pos_stride + (qh_base + qi)*q_head_stride;
                float s = 0.0f;
                if (live) {
                    #pragma unroll
                    for (int k = 0; k < D; k++) s = std::max(s, std::fabsf(qv[k]));
                    s /= 127.0f;
                    if (s == 0.0f) s = 1.0f;
                }
                int8_t * base_h = A8_p + (size_t)b*8192 + (r >> 3)*2048;
                int8_t * base_l = base_h + 4096;
                #pragma unroll
                for (int blk = 0; blk < 8; blk++) {
                    sycl::vec<int8_t, 16> v0, v1, w0, w1;
                    if (live) {
                        #pragma unroll
                        for (int k = 0; k < 16; k++) {
                            const float q0 = qv[blk*32 + k], q1 = qv[blk*32 + 16 + k];
                            const int hi0 = (int) std::nearbyint(q0 / s);
                            const int hi1 = (int) std::nearbyint(q1 / s);
                            v0[k] = (int8_t) hi0;
                            v1[k] = (int8_t) hi1;
                            w0[k] = (int8_t) std::nearbyint((q0 - s*(float)hi0) / s * 16.0f);
                            w1[k] = (int8_t) std::nearbyint((q1 - s*(float)hi1) / s * 16.0f);
                        }
                    } else {
                        #pragma unroll
                        for (int k = 0; k < 16; k++) { v0[k] = w0[k] = 0; v1[k] = w1[k] = 0; }
                    }
                    int8_t * dh = base_h + blk*256 + (r & 7)*32;
                    int8_t * dl = base_l + blk*256 + (r & 7)*32;
                    *(sycl::vec<int8_t, 16> *)dh = v0;
                    *(sycl::vec<int8_t, 16> *)(dh + 16) = v1;
                    *(sycl::vec<int8_t, 16> *)dl = w0;
                    *(sycl::vec<int8_t, 16> *)(dl + 16) = w1;
                }
                sQ_p[(size_t)b*16 + r] = s;
            });
    SYCL_CHECK(0);

    // M = XMX tile rows; the storage row stride in the partials = M
    #define XMX_DECODE_DISPATCH(SPL) \
        if (MQ >= 3) { \
            switch (MQ) { \
                case 3: XMX_DECODE_LAUNCH(16, 3, 3, SPL); break; \
                case 4: XMX_DECODE_LAUNCH(16, 3, 4, SPL); break; \
                case 5: XMX_DECODE_LAUNCH(16, 3, 5, SPL); break; \
            } \
        } else if (MQ == 1) { \
            switch (gqa) { \
                case 1: XMX_DECODE_LAUNCH(1, 1, 1, SPL); break; \
                case 2: XMX_DECODE_LAUNCH(2, 2, 1, SPL); break; \
                case 3: XMX_DECODE_LAUNCH(3, 3, 1, SPL); break; \
                case 4: XMX_DECODE_LAUNCH(4, 4, 1, SPL); break; \
                case 5: XMX_DECODE_LAUNCH(5, 5, 1, SPL); break; \
                case 6: XMX_DECODE_LAUNCH(6, 6, 1, SPL); break; \
                case 7: XMX_DECODE_LAUNCH(7, 7, 1, SPL); break; \
                case 8: XMX_DECODE_LAUNCH(8, 8, 1, SPL); break; \
                default: GGML_ABORT("xmx decode: unsupported gqa %d", gqa); \
            } \
        } else { \
            switch (gqa) { \
                case 4: XMX_DECODE_LAUNCH(16, 4, 2, SPL); break; \
                case 5: XMX_DECODE_LAUNCH(16, 5, 2, SPL); break; \
                case 6: XMX_DECODE_LAUNCH(16, 6, 2, SPL); break; \
                case 7: XMX_DECODE_LAUNCH(16, 7, 2, SPL); break; \
                case 8: XMX_DECODE_LAUNCH(16, 8, 2, SPL); break; \
                default: GGML_ABORT("xmx decode verify: unsupported gqa %d", gqa); \
            } \
        }
    #define XMX_DECODE_RUN(SPL) \
        { \
            const int n_splits_l = (n_kv + SPL - 1) / SPL; \
            /* per-WG row stride in the partials: matches the kernel's M */ \
            const int p_stride_l = (MQ == 1) ? gqa : 16; \
            /* each split holds one p_stride_l-row block per (virtual) kv head */ \
            const size_t rows_l = (size_t) n_splits_l * n_kv_heads * n_hg_l * p_stride_l; \
            std::optional<ggml_sycl_pool_alloc<float>> pOf; \
            ggml_sycl_pool_alloc<float> pm(pool); \
            ggml_sycl_pool_alloc<float> pl(pool); \
            pOf.emplace(pool); pOf->alloc(rows_l * D); void * pO_pv = pOf->ptr; \
            pm.alloc(rows_l); \
            pl.alloc(rows_l); \
            const sycl::range<3> wg_global_l(n_splits_l, n_kv_heads * n_hg_l, 16); \
            float * pm_p = pm.ptr; float * pl_p = pl.ptr; \
            XMX_DECODE_DISPATCH(SPL) \
            { \
                const sycl::range<3> c_local(1, 1, 16); \
                /* two-phase: chunk the split range into independent merge streams */ \
                const int n_chunks_l = std::min(64, (n_splits_l + 15) / 16); \
                ggml_sycl_pool_alloc<float> iO(pool); \
                ggml_sycl_pool_alloc<float> iml(pool); \
                iO.alloc((size_t) n_chunks_l * n_q_rows * D); \
                iml.alloc((size_t) n_chunks_l * n_q_rows * 2); \
                float * iO_p = iO.ptr; float * iml_p = iml.ptr; \
                stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(n_q_rows * n_chunks_l, D / 64, 16), c_local), \
                    [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] { \
                        xmx_decode_combine2a(pO_pv, pm_p, pl_p, iO_p, iml_p, iml_p + (size_t) n_chunks_l * n_q_rows, \
                                             n_q_heads, n_kv_heads, MQ, n_hg_l, \
                                             D, n_splits_l, p_stride_l, n_chunks_l, it); \
                    }); \
                SYCL_CHECK(0); \
                stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(n_q_rows, (D + 15) / 16, 16), c_local), \
                    [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] { \
                        xmx_decode_combine2b(iO_p, iml_p, iml_p + (size_t) n_chunks_l * n_q_rows, out_f, \
                                             n_q_heads, MQ, D, n_chunks_l, out_head_stride, out_pos_stride, it); \
                    }); \
                SYCL_CHECK(0); \
            } \
        }

    #define XMX_DECODE_LAUNCH(MM, GQACT, MQCT, SPLITCT) \
        stream->parallel_for(sycl::nd_range<3>(wg_global_l, wg_local), \
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] { \
                xmx_decode_main<MM, GQACT, SPLITCT, MQCT>(m_h, pO_pv, pm_p, pl_p, K_q8_p, V_q8_p, A8_p, sQ_p, \
                    n_kv, n_kv_heads, n_q_heads, k_pos_stride_b, v_pos_stride_b, \
                    mask_head_stride, mask_ne1, scale, gqa, n_hg_l, it); \
            }); \
        SYCL_CHECK(0)

    if (MQ == 1) {
        XMX_DECODE_RUN(XMX_DECODE_SPLIT)
    } else {
        XMX_DECODE_RUN(XMX_VERIFY_SPLIT)
    }

#undef XMX_DECODE_DISPATCH
#undef XMX_DECODE_LAUNCH
#undef XMX_DECODE_RUN
}
