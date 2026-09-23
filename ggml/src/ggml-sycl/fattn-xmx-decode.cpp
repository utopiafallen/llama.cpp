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
#include <cstdlib>
#include <algorithm>

namespace mx = sycl::ext::oneapi::experimental::matrix;
using mx::use;
using mx::layout;
namespace syclex = sycl::ext::oneapi::experimental;

// KV positions per work-group split (multiple of the XMX N tile).
#define XMX_DECODE_SPLIT 128

using xmp_g_h = sycl::multi_ptr<sycl::half, sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
using xmp_g_f = sycl::multi_ptr<float,  sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
using xmp_l_h = sycl::multi_ptr<sycl::half, sycl::access::address_space::local_space,  sycl::access::decorated::legacy>;
using xmp_l_f = sycl::multi_ptr<float,  sycl::access::address_space::local_space,  sycl::access::decorated::legacy>;

// One work-group (one 16-lane sub_group) handles one KV split and one KV head.
// MQ query positions are batched into M XMX rows:
//   NCHUNK==1: row r = (q = r/GQA, qi = r%GQA), packed densely (M may pad beyond ROWS)
//   NCHUNK==2: each A-chunk qc holds one query position's GQA rows at offset qc*M
// All LDS buffers and the per-split global partial storage use row stride PSTRIDE
// (>= M + padding) so full-tile joint_matrix_store writes never cross block bounds.
// D is fixed at 256 by the dispatch gate; SPLIT/MQ/NCHUNK are compile-time.
template <int M, int GQA, int SPLIT, int MQ, int NCHUNK, int QG = 0>
static void xmx_decode_main(
        const float * __restrict__ Q,
        const sycl::half * __restrict__ Q16g, // [n_wg_heads][PSTRIDE][D] f16 A-matrix scratch (QG=1 only)
        const sycl::half * __restrict__ K,
        const sycl::half * __restrict__ V,
        const char  * __restrict__ K_q8,
        const char  * __restrict__ V_q8,
        const sycl::half * __restrict__ mask,
        float * __restrict__ partial_O,
        float * __restrict__ partial_m,
        float * __restrict__ partial_l,
        const int n_kv, const int n_kv_heads, const int n_q_heads, const int n_splits,
        const int q_pos_stride, const int q_head_stride, const int k_pos_stride, const int k_head_stride,
        const int v_pos_stride, const int v_head_stride, const int mask_head_stride, const int mask_ne1,
        const int k_pos_stride_b, const int k_head_stride_b,
        const int v_pos_stride_b, const int v_head_stride_b,
        const float scale, const bool q8_input, const int f16_lds, const bool q8_soa, const int ne_row_b,
        const int gqa_r, const int n_hg, // real GQA ratio and head-groups per kv head (1 = dense)
        const bool p1only, // diagnostic: skip the PV section (QK+softmax cost probe), output invalid
        const sycl::nd_item<3> & it) {
    constexpr int D    = 256;
    constexpr int ROWS = GQA * MQ; // live rows
    constexpr int PSTRIDE = NCHUNK == 2 ? 2*M : M; // storage row stride (>= live extent)
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

    // QG=1: no Q16 in LDS (A matrix comes from the global f16 scratch) - frees PSTRIDE*D*2B for
    // higher occupancy at 16-row configs (see the SPLIT/LDS cliff notes in the b70-decode-perf skill)
    constexpr int LDS_BYTES = (QG == 0 ? PSTRIDE*D*2 : 0) + PSTRIDE*SPLIT*4 + PSTRIDE*SPLIT*2 + 16*16*2;
    syclex::work_group_static<char[LDS_BYTES]> lsm;
    sycl::half * Q16    = (sycl::half *)&lsm;                    // [PSTRIDE][D] (QG=0 only)
    float    * scores  = QG == 0 ? (float *)(Q16 + PSTRIDE*D) : (float *)&lsm; // [PSTRIDE][SPLIT]
    sycl::half * P16   = (sycl::half *)(scores + PSTRIDE*SPLIT); // [PSTRIDE][SPLIT]
    sycl::half * tile_buf = (sycl::half *)(P16 + PSTRIDE*SPLIT); // [16][16] staging

    // Q F32 -> F16 into LDS, all chunks at once (skipped with QG=1: A comes from global scratch)
    if constexpr (QG == 0) {
        for (int i = lane; i < PSTRIDE*D; i += 16) {
            const int r = i / D, dim = i % D;
            float val = 0.0f;
            if constexpr (NCHUNK == 2) {
                const int qc = r / M, qi = r % M;
                if (qi < GQA) {
                    val = Q[qc*q_pos_stride + (qh0 + qi)*q_head_stride + dim];
                }
            } else if constexpr (MQ > 1) {
                if (r < ROWS) {
                    const int q = r / GQA, qi = r % GQA;
                    val = Q[q*q_pos_stride + (qh0 + qi)*q_head_stride + dim];
                }
            } else {
                val = Q[(qh0 + r)*q_head_stride + dim];
            }
            Q16[i] = (sycl::half) val;
        }
        sg.barrier();
    }

    mx::joint_matrix<sycl::sub_group, sycl::half, use::a, M, 16, layout::row_major> A_jm;
    mx::joint_matrix<sycl::sub_group, sycl::half, use::b, 16, 16, layout::col_major> B_k;
    mx::joint_matrix<sycl::sub_group, float, use::accumulator, M, 16> C_jm[NCHUNK];

    // A-chunk base row in the [PSTRIDE] buffers
    const auto a_row = [](int qc) { return NCHUNK == 2 ? qc*M : 0; };

    // 2. QK^T: scores = Q @ K^T, per 16-pos chunk
    for (int c = 0; c < SPLIT/16; c++) {
        if constexpr (NCHUNK == 2) {
            mx::joint_matrix_fill(sg, C_jm[0], 0.0f);
            mx::joint_matrix_fill(sg, C_jm[1], 0.0f);
        } else {
            mx::joint_matrix_fill(sg, C_jm[0], 0.0f);
        }
        // SoA: this lane's position row and its head scales (all 8 blocks) for the dc loop
        const char * k_row = nullptr;
        sycl::vec<sycl::half, 8> k_scv;
        if (q8_input && q8_soa) {
            k_row = K_q8 + (pos_base + c*16 + lane)*k_pos_stride_b;
            // per-head segment: [D qs][D/32 half scales]; scales start at seg_start + D
            k_scv = *(const sycl::vec<sycl::half, 8> *)(k_row + kv_real*(D + 16) + D);
        }
        for (int dc = 0; dc < D/16; dc++) {
            const int dim0 = dc * 16;
            if (q8_input) {
                if (q8_soa) {
                    const int8_t * qs = (const int8_t *)(k_row + kv_real*(D + 16) + dim0);
                    const sycl::vec<int8_t, 16> qv = *(const sycl::vec<int8_t, 16> *)qs;
                    const float s = (float) k_scv[dc/2];
                    #pragma unroll
                    for (int d = 0; d < 16; d++) {
                        tile_buf[d + lane*16] = (sycl::half)((float)qv[d] * s);
                    }
                } else {
                    // Per-tile Q8_0 dequant: each lane handles one position
                    const int pos = pos_base + c*16 + lane;
                    const int blk  = dim0 / 32;
                    const int off  = dim0 % 32;
                    const char * blk_ptr = K_q8 + kv_real*k_head_stride_b + pos*k_pos_stride_b + blk*34;
                    const sycl::half sc = *(const sycl::half *)blk_ptr;
                    const float s = (float)sc;
                    const int8_t * qs = (const int8_t *)(blk_ptr + 2);
                    #pragma unroll
                    for (int d = 0; d < 16; d++) {
                        tile_buf[d + lane*16] = (sycl::half)((float)qs[off+d] * s);
                    }
                }
            } else if (f16_lds == 0 && NCHUNK == 1) {
                // f16 K tile: direct global B load for the single-chunk path
                sg.barrier();
                mx::joint_matrix_load(sg, B_k,
                    xmp_g_h((sycl::half *) (K + kv_real*k_head_stride + (pos_base + c*16)*k_pos_stride + dc*16)),
                    k_pos_stride);
                if constexpr (QG == 0) {
                    mx::joint_matrix_load(sg, A_jm, xmp_l_h(Q16 + a_row(0)*D + dc*16), D);
                } else {
                    mx::joint_matrix_load(sg, A_jm, xmp_g_h((sycl::half *)(Q16g + (size_t) kv_head*PSTRIDE*D + a_row(0)*D + dc*16)), D);
                }
                mx::joint_matrix_mad(sg, C_jm[0], A_jm, B_k, C_jm[0]);
                continue;
            } else {
                // f16 K staged through LDS (diagnostic path and NCHUNK==2 source)
                const sycl::half * krow = K + kv_real*k_head_stride + (pos_base + c*16 + lane)*k_pos_stride + dc*16;
                #pragma unroll
                for (int d = 0; d < 16; d++) {
                    tile_buf[d + lane*16] = krow[d];
                }
            }
            sg.barrier();
            if constexpr (NCHUNK == 2) {
                // B staged in LDS, reloaded per A-chunk (register reuse across mads unproven)
                #pragma unroll
                for (int qc = 0; qc < 2; qc++) {
                    mx::joint_matrix_load(sg, A_jm, xmp_l_h(Q16 + a_row(qc)*D + dc*16), D);
                    mx::joint_matrix_load(sg, B_k, xmp_l_h(tile_buf), 16);
                    mx::joint_matrix_mad(sg, C_jm[qc], A_jm, B_k, C_jm[qc]);
                }
            } else {
                mx::joint_matrix_load(sg, B_k, xmp_l_h(tile_buf), 16);
                if constexpr (QG == 0) {
                    mx::joint_matrix_load(sg, A_jm, xmp_l_h(Q16 + a_row(0)*D + dc*16), D);
                } else {
                    mx::joint_matrix_load(sg, A_jm, xmp_g_h((sycl::half *)(Q16g + (size_t) kv_head*PSTRIDE*D + a_row(0)*D + dc*16)), D);
                }
                mx::joint_matrix_mad(sg, C_jm[0], A_jm, B_k, C_jm[0]);
            }
        }
        if constexpr (NCHUNK == 2) {
            #pragma unroll
            for (int qc = 0; qc < 2; qc++) {
                mx::joint_matrix_store(sg, C_jm[qc], xmp_l_f(scores + (a_row(qc)*16 + c*16)), SPLIT, layout::row_major);
            }
        } else {
            mx::joint_matrix_store(sg, C_jm[0], xmp_l_f(scores + (a_row(0)*16 + c*16)), SPLIT, layout::row_major);
        }
        sg.barrier();
        // scale + causal + mask -> -FLT_MAX (each element handled by one lane)
        for (int idx = lane; idx < PSTRIDE*16; idx += 16) {
            const int r = idx / 16, p = idx % 16;
            float & sref = scores[r*SPLIT + c*16 + p];
            if constexpr (NCHUNK == 2) {
                if (r % M >= GQA) { sref = -FLT_MAX; continue; }
            } else if constexpr (MQ > 1) {
                if (r >= ROWS)      { sref = -FLT_MAX; continue; }
            }
            const int pos = pos_base + c*16 + p;
            const int q   = NCHUNK == 2 ? r / M : (MQ > 1 ? r / GQA : 0);
            float s = sref * scale;
            if (pos >= std::min(pos_end, n_kv - MQ + q + 1)) {
                s = -FLT_MAX;
            } else if (mask != nullptr) {
                const int qi = NCHUNK == 2 ? r % M : (MQ > 1 ? r % GQA : r);
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
        const int q  = MQ > 1 ? r / GQA : 0;
        const int qi = MQ > 1 ? r % GQA : r;
        const int rr = NCHUNK == 2 ? q*M + qi : r; // storage row (chunk-padded)
        float pm = -FLT_MAX;
        for (int pos = lane; pos < SPLIT; pos += 16) pm = std::max(pm, scores[rr*SPLIT + pos]);
        const float m = sycl::reduce_over_group(sg, pm, sycl::maximum<float>());
        float ps = 0.0f;
        for (int pos = lane; pos < SPLIT; pos += 16) {
            const float e = exp2f((scores[rr*SPLIT + pos] - m) * 1.4426950408889634f);
            P16[rr*SPLIT + pos] = (sycl::half) e;
            ps += e;
        }
        const float l = sycl::reduce_over_group(sg, ps, sycl::plus<float>());
        if (lane == 0) {
            // one PSTRIDE-row block per (split, virtual kv head)
            const size_t base = ((size_t) split * n_wg_heads + kv_head) * PSTRIDE;
            partial_m[base + rr] = m;
            partial_l[base + rr] = l;
        }
    }
    sg.barrier();

    // 4. PV: O = P @ V, per 16-dim chunk (skipped by the P1ONLY diagnostic)
    if (!p1only) {
    mx::joint_matrix<sycl::sub_group, sycl::half, use::b, 16, 16, layout::row_major> B_v;
    mx::joint_matrix<sycl::sub_group, float, use::accumulator, M, 16> O_jm[NCHUNK];
    for (int dc = 0; dc < D/16; dc++) {
        if constexpr (NCHUNK == 2) {
            mx::joint_matrix_fill(sg, O_jm[0], 0.0f);
            mx::joint_matrix_fill(sg, O_jm[1], 0.0f);
        } else {
            mx::joint_matrix_fill(sg, O_jm[0], 0.0f);
        }
        const int dim0 = dc * 16;
        for (int c = 0; c < SPLIT/16; c++) {
            if (q8_input) {
                if (q8_soa) {
                    const char * v_row = V_q8 + (pos_base + c*16 + lane)*v_pos_stride_b;
                    const int8_t * qs = (const int8_t *)(v_row + kv_real*(D + 16) + dim0);
                    const sycl::vec<int8_t, 16> qv = *(const sycl::vec<int8_t, 16> *)qs;
                    const sycl::vec<sycl::half, 8> v_scv = *(const sycl::vec<sycl::half, 8> *)(v_row + kv_real*(D + 16) + D);
                    const float s = (float) v_scv[dc/2];
                    #pragma unroll
                    for (int d = 0; d < 16; d++) {
                        tile_buf[lane*16 + d] = (sycl::half)((float)qv[d] * s);
                    }
                } else {
                    const int pos = pos_base + c*16 + lane;
                    const int blk  = dim0 / 32;
                    const int off  = dim0 % 32;
                    const char * blk_ptr = V_q8 + kv_real*v_head_stride_b + pos*v_pos_stride_b + blk*34;
                    const sycl::half sc = *(const sycl::half *)blk_ptr;
                    const float s = (float)sc;
                    const int8_t * qs = (const int8_t *)(blk_ptr + 2);
                    #pragma unroll
                    for (int d = 0; d < 16; d++) {
                        tile_buf[lane*16 + d] = (sycl::half)((float)qs[off+d] * s);
                    }
                }
            } else if (f16_lds == 0 && NCHUNK == 1) {
                // f16 V tile: direct global B load for the single-chunk path
                sg.barrier();
                mx::joint_matrix_load(sg, B_v,
                    xmp_g_h((sycl::half *) (V + kv_real*v_head_stride + (pos_base + c*16)*v_pos_stride + dc*16)),
                    v_pos_stride);
                mx::joint_matrix_load(sg, A_jm, xmp_l_h(P16 + a_row(0)*SPLIT + c*16), SPLIT);
                mx::joint_matrix_mad(sg, O_jm[0], A_jm, B_v, O_jm[0]);
                continue;
            } else {
                // f16 V staged through LDS (diagnostic path and NCHUNK==2 source)
                const sycl::half * vrow = V + kv_real*v_head_stride + (pos_base + c*16 + lane)*v_pos_stride + dc*16;
                #pragma unroll
                for (int d = 0; d < 16; d++) {
                    tile_buf[lane*16 + d] = vrow[d];
                }
            }
            sg.barrier();
            if constexpr (NCHUNK == 2) {
                #pragma unroll
                for (int qc = 0; qc < 2; qc++) {
                    mx::joint_matrix_load(sg, A_jm, xmp_l_h(P16 + a_row(qc)*SPLIT + c*16), SPLIT);
                    mx::joint_matrix_load(sg, B_v, xmp_l_h(tile_buf), 16);
                    mx::joint_matrix_mad(sg, O_jm[qc], A_jm, B_v, O_jm[qc]);
                }
            } else {
                mx::joint_matrix_load(sg, B_v, xmp_l_h(tile_buf), 16);
                mx::joint_matrix_load(sg, A_jm, xmp_l_h(P16 + a_row(0)*SPLIT + c*16), SPLIT);
                mx::joint_matrix_mad(sg, O_jm[0], A_jm, B_v, O_jm[0]);
            }
        }
        if constexpr (NCHUNK == 2) {
            #pragma unroll
            for (int qc = 0; qc < 2; qc++) {
                mx::joint_matrix_store(sg, O_jm[qc],
                    xmp_g_f(partial_O + (((size_t)split * n_wg_heads + kv_head) * PSTRIDE
                                         + a_row(qc))*D + dc*16),
                    D, layout::row_major);
            }
        } else {
            mx::joint_matrix_store(sg, O_jm[0],
                xmp_g_f(partial_O + (((size_t)split * n_wg_heads + kv_head) * PSTRIDE
                                     + a_row(0))*D + dc*16),
                D, layout::row_major);
        }
    }
    } // !p1only
}

// Combine the per-split partials (flash-decoding merge) into the final output.
// out[q][q_head][dim] = sum_split e^{m_s-M} O_s[dim] / sum_split e^{m_s-M} l_s, M = max_s m_s.
// Per-split storage is local: p_stride rows, row rr = q*rows_per_q + (h % gqa)
// (dense GQA packing when the tile fits all rows, chunk-padded otherwise).
static void xmx_decode_combine(
        const float * __restrict__ partial_O,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        float * __restrict__ out,
        const int n_q_heads, const int n_kv_heads, const int MQ, const int m_tile, const int n_hg,
        const int D, const int n_splits, const int p_stride,
        const int out_head_stride, const int out_pos_stride, const sycl::nd_item<3> & it) {
    const int q_head = it.get_group(0);
    const int dim    = it.get_group(1)*16 + it.get_local_id(2);
    if (dim >= D) {
        return;
    }
    const int gqa   = n_q_heads / n_kv_heads;
    const int gqa_a = (n_hg > 1) ? gqa / n_hg : gqa; // heads per WG block
    const int rows_per_q = (m_tile >= gqa_a * MQ) ? gqa_a : m_tile;
    const int q = q_head / n_q_heads;
    const int h = q_head % n_q_heads;
    // per-split layout: n_kv_heads*n_hg blocks of p_stride rows; this head's block is (kv, group)
    const int kv  = h / gqa;
    const int hg  = (h % gqa) / gqa_a;
    const int rr  = q * rows_per_q + (h % gqa_a); // row within the WG block
    const size_t blk = ((size_t) kv * n_hg + hg) * p_stride;
    const float * m_s = partial_m + blk + rr;
    const float * l_s = partial_l + blk + rr;
    const float * O_s = partial_O + (blk + rr) * D;
    const size_t sp = (size_t) n_kv_heads * n_hg * p_stride; // rows per split
    float M = -FLT_MAX;
    for (int s = 0; s < n_splits; s++) {
        M = std::max(M, m_s[s * sp]);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        const float w = exp2f((m_s[s * sp] - M) * 1.4426950408889634f);
        den += w * l_s[s * sp];
        num += w * O_s[s * sp * D + dim];
    }
    out[(size_t) q * (MQ > 1 ? out_pos_stride : 0) + (size_t) h * out_head_stride + dim] = num / den;
}

// Vectorized combine: each lane owns VEC contiguous dims and loads/stores them as one vector
// per split - 4x the bytes in flight per instruction vs the scalar version (DRAM saturation at
// long context, where the partials stream is read exactly once).
template<int VEC>
static void xmx_decode_combine_vec(
        const float * __restrict__ partial_O,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        float * __restrict__ out,
        const int n_q_heads, const int n_kv_heads, const int MQ, const int m_tile, const int n_hg,
        const int D, const int n_splits, const int p_stride,
        const int out_head_stride, const int out_pos_stride, const sycl::nd_item<3> & it) {
    const int q_head = it.get_group(0);
    const int dim0   = it.get_group(1)*(16*VEC) + it.get_local_id(2)*VEC;
    if (dim0 >= D) {
        return;
    }
    const int gqa   = n_q_heads / n_kv_heads;
    const int gqa_a = (n_hg > 1) ? gqa / n_hg : gqa; // heads per WG block
    const int rows_per_q = (m_tile >= gqa_a * MQ) ? gqa_a : m_tile;
    const int q = q_head / n_q_heads;
    const int h = q_head % n_q_heads;
    const int kv  = h / gqa;
    const int hg  = (h % gqa) / gqa_a;
    const int rr  = q * rows_per_q + (h % gqa_a);
    const size_t blk = ((size_t) kv * n_hg + hg) * p_stride;
    const float * m_s = partial_m + blk + rr;
    const float * l_s = partial_l + blk + rr;
    const float * O_s = partial_O + (blk + rr) * D;
    const size_t sp = (size_t) n_kv_heads * n_hg * p_stride; // rows per split
    float M = -FLT_MAX;
    for (int s = 0; s < n_splits; s++) {
        M = std::max(M, m_s[s * sp]);
    }
    sycl::vec<float, VEC> num_v = sycl::vec<float, VEC>(0.0f);
    float den = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        const float w = exp2f((m_s[s * sp] - M) * 1.4426950408889634f);
        den += w * l_s[s * sp];
        const sycl::vec<float, VEC> o = *(const sycl::vec<float, VEC> *)(O_s + s * sp * D + dim0);
        for (int v = 0; v < VEC; v++) {
            num_v[v] += w * o[v];
        }
    }
    float * out_p = out + (size_t) q * (MQ > 1 ? out_pos_stride : 0) + (size_t) h * out_head_stride + dim0;
    for (int v = 0; v < VEC; v++) {
        if (dim0 + v < D) {
            out_p[v] = num_v[v] / den;
        }
    }
}

// Two-phase combine (two launches, no cross-kernel sync needed): phase A chunks the split
// range into n_chunks independent merge streams, each computing a chunk-level m/l/O partial
// (flash-decoding associativity: exp2(m_s-M) = exp2(m_s-m_c)*exp2(m_c-M)); phase B merges the
// chunk partials. The single-phase loop is serial over all splits with few threads - at long
// context it only reaches ~200 GB/s on the partials stream; chunking raises in-flight bytes
// until the read saturates DRAM. Intermediates: [n_chunks][n_q_rows][D] f32 O + m,l.
static void xmx_decode_combine2a(
        const float * __restrict__ partial_O,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        float * __restrict__ inter_O,      // [n_chunks][n_q_rows][D]
        float * __restrict__ inter_m,      // [n_chunks][n_q_rows]
        float * __restrict__ inter_l,      // [n_chunks][n_q_rows]
        const int n_q_heads, const int n_kv_heads, const int MQ, const int m_tile, const int n_hg,
        const int D, const int n_splits, const int p_stride, const int n_chunks,
        const sycl::nd_item<3> & it) {
    // grid: (n_q_rows * n_chunks, D/16, 16); row index spans all output rows (heads x MQ)
    const int n_rows = n_q_heads * MQ;
    const int q_head = it.get_group(0) % n_rows;
    const int chunk  = it.get_group(0) / n_rows;
    const int dim    = it.get_group(1)*16 + it.get_local_id(2);
    if (dim >= D) {
        return;
    }
    const int gqa   = n_q_heads / n_kv_heads;
    const int gqa_a = (n_hg > 1) ? gqa / n_hg : gqa; // heads per WG block
    const int rows_per_q = (m_tile >= gqa_a * MQ) ? gqa_a : m_tile;
    const int h  = q_head % n_q_heads;
    const int kv = h / gqa;
    const int hg = (h % gqa) / gqa_a;
    const int rr = (q_head / n_q_heads) * rows_per_q + (h % gqa_a);
    const size_t blk = ((size_t) kv * n_hg + hg) * p_stride;
    const float * m_s = partial_m + blk + rr;
    const float * l_s = partial_l + blk + rr;
    const float * O_s = partial_O + (blk + rr) * D;
    const size_t sp = (size_t) n_kv_heads * n_hg * p_stride; // rows per split
    const int csz = (n_splits + n_chunks - 1) / n_chunks;
    const int s0 = chunk * csz;
    const int s1 = std::min(s0 + csz, n_splits);
    float Mc = -FLT_MAX;
    for (int s = s0; s < s1; s++) {
        Mc = std::max(Mc, m_s[s * sp]);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = s0; s < s1; s++) {
        const float w = exp2f((m_s[s * sp] - Mc) * 1.4426950408889634f);
        den += w * l_s[s * sp];
        num += w * O_s[s * sp * D + dim];
    }
    inter_m[chunk*n_q_heads + q_head] = Mc;
    inter_l[chunk*n_q_heads + q_head] = den;
    inter_O[((size_t) chunk*n_q_heads + q_head)*D + dim] = num;
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
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; c++) {
        M = std::max(M, inter_m[c*n_q_heads + q_head]);
    }
    float num = 0.0f, den = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        const float w = exp2f((inter_m[c*n_q_heads + q_head] - M) * 1.4426950408889634f);
        den += w * inter_l[c*n_q_heads + q_head];
        num += w * inter_O[((size_t) c*n_q_heads + q_head)*D + dim];
    }
    out[(size_t) (q_head / n_q_heads) * (MQ > 1 ? out_pos_stride : 0) + (size_t) h * out_head_stride + dim] = num / den;
}

static int xmx_decode_verify2() {
    // Q=2 (MTP verify) uses XMX by default; GGML_SYCL_XMX_VERIFY2=0 forces the tile fallback.
    static int v = ggml_sycl_get_env("GGML_SYCL_XMX_VERIFY2", 2);
    return v;
}
static int xmx_decode_verify3() {
    // Q=3..5 (deep MTP verify): GQA head-group split into M=16 tiles, A-matrix in global
    // f16 scratch. Default on 2026-09-21: per-step graph cost at or below the tile FA at
    // 8K/37K/152K q8 (FA sum 3.8 vs 5.4 ms/graph @8K, 45.6 vs 47.7 @152K), wall parity.
    // GGML_SYCL_XMX_VERIFY3=0 forces the tile fallback.
    static int v = ggml_sycl_get_env("GGML_SYCL_XMX_VERIFY3", 1);
    return v;
}
static int xmx_decode_split_env() {
    // SPLIT defaults to 64 for the M=16 verify variant (measured optimum at Q=2),
    // 128 elsewhere.
    static int v = -1;
    if (v < 0) {
        const char * e = std::getenv("GGML_SYCL_XMX_SPLIT");
        v = e ? std::atoi(e) : (xmx_decode_verify2() == 2 ? 64 : XMX_DECODE_SPLIT);
    }
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
    if (K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0) {
        return false;
    }
    if (V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0) {
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
    if (n_q_pos == 2) { // verify variants are opt-in
        const int v2 = xmx_decode_verify2();
        if (v2 == 0) {
            return false;
        }
        const int split = xmx_decode_split_env();
        if (split != 32 && split != 64 && split != XMX_DECODE_SPLIT) {
            return false;
        }
        // M=8/NCHUNK=2 supports SPLIT 64/128; M=16 single tile supports 32/64
        if (v2 == 1 && split != 64 && split != XMX_DECODE_SPLIT) return false;
        if (v2 == 2 && split > 64) return false;
    }
    if (n_q_pos >= 3) { // deep verify: head-group split, requires GQA rows to fit M=16
        if (xmx_decode_verify3() != 1) {
            return false;
        }
        if (gqa < 3 || gqa % 3 != 0) {
            return false;
        }
        const int split = xmx_decode_split_env();
        if (split != 64 && split != XMX_DECODE_SPLIT) {
            return false;
        }
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

    int SPLIT = MQ > 1 ? xmx_decode_split_env() : XMX_DECODE_SPLIT;
    if (MQ > 1 && SPLIT != 32 && SPLIT != 64 && SPLIT != XMX_DECODE_SPLIT) {
        SPLIT = XMX_DECODE_SPLIT;
    }

    dpct::queue_ptr stream = ctx.stream();
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    ggml_sycl_pool & pool = ctx.pool();
    const int n_q_rows = n_q_heads * MQ;

    const float * Q_h = (const float *) Q->data;

    const bool q8_input = (K->type == GGML_TYPE_Q8_0);
    const bool q8_soa   = q8_input && ggml_sycl_is_q8_0_soa(K);
    const int  ne_row_b = D * n_kv_heads; // int8 bytes per position row in the SoA qs region
    if (q8_soa) {
        static bool soa_logged = false;
        if (!soa_logged) {
            soa_logged = true;
            fprintf(stderr, "[SOA] XMX decode: reading per-row SoA Q8_0 KV (row_ne=%d, n_kv_heads=%d, D=%d)\n",
                    ne_row_b, n_kv_heads, D);
        }
    }
    const sycl::half * K_h = q8_input ? nullptr : (const sycl::half *) K->data;
    const sycl::half * V_h = q8_input ? nullptr : (const sycl::half *) V->data;
    const char  * K_q8_p  = q8_input ? (const char *) K->data : nullptr;
    const char  * V_q8_p  = q8_input ? (const char *) V->data : nullptr;

    const int k_pos_stride    = q8_input ? 0 : (int) (K->nb[1]  / sizeof(sycl::half));
    const int k_head_stride   = q8_input ? 0 : (int) (K->nb[2]  / sizeof(sycl::half));
    const int v_pos_stride    = q8_input ? 0 : (int) (V->nb[1]  / sizeof(sycl::half));
    const int v_head_stride   = q8_input ? 0 : (int) (V->nb[2]  / sizeof(sycl::half));
    const int k_pos_stride_b  = (int) K->nb[1];
    const int k_head_stride_b = (int) K->nb[2];
    const int v_pos_stride_b  = (int) V->nb[1];
    const int v_head_stride_b = (int) V->nb[2];

    const sycl::half * m_h = mask ? (const sycl::half *) mask->data : nullptr;
    float * out_f = (float *) dst->data;

    const int q_pos_stride    = (int) (Q->nb[1] / sizeof(float));
    const int q_head_stride   = (int) (Q->nb[2] / sizeof(float));
    const int mask_head_stride = mask ? (int) (mask->nb[1] / sizeof(sycl::half)) : 0;
    const int mask_ne1         = mask ? (int) (mask->ne[1]) : 1;
    const int out_head_stride = (int) (dst->nb[1] / sizeof(float));
    const int out_pos_stride  = (int) (dst->nb[2] / sizeof(float));

    static int xmx_dbg = ggml_sycl_get_env("GGML_SYCL_XMX_DECODE_DEBUG", 0);
    // head-group split for deep verify (MQ 3..5, gqa % 3 == 0): one M=16 tile per 3-head group
    const int n_hg_l = (MQ >= 3 && xmx_decode_verify3() == 1) ? gqa / 3 : 1;
    static bool v3_printed = false;
    if (MQ >= 3 && xmx_decode_verify3() == 1 && !v3_printed) {
        v3_printed = true;
        fprintf(stderr, "[XMQ-V3] deep verify: MQ=%d gqa=%d n_hg=%d M=16 NCHUNK=1 SPLIT=%d n_kv=%d n_kv_heads=%d q8_soa=%d\n",
                MQ, gqa, n_hg_l, SPLIT, n_kv, n_kv_heads, (int) q8_soa);
    }
    // Diagnostic: force f16 K/V tiles through LDS staging (1 = with barrier, 2 = no barrier)
    const int xmx_f16_lds = ggml_sycl_get_env("GGML_SYCL_XMX_FA_LDS_F16", 0);
    static bool xmx_dbg_printed = false;
    if (xmx_dbg && !xmx_dbg_printed) {
        xmx_dbg_printed = true;
        fprintf(stderr, "[XMX-DECODE] n_kv=%d n_kv_heads=%d n_q_heads=%d gqa=%d D=%d mask=%s "
                "mask_ne=%dx%d mask_nb=%lld/%lld stride_mask=%d MQ=%d SPLIT=%d v2=%d\n",
                n_kv, n_kv_heads, n_q_heads, gqa, D, mask ? "yes" : "no",
                mask ? (int) mask->ne[0] : 0, mask ? (int) mask->ne[1] : 0,
                mask ? (long long) mask->nb[0] : 0, mask ? (long long) mask->nb[1] : 0,
                mask_head_stride, MQ, SPLIT, xmx_decode_verify2());
    }

    const sycl::range<3> wg_local(1, 1, 16);

    // QG=1 A-matrix scratch for the GQA head-group verify path: Q pre-converted to f16 in the
    // per-WG block layout ([n_wg_heads][PSTRIDE][D]) so the main kernel loads A from global and
    // drops its PSTRIDE*D*2B LDS staging (occupancy cliff at 16 rows, see b70-decode-perf skill)
    ggml_sycl_pool_alloc<sycl::half> qg_scratch(pool);
    sycl::half * Q16g_p = nullptr;
    if (MQ >= 3 && xmx_decode_verify3() == 1) {
        const int gqa_r_l = 3; // V3 dispatches groups of 3 heads
        const size_t qg_rows = (size_t) n_kv_heads * n_hg_l * 16; // PSTRIDE = 16 for NCHUNK == 1
        qg_scratch.alloc(qg_rows * D);
        Q16g_p = qg_scratch.ptr;
        stream->parallel_for(sycl::range<1>((size_t) n_kv_heads * n_hg_l * 16 * (D / 16)),
            [=](sycl::item<1> it2) {
                const size_t i = it2.get_linear_id();
                const int dim0 = (int) (i % (D / 16)) * 16;
                const int r    = (int) ((i / (D / 16)) % 16);
                const int b    = (int) (i / ((D / 16) * 16));
                const int kv2  = b / n_hg_l, hg2 = b % n_hg_l;
                const int qh_base = kv2*gqa + hg2*gqa_r_l;
                sycl::vec<sycl::half, 16> out;
                for (int d = 0; d < 16; d++) {
                    float val = 0.0f;
                    if (r < gqa_r_l * MQ) { // live rows; the rest stays zero-padded
                        const int q = r / gqa_r_l, qi = r % gqa_r_l;
                        val = Q_h[q*q_pos_stride + (qh_base + qi)*q_head_stride + dim0 + d];
                    }
                    out[d] = sycl::half(val);
                }
                *(sycl::vec<sycl::half, 16> *)(Q16g_p + (size_t) b*16*D + r*D + dim0) = out;
            });
        SYCL_CHECK(0);
    }

    // Optional per-kernel wall timing (serializes the stream; debug only): [XMQ-T] lines on stderr
    static int xmx_t = ggml_sycl_get_env("GGML_SYCL_XMX_TIMING", 0);
    // Diagnostic: skip the PV section of the main kernel (QK+softmax-only wall, two-phase-P probe).
    // Output is INVALID while on - timing only.
    static int xmx_p1 = ggml_sycl_get_env("GGML_SYCL_XMX_P1ONLY", 0);
    const bool p1only = xmx_p1 != 0;
    static bool p1_warned = false;
    if (p1only && !p1_warned) {
        p1_warned = true;
        fprintf(stderr, "[XMQ-P1] P1ONLY diagnostic ON: FA output INVALID, main-kernel timing only\n");
    }
    // Combine kernel variant: default 4 = two-phase chunked merge (faster at long context:
    // combine 1.50->1.22ms/node @152K MQ=5, 0.55->0.17 @MQ=1, output-equivalent f32 reassociation;
    // wall A/B parity-to-better vs scalar 63.2 vs 64.4 avg). Env: 0 = scalar single-phase, 2 = vec4.
    static int xmx_c = ggml_sycl_get_env("GGML_SYCL_XMQ_COMBINE", 4);

    // M = XMX tile rows; NCHUNK = A-chunks per position chunk (2 when M < GQA*MQ)
    #define XMX_DECODE_RUN(SPL) \
        { \
            const int n_splits_l = (n_kv + SPL - 1) / SPL; \
            /* per-WG row stride in the partials: matches the kernel's PSTRIDE */ \
            const int p_stride_l = (MQ == 1) ? gqa : 16; \
            /* each split holds one p_stride_l-row block per (virtual) kv head */ \
            const size_t rows_l = (size_t) n_splits_l * n_kv_heads * n_hg_l * p_stride_l; \
            ggml_sycl_pool_alloc<float> pO(pool); \
            ggml_sycl_pool_alloc<float> pm(pool); \
            ggml_sycl_pool_alloc<float> pl(pool); \
            pO.alloc(rows_l * D); \
            pm.alloc(rows_l); \
            pl.alloc(rows_l); \
            const sycl::range<3> wg_global_l(n_splits_l, n_kv_heads * n_hg_l, 16); \
            float * pO_p = pO.ptr; float * pm_p = pm.ptr; float * pl_p = pl.ptr; \
            int64_t t_main0 = 0, t_main1 = 0, t_comb1 = 0; \
            if (xmx_t) { stream->wait(); t_main0 = ggml_time_us(); } \
            if (MQ >= 3 && xmx_decode_verify3() == 1) { \
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
                    case 4: if (xmx_decode_verify2() == 1) { XMX_DECODE_LAUNCH(8, 4, 2, SPL); } else { XMX_DECODE_LAUNCH(16, 4, 2, SPL); } break; \
                    case 5: if (xmx_decode_verify2() == 1) { XMX_DECODE_LAUNCH(8, 5, 2, SPL); } else { XMX_DECODE_LAUNCH(16, 5, 2, SPL); } break; \
                    case 6: if (xmx_decode_verify2() == 1) { XMX_DECODE_LAUNCH(8, 6, 2, SPL); } else { XMX_DECODE_LAUNCH(16, 6, 2, SPL); } break; \
                    case 7: if (xmx_decode_verify2() == 1) { XMX_DECODE_LAUNCH(8, 7, 2, SPL); } else { XMX_DECODE_LAUNCH(16, 7, 2, SPL); } break; \
                    case 8: if (xmx_decode_verify2() == 1) { XMX_DECODE_LAUNCH(8, 8, 2, SPL); } else { XMX_DECODE_LAUNCH(16, 8, 2, SPL); } break; \
                    default: GGML_ABORT("xmx decode verify: unsupported gqa %d", gqa); \
                } \
            } \
            if (xmx_t) { stream->wait(); t_main1 = ggml_time_us(); } \
            { \
                const sycl::range<3> c_local(1, 1, 16); \
                /* storage tile rows: matches the XMX_DECODE_LAUNCH(MM, ...) dispatch above */ \
                const int m_tile_l = (MQ == 1) ? gqa : ((MQ >= 3 && xmx_decode_verify3() == 1) ? 16 : (xmx_decode_verify2() == 1 ? 8 : 16)); \
                if ((xmx_c & 4) && n_splits_l >= 64) { \
                    /* two-phase: chunk the split range into independent merge streams */ \
                    const int n_chunks_l = std::min(64, (n_splits_l + 15) / 16); \
                    ggml_sycl_pool_alloc<float> iO(pool); \
                    ggml_sycl_pool_alloc<float> iml(pool); \
                    iO.alloc((size_t) n_chunks_l * n_q_rows * D); \
                    iml.alloc((size_t) n_chunks_l * n_q_rows * 2); \
                    float * iO_p = iO.ptr; float * iml_p = iml.ptr; \
                    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(n_q_rows * n_chunks_l, (D + 15) / 16, 16), c_local), \
                        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] { \
                            xmx_decode_combine2a(pO_p, pm_p, pl_p, iO_p, iml_p, iml_p + (size_t) n_chunks_l * n_q_rows, \
                                                 n_q_heads, n_kv_heads, MQ, m_tile_l, n_hg_l, \
                                                 D, n_splits_l, p_stride_l, n_chunks_l, it); \
                        }); \
                    SYCL_CHECK(0); \
                    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(n_q_rows, (D + 15) / 16, 16), c_local), \
                        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] { \
                            xmx_decode_combine2b(iO_p, iml_p, iml_p + (size_t) n_chunks_l * n_q_rows, out_f, \
                                                 n_q_heads, MQ, D, n_chunks_l, out_head_stride, out_pos_stride, it); \
                        }); \
                    SYCL_CHECK(0); \
                } else if (xmx_c & 2) { \
                    const sycl::range<3> c_global_v(n_q_rows, D / (16*4), 16); \
                    stream->parallel_for(sycl::nd_range<3>(c_global_v, c_local), \
                        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] { \
                            xmx_decode_combine_vec<4>(pO_p, pm_p, pl_p, out_f, n_q_heads, n_kv_heads, MQ, m_tile_l, n_hg_l, \
                                                      D, n_splits_l, p_stride_l, \
                                                      out_head_stride, out_pos_stride, it); \
                        }); \
                    SYCL_CHECK(0); \
                } else { \
                    const sycl::range<3> c_global(n_q_rows, (D + 15) / 16, 16); \
                    stream->parallel_for(sycl::nd_range<3>(c_global, c_local), \
                        [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] { \
                            xmx_decode_combine(pO_p, pm_p, pl_p, out_f, n_q_heads, n_kv_heads, MQ, m_tile_l, n_hg_l, \
                                               D, n_splits_l, p_stride_l, \
                                               out_head_stride, out_pos_stride, it); \
                        }); \
                    SYCL_CHECK(0); \
                } \
                if (xmx_t) { stream->wait(); t_comb1 = ggml_time_us(); } \
            } \
            if (xmx_t) { \
                fprintf(stderr, "[XMQ-T] MQ=%d gqa=%d n_hg=%d SPLIT=%d n_kv=%d n_splits=%d main=%.2fms combine=%.2fms partials_KB=%zu\n", \
                        MQ, gqa, n_hg_l, SPL, n_kv, n_splits_l, \
                        (t_main1 - t_main0) / 1000.0, (t_comb1 - t_main1) / 1000.0, rows_l * D / 256); \
            } \
        }

    #define XMX_DECODE_LAUNCH(MM, GQACT, MQCT, SPLITCT) \
        stream->parallel_for(sycl::nd_range<3>(wg_global_l, wg_local), \
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] { \
                xmx_decode_main<MM, GQACT, SPLITCT, MQCT, (MM >= GQACT*MQCT) ? 1 : 2, (MQCT >= 3) ? 1 : 0>(Q_h, Q16g_p, K_h, V_h, K_q8_p, V_q8_p, m_h, pO_p, pm_p, pl_p, \
                    n_kv, n_kv_heads, n_q_heads, n_splits_l, q_pos_stride, q_head_stride, k_pos_stride, \
                    k_head_stride, v_pos_stride, v_head_stride, mask_head_stride, mask_ne1, \
                    k_pos_stride_b, k_head_stride_b, v_pos_stride_b, v_head_stride_b, \
                    scale, q8_input, xmx_f16_lds, q8_soa, ne_row_b, gqa, n_hg_l, p1only, it); \
            }); \
        SYCL_CHECK(0)

    // SPLIT must be a compile-time template argument: dispatch on the env-selected value
    if (SPLIT == 32) {
        XMX_DECODE_RUN(32)
    } else if (SPLIT == 64) {
        XMX_DECODE_RUN(64)
    } else {
        XMX_DECODE_RUN(XMX_DECODE_SPLIT)
    }
    #undef XMX_DECODE_LAUNCH
    #undef XMX_DECODE_RUN
}
