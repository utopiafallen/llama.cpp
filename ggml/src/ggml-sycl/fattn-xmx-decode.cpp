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

namespace mx = sycl::ext::oneapi::experimental::matrix;
using mx::use;
using mx::layout;
namespace syclex = sycl::ext::oneapi::experimental;

// KV positions handled per work-group split (multiple of the XMX N tile).
#define XMX_DECODE_SPLIT 256

using xmp_g_h = sycl::multi_ptr<sycl::half, sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
using xmp_g_f = sycl::multi_ptr<float,  sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
using xmp_l_h = sycl::multi_ptr<sycl::half, sycl::access::address_space::local_space,  sycl::access::decorated::legacy>;
using xmp_l_f = sycl::multi_ptr<float,  sycl::access::address_space::local_space,  sycl::access::decorated::legacy>;

// One work-group (one 16-lane sub_group) handles one KV split and one KV head,
// batching all GQA queries into a single M=GQA XMX tile. Computes the per-split
// partial (O[GQA][D], m[GQA], l[GQA]) written to the partial buffers.
template <int GQA, int D, int SPLIT>
static void xmx_decode_main(
        const float * __restrict__ Q,
        const sycl::half * __restrict__ K,
        const sycl::half * __restrict__ V,
        const sycl::half * __restrict__ mask,
        float * __restrict__ partial_O,
        float * __restrict__ partial_m,
        float * __restrict__ partial_l,
        const int n_kv, const int n_kv_heads, const int n_q_heads, const int n_splits,
        const int q_head_stride, const int k_pos_stride, const int k_head_stride,
        const int v_pos_stride, const int v_head_stride, const int mask_head_stride, const int mask_ne1,
        const float scale, const sycl::nd_item<3> & it) {
    const int split   = it.get_group(0);
    const int kv_head = it.get_group(1);
    const int lane    = it.get_local_id(2);
    sycl::sub_group sg = it.get_sub_group();

    const int pos_base = split * SPLIT;
    const int pos_end  = std::min(pos_base + SPLIT, n_kv);

    constexpr int LDS_BYTES = GQA*D*2 + GQA*SPLIT*4 + GQA*SPLIT*2;
    syclex::work_group_static<char[LDS_BYTES]> lsm;
    sycl::half * Q16    = (sycl::half *)&lsm;              // [GQA][D]
    float    * scores  = (float *)(Q16 + GQA*D);           // [GQA][SPLIT]
    sycl::half * P16   = (sycl::half *)(scores + GQA*SPLIT); // [GQA][SPLIT]

    // 1. Q F32 -> F16 into LDS (row qi is global q_head kv_head*GQA+qi)
    for (int i = lane; i < GQA*D; i += 16) {
        const int qi = i / D, dim = i % D;
        Q16[i] = (sycl::half) Q[(kv_head*GQA + qi)*q_head_stride + dim];
    }
    sg.barrier();

    // 2. QK^T: scores[GQA][SPLIT] = Q[GQA][D] @ K[SPLIT][D]^T, per 16-pos chunk
    for (int c = 0; c < SPLIT/16; c++) {
        mx::joint_matrix<sycl::sub_group, sycl::half, use::a, GQA, 16, layout::row_major> A_jm;
        mx::joint_matrix<sycl::sub_group, sycl::half, use::b, 16, 16, layout::col_major> B_jm;
        mx::joint_matrix<sycl::sub_group, float, use::accumulator, GQA, 16> C_jm;
        mx::joint_matrix_fill(sg, C_jm, 0.0f);
        for (int dc = 0; dc < D/16; dc++) {
            mx::joint_matrix_load(sg, A_jm, xmp_l_h(Q16 + dc*16), D);
            mx::joint_matrix_load(sg, B_jm,
                xmp_g_h((sycl::half *) (K + kv_head*k_head_stride + (pos_base + c*16)*k_pos_stride + dc*16)),
                k_pos_stride);
            mx::joint_matrix_mad(sg, C_jm, A_jm, B_jm, C_jm);
        }
        mx::joint_matrix_store(sg, C_jm, xmp_l_f(scores + c*16), SPLIT, layout::row_major);
        sg.barrier();
        // scale + mask + OOB -> -FLT_MAX (each element handled by one lane)
        for (int idx = lane; idx < GQA*16; idx += 16) {
            const int qi = idx / 16, p = idx % 16;
            const int pos = pos_base + c*16 + p;
            float s = scores[qi*SPLIT + c*16 + p] * scale;
            if (pos >= pos_end) {
                s = -FLT_MAX;
            } else if (mask != nullptr) {
                const int qh = kv_head*GQA + qi;
                const int mi = mask_ne1 > 1 ? qh : 0; // dim-1 is query-pos (1 for decode); per-head only if present
                const float msk = (float) mask[pos + mi*mask_head_stride];
                s = msk < 0.0f ? -FLT_MAX : s + msk;
            }
            scores[qi*SPLIT + c*16 + p] = s;
        }
        sg.barrier();
    }

    // 3. softmax over the split, per query (unnormalized; cross-split scale in combine)
    float m[GQA], l[GQA];
    for (int qi = 0; qi < GQA; qi++) {
        float pm = -FLT_MAX;
        for (int pos = lane; pos < SPLIT; pos += 16) pm = std::max(pm, scores[qi*SPLIT + pos]);
        m[qi] = sycl::reduce_over_group(sg, pm, sycl::maximum<float>());
    }
    for (int qi = 0; qi < GQA; qi++) {
        float ps = 0.0f;
        for (int pos = lane; pos < SPLIT; pos += 16) {
            const float e = std::exp(scores[qi*SPLIT + pos] - m[qi]);
            P16[qi*SPLIT + pos] = (sycl::half) e;
            ps += e;
        }
        l[qi] = sycl::reduce_over_group(sg, ps, sycl::plus<float>());
    }
    if (lane < GQA) {
        partial_m[split*n_q_heads + kv_head*GQA + lane] = m[lane];
        partial_l[split*n_q_heads + kv_head*GQA + lane] = l[lane];
    }
    sg.barrier();

    // 4. PV: O[GQA][D] = P[GQA][SPLIT] @ V[SPLIT][D], per 16-dim chunk
    for (int dc = 0; dc < D/16; dc++) {
        mx::joint_matrix<sycl::sub_group, sycl::half, use::a, GQA, 16, layout::row_major> A_jm;
        mx::joint_matrix<sycl::sub_group, sycl::half, use::b, 16, 16, layout::row_major> B_jm;
        mx::joint_matrix<sycl::sub_group, float, use::accumulator, GQA, 16> O_jm;
        mx::joint_matrix_fill(sg, O_jm, 0.0f);
        for (int c = 0; c < SPLIT/16; c++) {
            mx::joint_matrix_load(sg, A_jm, xmp_l_h(P16 + c*16), SPLIT);
            mx::joint_matrix_load(sg, B_jm,
                xmp_g_h((sycl::half *) (V + kv_head*v_head_stride + (pos_base + c*16)*v_pos_stride + dc*16)),
                v_pos_stride);
            mx::joint_matrix_mad(sg, O_jm, A_jm, B_jm, O_jm);
        }
        mx::joint_matrix_store(sg, O_jm,
            xmp_g_f(partial_O + (size_t)(split*n_q_heads + kv_head*GQA)*D + dc*16),
            D, layout::row_major);
    }
}

// Combine the per-split partials (flash-decoding merge) into the final output.
// out[q_head][dim] = sum_split e^{m_s-M} O_s[dim] / sum_split e^{m_s-M} l_s, M = max_s m_s.
static void xmx_decode_combine(
        const float * __restrict__ partial_O,
        const float * __restrict__ partial_m,
        const float * __restrict__ partial_l,
        float * __restrict__ out,
        const int n_q_heads, const int D, const int n_splits,
        const int out_head_stride, const sycl::nd_item<3> & it) {
    const int q_head = it.get_group(0);
    const int dim    = it.get_group(1)*16 + it.get_local_id(2);
    if (dim >= D) {
        return;
    }
    const float * m_s = partial_m + q_head;      // stride n_q_heads per split
    const float * l_s = partial_l + q_head;
    const float * O_s = partial_O + (size_t) q_head * D;
    float M = -FLT_MAX;
    for (int s = 0; s < n_splits; s++) {
        M = std::max(M, m_s[(size_t) s * n_q_heads]);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        const float w = std::exp(m_s[(size_t) s * n_q_heads] - M);
        den += w * l_s[(size_t) s * n_q_heads];
        num += w * O_s[(size_t) s * n_q_heads * D + dim];
    }
    out[(size_t) q_head * out_head_stride + dim] = num / den;
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
    if (Q->ne[1] != 1) { // decode only
        return false;
    }
    if (Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) { // single batch
        return false;
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || Q->type != GGML_TYPE_F32) {
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
    // XMX M tile is 1..8, so gqa must fit
    const int gqa = (int) (Q->ne[2] / K->ne[2]);
    return gqa >= 1 && gqa <= 8;
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
    const int gqa          = n_q_heads / n_kv_heads;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int SPLIT   = XMX_DECODE_SPLIT;
    const int n_splits = (n_kv + SPLIT - 1) / SPLIT;

    dpct::queue_ptr stream = ctx.stream();
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    ggml_sycl_pool & pool = ctx.pool();
    ggml_sycl_pool_alloc<float> pO(pool);
    ggml_sycl_pool_alloc<float> pm(pool);
    ggml_sycl_pool_alloc<float> pl(pool);
    pO.alloc((size_t) n_splits * n_q_heads * D);
    pm.alloc((size_t) n_splits * n_q_heads);
    pl.alloc((size_t) n_splits * n_q_heads);

    const float * Q_h = (const float *) Q->data;
    const sycl::half * K_h = (const sycl::half *) K->data;
    const sycl::half * V_h = (const sycl::half *) V->data;
    const sycl::half * m_h = mask ? (const sycl::half *) mask->data : nullptr;
    float * out_f = (float *) dst->data;
    float * pO_p  = pO.ptr;
    float * pm_p  = pm.ptr;
    float * pl_p  = pl.ptr;

    const int q_head_stride   = (int) (Q->nb[2]  / sizeof(float));
    const int k_pos_stride    = (int) (K->nb[1]  / sizeof(sycl::half));
    const int k_head_stride   = (int) (K->nb[2]  / sizeof(sycl::half));
    const int v_pos_stride    = (int) (V->nb[1]  / sizeof(sycl::half));
    const int v_head_stride   = (int) (V->nb[2]  / sizeof(sycl::half));
    const int mask_head_stride = mask ? (int) (mask->nb[1] / sizeof(sycl::half)) : 0;
    const int mask_ne1         = mask ? (int) (mask->ne[1]) : 1;
    const int out_head_stride = (int) (dst->nb[1] / sizeof(float));

    static int xmx_dbg = ggml_sycl_get_env("GGML_SYCL_XMX_DECODE_DEBUG", 0);
    static bool xmx_dbg_printed = false;
    if (xmx_dbg && !xmx_dbg_printed) {
        xmx_dbg_printed = true;
        fprintf(stderr, "[XMX-DECODE] n_kv=%d n_kv_heads=%d n_q_heads=%d gqa=%d D=%d n_splits=%d mask=%s "
                "mask_ne=%dx%d mask_nb=%lld/%lld stride_mask=%d\n",
                n_kv, n_kv_heads, n_q_heads, gqa, D, n_splits, mask ? "yes" : "no",
                mask ? (int) mask->ne[0] : 0, mask ? (int) mask->ne[1] : 0,
                mask ? (long long) mask->nb[0] : 0, mask ? (long long) mask->nb[1] : 0,
                mask_head_stride);
    }

    const sycl::range<3> wg_local(1, 1, 16);
    const sycl::range<3> wg_global(n_splits, n_kv_heads, 16);

    // dispatch on gqa (XMX M tile is a compile-time size)
    #define XMX_DECODE_LAUNCH(G) \
        stream->parallel_for(sycl::nd_range<3>(wg_global, wg_local), \
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] { \
                xmx_decode_main<G, 256, SPLIT>(Q_h, K_h, V_h, m_h, pO_p, pm_p, pl_p, \
                    n_kv, n_kv_heads, n_q_heads, n_splits, q_head_stride, k_pos_stride, \
                    k_head_stride, v_pos_stride, v_head_stride, mask_head_stride, mask_ne1, scale, it); \
            }); \
        SYCL_CHECK(0)
    switch (gqa) {
        case 1: XMX_DECODE_LAUNCH(1); break;
        case 2: XMX_DECODE_LAUNCH(2); break;
        case 3: XMX_DECODE_LAUNCH(3); break;
        case 4: XMX_DECODE_LAUNCH(4); break;
        case 5: XMX_DECODE_LAUNCH(5); break;
        case 6: XMX_DECODE_LAUNCH(6); break;
        case 7: XMX_DECODE_LAUNCH(7); break;
        case 8: XMX_DECODE_LAUNCH(8); break;
        default: GGML_ABORT("xmx decode: unsupported gqa %d", gqa);
    }
    #undef XMX_DECODE_LAUNCH

    // combine
    {
        const sycl::range<3> c_local(1, 1, 16);
        const sycl::range<3> c_global(n_q_heads, (D + 15) / 16, 16);
        stream->parallel_for(sycl::nd_range<3>(c_global, c_local),
            [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                xmx_decode_combine(pO_p, pm_p, pl_p, out_f, n_q_heads, D, n_splits,
                                   out_head_stride, it);
            });
        SYCL_CHECK(0);
    }
}
