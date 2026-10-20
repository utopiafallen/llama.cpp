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

#ifndef GGML_SYCL_FATTN_XMX_DECODE_HPP
#define GGML_SYCL_FATTN_XMX_DECODE_HPP

#include "common.hpp"

// Short-sequence (n_q 1..5) flash attention that computes the QK^T and PV matmuls with the
// Intel XMX matrix engine (SYCL joint_matrix) instead of the EU FMA loop: single-token
// decode (n_q = 1) and MTP verify (n_q = 2..5, GGML_SYCL_FA_XMX_VERIFY). Each work-group
// handles one KV split and one KV head, batching all gqa queries (and query positions)
// into a single XMX tile so the K/V row is loaded once and reused across the group.
// QK^T uses dual-plane int8 over the SoA q8_0 KV cache (required; other KV layouts fall
// back to the tile path). A two-phase kernel merges the per-split partials (flash-decoding).
// See fattn-xmx-decode.cpp.
void ggml_sycl_flash_attn_ext_xmx_decode(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// True when the XMX decode kernel can handle this flash_attn_ext op.
bool ggml_sycl_flash_attn_ext_xmx_decode_supported(int device, const ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_XMX_DECODE_HPP
