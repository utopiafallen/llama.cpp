//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MMVQ_XMX_GEMM_HPP
#define GGML_SYCL_MMVQ_XMX_GEMM_HPP

#include "common.hpp"

// Small-M (2..8) GEMM on the XMX matrix engine for the reordered MMVQ path:
// weight tiles are dequanted from the SoA reorder planes into f16 and the MAC
// runs on joint_matrix, replacing the dp4a issue-bound MMVQ inner loop.
// Returns true when the op was fully handled (the caller must not fall through).
bool ggml_sycl_op_mul_mat_xmx_gemm(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                                   const ggml_tensor * src1, ggml_tensor * dst);

#endif // GGML_SYCL_MMVQ_XMX_GEMM_HPP
