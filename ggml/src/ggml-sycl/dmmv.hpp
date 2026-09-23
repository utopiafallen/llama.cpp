//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_DMMV_HPP
#define GGML_SYCL_DMMV_HPP

#include "common.hpp"


void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
bool ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    enum ggml_type src0_type, enum ggml_glu_op glu_op,
    const void * vx, const void * vgate, const float * y,
    float * dst, int ncols, int nrows, dpct::queue_ptr stream);

// ESIMD q6_K DMMV variant with a residual-add store epilogue for the {mul_mat, add} fusion
void ggml_sycl_q6_k_dmmv_reorder_esimd_add(const void * vx, const float * y, const float * res, float * dst,
                                           const int ncols, const int nrows, dpct::queue_ptr stream);

// small-batch (M=2..8) eSIMD Q6_K GEMV for the MTP verify; f32 activation
void dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd_m_dispatch(const void *vx, const float *y,
                                                                 float *dst, const int ncols, const int nrows,
                                                                 const int M, const int dst_col_stride,
                                                                 dpct::queue_ptr stream);

#ifdef GGML_SYCL_F16
// fp16 variant: MAC in half (lower register pressure, more WGs per EU)
void dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd_m_f16_dispatch(const void *vx, const float *y,
                                                                     float *dst, const int ncols, const int nrows,
                                                                     const int M, const int dst_col_stride,
                                                                     dpct::queue_ptr stream);

void dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd_m_f16_dispatch(const void *vx, const float *y,
                                                                    float *dst, const int ncols, const int nrows,
                                                                    const int M, const int dst_col_stride,
                                                                    dpct::queue_ptr stream);

void dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd_m_f16_dispatch(const void *vx, const float *y,
                                                                    float *dst, const int ncols, const int nrows,
                                                                    const int M, const int dst_col_stride,
                                                                    dpct::queue_ptr stream);
#endif // GGML_SYCL_F16

// small-batch (M=2..8) eSIMD Q5_K GEMV for the MTP verify; f32 activation
void dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd_m_dispatch(const void *vx, const float *y,
                                                                float *dst, const int ncols, const int nrows,
                                                                const int M, const int dst_col_stride,
                                                                dpct::queue_ptr stream);

// small-batch (M=2..8) eSIMD Q8_0 GEMV (lm_head) for the MTP verify; f32 activation
void dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd_m_dispatch(const void *vx, const float *y,
                                                                float *dst, const int ncols, const int nrows,
                                                                const int M, const int dst_col_stride,
                                                                dpct::queue_ptr stream);

#endif // GGML_SYCL_DMMV_HPP
