#pragma once

#include "common.hpp"

#define GGML_SYCL_MAX_CONV_STATE_BATCH 5

struct ggml_sycl_batched_cpy_params {
    const float * srcs[GGML_SYCL_MAX_CONV_STATE_BATCH];
    float *       dsts[GGML_SYCL_MAX_CONV_STATE_BATCH];
    int           n_copies;
    int           rows;
    int           cols;
    int           stride_row;
    int           stride_col;
};

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_batched_conv_state_cpy(ggml_backend_sycl_context & ctx,
                                         const ggml_sycl_batched_cpy_params & params);
