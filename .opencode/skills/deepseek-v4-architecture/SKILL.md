---
name: deepseek-v4-architecture
description: DeepSeek-V4 architecture reference for inference optimization work. Use when working on MoE prefill, attention, or any DeepSeek-V4 specific kernel changes in llama.cpp SYCL backend.
---

# DeepSeek-V4 Architecture Reference

Distilled from the DeepSeek-V4 technical report (arxiv 2606.19348). Focuses on inference-relevant details for SYCL/GGML optimization.

## Model Variants

Two production variants exist. All optimization work targets DeepSeek-V4-Flash unless stated otherwise.

### DeepSeek-V4-Flash (primary target)

| Parameter | Value |
|---|---|
| Total params | 284B |
| Activated params | 13B |
| Transformer layers | 43 |
| Hidden dim (d) | 4096 |

### DeepSeek-V4-Pro

| Parameter | Value |
|---|---|
| Total params | 1.6T |
| Activated params | 49B |
| Transformer layers | 61 |
| Hidden dim (d) | 7168 |

## MoE (Mixture of Experts)

All transformer layers use MoE. This is the dominant compute path during prefill.

### Expert configuration

| Parameter | Flash | Pro |
|---|---|---|
| Shared experts | 1 | 1 |
| Routed experts | 256 | 384 |
| top-k per token | 6 | 6 |
| Expert intermediate dim | 2048 | 3072 |

### Routing

- First 3 MoE layers: hash-based routing (deterministic, no learned gate)
- Remaining layers: learned routing via softmax gate
- Auxiliary-loss-free load balancing: bias update speed 0.001, balance loss weight 0.0001

### FFN structure

Standard SwiGLU per expert:
- Gate projection: input -> intermediate (fp8 or quantized)
- Up projection: input -> intermediate (fp8 or quantized)
- Activation: SiLU(gate) * up, with linear clamp [-10, 10]
- Down projection: intermediate -> hidden
- Per-expert scalar scale applied after K-loop accumulator

Weight layout (sglang FusedMoE convention, NK / N-major):
- w13 = [E, 2*I, H] -- gate + up fused
- w2 = [E, I, H] -- down

### MoE compute per token during prefill

For V4-Flash: 6 experts x (2 * 4096 * 2048 + 2048 * 4096) = 6 * 24.5M = ~147M MACs per token for MoE alone. This dominates the prefill FLOP budget.

## Attention

Hybrid attention: layers interleave CSA and HCA. Layers 1-2 are pure sliding window.

### CSA (Compressed Sparse Attention)

| Parameter | Flash | Pro |
|---|---|---|
| Compression rate (m) | 4 | 4 |
| Indexer query heads (n_h^I) | 64 | 64 |
| Indexer head dim (c^I) | 128 | 128 |
| Attention top-k | 512 | 1024 |
| Query heads (n_h) | 64 | 128 |
| Head dim (c) | 512 | 512 |
| Query compression dim (d_c) | 1024 | 1536 |

### HCA (Heavily Compressed Attention)

| Parameter | Flash | Pro |
|---|---|---|
| Compression rate (m') | 128 | 128 |
| Query heads (n_h) | 64 | 128 |
| Head dim (c) | 512 | 512 |

### Shared attention parameters

| Parameter | Value |
|---|---|
| Output projection groups (g) | 8 (Flash), 16 (Pro) |
| Intermediate attn output dim (d_g) | 1024 |
| Sliding window size (n_win) | 128 |
| MQA: shared K/V | Yes |

### Attention details relevant to SYCL

- Query and KV entry normalization is applied before attention
- Partial RoPE: only applied to a subset of head dimensions
- Additional sliding window attention branch exists alongside CSA/HCA
- Attention sink tokens are preserved for long-context stability

### Lightning Indexer

The lightning indexer selects top-k KV entries for sparse attention. Uses FP4 QK computation for 2x speedup with 99.7% KV entry recall. In llama.cpp SYCL, this is implemented in `lightning-indexer.cpp`.

## mHC (Manifold-Constrained Hyper-Connections)

Residual connections between non-consecutive layers via learned combination matrix.

| Parameter | Value |
|---|---|
| Expansion factor (n_hc) | 4 |
| Sinkhorn-Knopp iterations (t_max) | 20 |

In llama.cpp SYCL, implemented in `dsv4-hc.cpp`: pre, comb (softmax + Sinkhorn), post kernels.

## Multi-Token Prediction (MTP)

- MTP depth = 1 (single lookahead token)
- Inherited from DeepSeek-V3 design
- Not yet implemented in llama.cpp

## KV Cache

- Mixed precision: BF16 for RoPE dims, FP8 for remaining dims
- Heterogeneous layout: classical KV cache (CSA/HCA) + state cache (SWA + uncompressed tail tokens)
- On-disk KV cache storage for shared-prefix reuse
- V4-Flash uses 7% KV cache of V3.2 for 1M-token context

## Quantization

- FP4 (MXFP4) QAT applied to MoE expert weights and QK path in CSA indexer
- FP4-to-FP8 dequant is lossless (FP8 E4M3 has 2 extra exponent bits vs FP4 E2M1)
- llm-scaler reference kernels target FP8 (e4m3/e5m2) for MoE prefill on Intel BMG

## Key Files in llama.cpp SYCL Backend

| File | Purpose |
|---|---|
| `ggml/src/ggml-sycl/topk-moe.cpp` | TopK MoE routing kernel (softmax + argmax fusion) |
| `ggml/src/ggml-sycl/mmvq.cpp` | MoE GEMV: `mul_mat_vec_q_moe`, `mul_mat_vec_q_moe_reorder` |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | `ggml_sycl_mul_mat_id`, `ggml_sycl_mul_mat_id_mmvq_fused` |
| `ggml/src/ggml-sycl/dsv4-hc.cpp` | mHC pre/comb/post kernels |
| `ggml/src/ggml-sycl/lightning-indexer.cpp` | Lightning attention indexer |
| `ggml/src/ggml-sycl/fattn-*.cpp` | Flash attention kernels |

## Key Files in llm-scaler Reference

| File | Purpose |
|---|---|
| `csrc/moe_prefill/moe_prefill_fp8.sycl` | FP8 DPAS MoE prefill kernels (UP + DOWN) |
| `csrc/moe_prefill/moe_prefill_int4.sycl` | Int4 reference kernel |
| `csrc/xpu/esimd_kernels/moe_ops.h` | TopK, scatter, SiLU, gather kernels |
| `csrc/xpu/esimd_kernels/prefill_dpas.h` | DPAS-based paged attention prefill |
| `csrc/xpu/esimd_kernels/fp8_moe_gemm.h` | FP8 MoE GEMM for batch decode |
| `csrc/xpu/esimd_kernels/fp8_GEMV_bmg.h` | FP8 GEMV optimized for BMG |

## Optimization Notes

- The llm-scaler FP8 prefill kernel uses `dpas<8,8,fp16,fp16,fp16,fp16>` with fp32 accumulators to avoid fp16 overflow in deep layers
- Weight is DPAS operand A in plain row-major [N=8, K=16]=128 format, NOT VNNI-packed
- Input activations are DPAS operand B in VNNI [K=16, 16-tok] layout
- Per-expert scalar scale applied post-accumulator, not per-group/per-column
- The `lsc_gather` with `cache_hint::cached` is used for input activation gathering
- `block_load<uint8_t, 16>` loads weight rows directly from FP8 bytes
- The kernel template parameters `MAX_M` (max tokens per batch, default 32) and `N` (N-tile size, 16 for UP, 32 for DOWN) control workgroup geometry
