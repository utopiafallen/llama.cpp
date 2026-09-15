---
name: b70-decode-perf
description: Qwen3.8-27B Q6_K decode performance findings on the Intel Arc Pro B70 (Battlemage) - DRAM ceiling, per-op GPU breakdowns, what works at long context. Use when optimizing SYCL decode t/s on the B70 or deciding whether a perf target is reachable.
---

# B70 Qwen3.8-27B Q6_K decode perf findings

Model: `Qwen3.8-27B-UD-Q6_K.gguf` (20.46 GiB, dense Q6_K), B70 = Intel Arc Pro B70
(32GB GDDR6, 256-bit @ 19 Gbps = 608 GB/s spec, 16MB L2).
Symlink: `D:\model.md` on the B70.

## Current best (2026-09-14, post-Q8_0-FA-fix)

| config | context | t/s | ms/tok | notes |
|--------|---------|-----|--------|-------|
| Plain decode | 32K | 22.7 | 44.0 | llama-bench tg32, -p 32768 |
| Plain decode | 142K | 22.6 | 44.2 | slot-save, server API |
| MTP (n_max=4) | 142K | 25.9-27.5 | 36.4-38.6 | +20% vs plain |
| MTP (n_max=6) | 32K | ~45.6 | - | llama-bench --mtp -d 32768 |

Key insight: **at long context (>=142K), MTP gives only +20% not the theoretical +5x.**
The verify step is weight-streaming bound (MUL_MAT ~31ms) and doesn't amortize with batch
size the way it does at short context where FA was dominant.

## Per-op GPU breakdown at 142K (GGML_SYCL_PROFILE=1, serialized)

Method: `stream->wait()` after each op in graph_compute. Times are CPU-enqueue + GPU-exec.
Profiling adds ~10ms/step overhead; use for breakdowns only, not throughput numbers.

### Plain decode (ne[1]=1, 4518 nodes, ~67ms serialized)

| Op | Time | % | Notes |
|---|---|---|---|
| MUL_MAT | 31.25ms | 59.8% | weight streaming (22GB), unavoidable |
| GET_ROWS | 11.41ms | 21.9% | token embedding + GDN state reads |
| rms_norm_fused | 1.04ms | 2.0% | |
| FLASH_ATTN_EXT | **0.92ms** | **1.8%** | Q8_0 tile load (was 69ms pre-fix) |
| RMS_NORM | 0.86ms | 1.7% | |
| CONCAT | 0.77ms | 1.5% | GDN conv_input |
| ADD | 0.71ms | 1.4% | residuals |
| batched_conv_state_cpy | 0.71ms | 1.4% | |
| misc (SCALE, UNARY, etc) | ~20ms | ~30% | small ops, many of them |

### MTP draft head (ne[1]=1, 55 nodes, ~3.3ms each x4 = 13ms/step)

Small graph: 1 transformer layer (eh_proj + attn + FFN) + shared lm_head.

### MTP step total: ~67ms (main verify) + 13ms (4x draft head) = ~80ms

## The Q8_0 FA tile load (the big long-context win, committed 2026-09-14)

Before: every FA op converted the ENTIRE Q8_0 KV cache to FP16 before the kernel ran.
At 142K context with 4 KV heads x 256 head_dim: K+V Q8_0 = 310MB, FP16 = 1168MB.
Per FA op: read 310MB Q8_0 + write 1168MB FP16 + read 1168MB FP16 = ~2.7GB traffic.

After: `flash_attn_tile_load_tile_q8` reads Q8_0 blocks directly, dequants to FP16 in
shared memory per tile. Per FA op: read 310MB Q8_0 only. **4.7x less DRAM traffic.**

Result: FA went from 69ms to 0.92ms at 142K context (75x faster). This made long-context
decode 2.5x faster overall (8.9 -> 22.6 t/s) because the conversion was the dominant cost,
not the FA compute itself.

Files: fattn-tile.hpp (load fn + q8_input branch), fattn-common.hpp (skip conversion +
q8_input kernel arg), fattn-vec.hpp (signature), fattn-tile.cpp (global var).

## Quant distribution + kernel paths

| type  | bytes   | frac  | decode kernel path       |
|-------|---------|-------|--------------------------|
| Q6_K  | 13.89GB | 63.2% | eSIMD DMMV (2-row SoA)   |
| Q5_K  | 4.93GB  | 22.4% | eSIMD DMMV (2-row SoA)   |
| Q8_0  | 2.85GB  | 13.0% | eSIMD DMMV               |
| IQ4_XS| 142MB   | 0.6%  | MMVQ dp4a                |
| Q4_K  | 100MB   | 0.5%  | MMVQ dp4a                |

output/lm_head = Q8_0 [5120, 248320]. token_embd = Q6_K [5120, 248320].
Model: 64 blocks (48 GDN + 16 full-attn every 4th), head_dim=256, gqa=6 (24 q heads, 4 kv).

## MTP (NextN) decode

Qwen3.8 ships 1 MTP block (`qwen35.nextn_predict_layers = 1`).
Enable: `--spec-type draft-mtp --spec-draft-n-max 4` (or 6).
MTP ctx shares main model weights (no second 20GB copy). Fits 32GB B70.

Mechanism per step:
1. Draft: 4x MTP head passes (55 nodes each, ~3.3ms) = 13ms
2. Verify: 1x main model pass with N+1 tokens (4518 nodes, ~67ms)
3. Accept/reject based on probability match

At 142K: ~80ms/step total, ~1.2-1.5 tokens accepted per step = 26-27 t/s.
At 32K: the fp16 M-kernel + eSIMD small-batch paths make verify faster (~45 t/s with n_max=6).

**Why MTP is less effective at long context:** FA was the bottleneck pre-Q8_0-fix, and MTP
amortizes it well (same KV read for 1 or 5 queries). Now that FA is 0.92ms (negligible),
the verify step is dominated by MUL_MAT weight streaming (31ms) which is IDENTICAL whether
processing 1 or 5 tokens. The MTP head overhead (13ms) is pure cost with no corresponding
savings in the verify. Net: only +20% at 142K vs +100%+ at short context.

## Plain decode ceiling analysis (short context, ~22.7 t/s)

- 22.7 t/s x 22.0 GB = 500 GB/s = 82% of 608 GB/s spec
- FFN q6_K/q5_K (fused GLU/add): 516-570 GB/s (saturated)
- attn projections: 380-500 GB/s
- lm_head Q8_0: 590 GB/s
- iq4_nl 17408x5120: 92.6 GB/s (biggest single inefficiency, ~2% of time)
- The 2-row SoA eSIMD DMMV is at the practical ceiling; ROWS>2, wider loads, XMX GEMV all lose.

## Dead ends (do NOT retry)

- **XMX Q6_K GEMV**: 2.3 t/s vs 22.5 eSIMD (10x loss). XMX's 2D tile access forces scattered
  reads; structurally inferior for quantized data. Stashed: `.opencode/dmmv-q6k-xmx-stash.patch`.
- **Interleaved Q6_K layouts**: all variants (210B, 212B padded, 420B pair) lose to SoA 23.05.
  The Xe2 backend doesn't handle misaligned/padded loads well.
- **bf16 M-kernel**: B70 has no fast bf16 FMA (0.52x of fp32). Dead.
- **XMX decode FA**: +8% at 32K, wash at short context. Now irrelevant since Q8_0 tile load
  made FA negligible (0.92ms at 142K). The XMX verify FA for MTP is also a wash vs tile FA.
- **SYCL graph mode**: -16%, avoid.
- **2-GPU split**: worse (cross-GPU sync overhead).

## Profiling infrastructure

- `GGML_SYCL_PROFILE=1`: per-op GPU timing via `stream->wait()` after each op in
  graph_compute. Writes summary to `sycl-profile.log` (CWD). Serializes execution (+10ms/step
  overhead). Shows top 15 ops by time with % of total.
- `GGML_SYCL_PROFILE=0`: no profiling (default for throughput runs).
- Server launch for profiling: see battlematrix-ssh skill for the SSH background pattern.

## Prefill reorder poison (pre-existing, not decode)

Any M<=8 pass triggers in-place weight REORDER (GGML_SYCL_ENABLE_OPT=1). After that, all
later M>8 prefills run ~21% slower (pp256: 752 -> 589 t/s). Affects serving, not decode.
Fix requires upstream architecture change (reusable reorder buffer or deferred first reorder).
