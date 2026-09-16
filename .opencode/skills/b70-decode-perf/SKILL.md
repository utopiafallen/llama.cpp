---
name: b70-decode-perf
description: Qwen3.8-27B Q6_K decode performance findings on the Intel Arc Pro B70 (Battlemage) - DRAM ceiling, per-op GPU breakdowns, what works at long context. Use when optimizing SYCL decode t/s on the B70 or deciding whether a perf target is reachable.
---

# B70 Qwen3.8-27B Q6_K decode perf findings

Model: `Qwen3.8-27B-UD-Q6_K.gguf` (20.46 GiB, dense Q6_K), B70 = Intel Arc Pro B70
(32GB GDDR6, 256-bit @ 19 Gbps = 608 GB/s spec, 16MB L2).
Symlink: `D:\model.md` on the B70.

## Current best (2026-09-15, post-XMX-Q8_0-decode)

| config | context | t/s | ms/tok | notes |
|--------|---------|-----|--------|-------|
| Tile FA (eSIMD) | 51K | 14.71 | 67.9 | baseline, Q8_0 per-tile dequant |
| XMX decode + per-tile Q8_0 dequant | 51K | **15.97** | **62.6** | best with Q8_0 KV |
| XMX decode + FP16 KV (no dequant) | 51K | 18.42 | 54.3 | ceiling, won't fit at 256K |
| Tile FA (eSIMD) | 144K | 8.6 | 116.0 | |
| XMX decode + per-tile Q8_0 dequant | 144K | **10.18** | **98.2** | +18% vs tile at long ctx |

**CRITICAL: Previous "22.6 t/s at 142K" numbers were INVALID.** The `slot_save` parameter in
the completion request body is silently ignored by llama-server. All prior slot-save tests
were measuring zero-context decode. Real long-context decode is 8.6-10.2 t/s at 144K.

MTP is parked (draft state save/restore has size-mismatch issues).

## Per-op GPU breakdown at 144K (GGML_SYCL_PROFILE=1, serialized, XMX decode FA active)

Method: `stream->wait()` after each op in graph_compute. Times are CPU-enqueue + GPU-exec.
Profiling serializes execution; use for breakdowns only, not throughput numbers.

| Op | Time | % | Notes |
|---|---|---|---|
| MUL_MAT | 533.47ms | 79.9% | weight streaming (20GB), dominates serialized time |
| FLASH_ATTN_EXT | 73.86ms | 11.1% | 32 ops x ~2.3ms, XMX per-tile Q8_0 dequant |
| GLU | 14.50ms | 2.2% | FFN activation |
| ADD | 8.55ms | 1.3% | residuals |
| SSM_CONV | 8.26ms | 1.2% | GDN conv |
| CONCAT | 6.75ms | 1.0% | GDN conv_input |
| misc | ~22ms | ~3% | UNARY, ROPE, CONT, RMS_NORM, etc |
| **Total** | **667.41ms** | | pipelined = 98ms/tok (10.18 t/s) |

FA is 32 ops (one per full-attention layer). Each XMX FA op at 144K: ~2.3ms
(563 splits x 4 kv_heads WGs, SPLIT=256, GQA=6, D=256).

## XMX decode FA with Q8_0 KV (committed 2026-09-15)

Three commits: bulk conv support, per-op profiler, per-tile dequant.

**Tile FA (eSIMD) is compute-bound:** FP16 MUL + CVT to FP32 + FP32 ADD = 3 instructions
per 2 FLOPs. No native FP16xFP16->FP32 FMA on eSIMD. XMX offloads to dedicated matrix HW.

**Per-tile Q8_0 dequant in XMX kernel:** Each 16x16 B-tile reads Q8_0 blocks (34B) from
global, dequants to FP16 in registers, stores to a 512B LDS staging buffer, then XMX loads
from LDS. No bulk tensor conversion needed. Eliminates the full write+read round trip.

Files: `fattn-xmx-decode.cpp` (kernel + host launch), `ggml-sycl.cpp` (profiler).

**Conversion overhead is real but smaller than theoretical:**
- Theoretical: ~1.4ms per FA op at 144K (894MB DRAM traffic at 640 GB/s)
- Measured delta (Q8_0 vs FP16 KV at 51K): ~10ms/step / 32 ops = 0.31ms per FA op
- Per-tile saves ~2.3% over bulk conv (15.97 vs 15.61 t/s)

**INT8 XMX: NOT supported on B70.** Hard crash during JIT (no catchable SYCL exception).
Backend lacks INT8 matrix builtins. Cannot avoid FP16 staging for XMX operands.

**Zero-dequant A/B test (proves LDS staging is the cost, not ALU):**
- Real dequant (multiply by scale): 20.41 t/s at 16.5K ctx
- Zero-dequant (no multiply, just cast): 20.60 t/s at 16.5K ctx
- Delta: 0.9% = noise. The scale multiply is free (overlapped with memory ops).
- Conclusion: the ~13% gap to FP16 KV at 51K is entirely the LDS write + barrier +
  XMX load-from-LDS path. Not reducible without INT8 XMX or a custom KV format.

**Double-buffered dequant (co-issue attempt): WORSE (-15%).**
- Restructured loop: XMX mad → dequant next tile → barrier (dequant co-issues with XMX?)
- Result: 17.24 t/s vs 20.41 single-buffer at 16.5K ctx.
- Confirms `joint_matrix_mad` effectively blocks the sub_group despite hardware 3-way
  co-issue claim. "Minimal XMX+vector co-issue" (Chips and Cheese) is accurate in practice.
- Do NOT retry double-buffering for XMX decode FA on B70.

**Full-batch dequant (all tiles, 1 barrier): WORSE (-5%).**
- Dequant all D/16=16 K tiles upfront (8KB LDS), one barrier, then 16 XMX mads with no
  intermediate barriers. Same for V in PV loop.
- Result: 19.37 t/s vs 20.41 single-tile at 16.5K ctx.
- Likely register pressure from 16 unrolled Q8_0 block reads, or LDS bank conflicts
  with the larger working set. Single-tile-per-iteration is optimal.

**Batch=4 dequant (4 tiles, 4 barriers): NEUTRAL (no gain).**
- Result: 20.44 t/s vs 20.41 single-tile at 16.5K ctx. Within noise.
- Reducing barrier count from 16 to 4 per c-iteration has no measurable effect.
- Confirms the per-barrier overhead is negligible; the LDS write+read round-trip is
  the fixed cost that cannot be amortized by batching.
- Do NOT retry batched dequant for XMX decode FA on B70.

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

## MTP (NextN) decode - PARKED

Qwen3.8 ships 1 MTP block (`qwen35.nextn_predict_layers = 1`).
Enable: `--spec-type draft-mtp --spec-draft-n-max 4` (or 6).
MTP ctx shares main model weights (no second 20GB copy). Fits 32GB B70.

**Parked:** Draft state slot save/restore has a size-mismatch issue (file=107KB vs
expected=16 post-erase). Unresolved. The .dft slot-save change was reverted.

At 142K context, MTP is expected to give modest gains since the verify step is
weight-streaming bound (MUL_MAT dominates) and FA is a smaller fraction with XMX.

## Plain decode at short context (~32K)

llama-bench tg32 at 32K: ~22-23 t/s (weight-streaming bound, ~82% of DRAM bandwidth).
This is the eSIMD DMMV ceiling - no FA optimization helps here since FA is negligible
at short context. The XMX FA gains only matter at long context (>=50K).

## Dead ends (do NOT retry)

- **XMX Q6_K GEMV**: 2.3 t/s vs 22.5 eSIMD (10x loss). XMX's 2D tile access forces scattered
  reads; structurally inferior for quantized data. Stashed: `.opencode/dmmv-q6k-xmx-stash.patch`.
- **Interleaved Q6_K layouts**: all variants (210B, 212B padded, 420B pair) lose to SoA 23.05.
  The Xe2 backend doesn't handle misaligned/padded loads well.
- **bf16 M-kernel**: B70 has no fast bf16 FMA (0.52x of fp32). Dead.
- **INT8 joint_matrix on B70**: hard crash during JIT, no builtins. Only FP16 supported.
- **nbatch_fa 64->128 in tile kernel**: 14.50 vs 14.71 t/s (register pressure). Worse.
- **XMX_DECODE_SPLIT 256->512**: 14.69 vs 15.61 t/s (LDS occupancy loss at 24KB/WG). Worse.
- **Custom KV cache format for XMX INT8**: backend has no discretion over tensor layout.
  Would require new ggml type (all backends) or shadow buffer (architecturally invasive).
  Blocked by lack of INT8 XMX support anyway.
- **SYCL graph mode**: -16%, avoid.
- **2-GPU split**: worse (cross-GPU sync overhead).

## Profiling infrastructure

- `GGML_SYCL_PROFILE=1`: per-op GPU timing via `stream->wait()` after each op in
  graph_compute. Writes per-op lines (>50us) + summary to `sycl-profile.log` (CWD).
  Serializes execution. Shows top ops by time with % of total.
- `GGML_SYCL_PROFILE=0`: no profiling (default for throughput runs).
- Server launch for profiling: see battlematrix-ssh skill for the SSH background pattern.

## Measurement methodology (IMPORTANT)

**The server is STATELESS.** Each completion request must carry the full conversation history.
Slot cache only persists KV, NOT prompt/token state. The `slot_save` parameter in the request
body is SILENTLY IGNORED - it does NOT save/restore the slot.

Correct workflow for measuring decode at N context tokens:
1. Launch server with `--slot-save-path D:\slot-save\`
2. Send the full prompt JSON (e.g. test-64k.json) to prefill the KV cache
3. `POST /slots/0?action=save` with `{"filename":"name"}` to save KV
4. `POST /slots/0?action=restore` with `{"filename":"name"}` to restore KV
5. Send the SAME full prompt JSON again - prefix hits restored KV (fast), then decode begins
6. Read `eval time` from server log for decode speed

NEVER send a short `" "` prompt after restore. Always send the same full prompt.
Test prompts on B70 desktop: `TestPrompt152k.json` (~144K), `test-64k.json` (~51K).

## Prefill reorder poison (pre-existing, not decode)

Any M<=8 pass triggers in-place weight REORDER (GGML_SYCL_ENABLE_OPT=1). After that, all
later M>8 prefills run ~21% slower (pp256: 752 -> 589 t/s). Affects serving, not decode.
Fix requires upstream architecture change (reusable reorder buffer or deferred first reorder).
