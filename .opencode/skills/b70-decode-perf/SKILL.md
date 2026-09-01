---
name: b70-decode-perf
description: Qwen3.8-27B Q6_K decode performance findings on the Intel Arc Pro B70 (Battlemage) - DRAM ceiling, what was tried, what works. Use when optimizing SYCL decode t/s on the B70 or deciding whether a perf target is reachable on this hardware.
---

# B70 Qwen3.8-27B Q6_K decode perf findings

Model: `Qwen3.8-27B-UD-Q6_K.gguf` (20.46 GiB, dense), B70 = Intel Arc Pro B70
(32GB GDDR6, 256-bit @ 19 Gbps = 608 GB/s spec, 16MB L2). Bench:
`llama-bench -r 2 --device SYCL0 -m <model> -p 8 -n 128` (see sycl16-build-deploy).

## Bottom line (2026-08-22, REVISED 2026-08-27, INTERLEAVED DEAD-END + MTP WIN 2026-08-30)

- **Best decode WITHOUT speculation = ~23.05 t/s** (Q6_K/Q5_K/Q8_0 on eSIMD DMMV + 1-row
  small-N kernel + f16-convert skip, committed `76f80b445`). 23.05 t/s x 21.97 GB/token =
  ~506 GB/s = 83% of the 608 GB/s spec. This is the practical ceiling for plain decode.
- **BEST decode WITH MTP speculation = ~30.8 t/s** (`--spec-type draft-mtp
  --spec-draft-n-max 6`), correct output, verified 2026-08-30 (n_max=4 -> 30.0, n_max=8
  collapses to 4.8). That is a +34% win over plain decode (30.8 vs 23.05). See the "MTP
  (NextN) decode" section. MTP is the route past the DRAM ceiling - it amortizes the
  20.46 GB main-model read over multiple accepted tokens. This is the recommended
  configuration now that plain-decode levers are exhausted.
- **The interleaved-weight-layout path is a DEAD END (exhausted 2026-08-30).** The
  2026-08-27 REVISED note predicted 25-26 t/s from GEMV-shaped probes (580-594 GB/s),
  but the probes over-predicted: every real interleaved variant loses to SoA 23.05.
  See "Interleaved Q6_K: all three layouts lose to SoA" below. Do NOT re-attempt any
  Q6_K interleaved/padded/pair layout for decode.

## Quant distribution (probed via gguf-inspect)

| type  | bytes   | frac  | kernel path (decode)       |
|-------|---------|-------|----------------------------|
| Q6_K  | 13.89GB | 63.2% | eSIMD DMMV (2-row)         |
| Q5_K  | 4.93GB  | 22.4% | eSIMD DMMV (2-row, #26376) |
| Q8_0  | 2.85GB  | 13.0% | eSIMD DMMV (this work)     |
| IQ4_XS| 142MB   | 0.6%  | (negligible, not touched)  |
| Q4_K  | 100MB   | 0.5%  | (negligible, not touched)  |

output/embed = Q6_K, vocab 248320. Q6_K+Q5_K = 85.6% of bytes, both on the same
eSIMD DMMV structure (generic `dequantize_mul_mat_vec_reorder_esimd` in dmmv.cpp,
per-quant `mac_pair` in esimd.hpp).

## The eSIMD DMMV kernel (what it does)

- work-group = 4 lanes (`GGML_SYCL_DMMV_ESIMD_WG_SIZE`), owns 2 consecutive output rows.
- loop over K-blocks (QK_K=256): lane `tid` handles blocks `tid, tid+4, ...`; the 4
  lanes cover 4 consecutive blocks = 512B of `ql` per row per iteration (coalesced).
- per block: `block_load<float,256>` of the activation `y` + per-quant dequant-MAC of
  the 2 rows. SoA (reordered) weight layout: ql/qh/scales are separate contiguous arrays.
- epilogue: per-lane partials -> lmem -> lane 0 reduces -> dst.
- GLU (gate+up) fusion = same but 4 accumulators (gate/up x 2 rows) + silu(g)*u epilogue.

## Experiments tried (all A/B'd vs 22.33 t/s baseline)

| change | idea | result | why |
|--------|------|--------|-----|
| **2-row (baseline)** | - | **22.33** | best |
| ROWS=4 (4 rows/WG) | amortize the shared `y` load over 2x rows | 22.13 | `y` is L2-cached (hidden-dim 16-32KB << 16MB L2), so cutting redundant `y` loads cuts L2 traffic, NOT the DRAM weight stream that is the real bottleneck. No DRAM win. |
| 2-block/lane-iter | wider contiguous weight stream (1KB ql vs 512B) for better DRAM bursts | 22.22 | marginal; already well-coalesced, no win |
| row-per-lane (sglang port, `GGML_SYCL_Q6K_GEMV_ROW=1`) | 1 lane/row, 512-elem tiles (llm-scaler design) | 20.39 | slower than our 2-row SoA kernel |
| `GGML_SYCL_ENABLE_ESIMD=0` (MMVQ) | dp4a path | 20.84 | eSIMD DMMV is faster |
| SYCL graph mode | - | 17.93 | -16%, avoid |
| 2-GPU `-ts 54,46` | split layers | 20.36 | worse (cross-GPU sync) |

Conclusion: the 2-row SoA eSIMD DMMV already beats Intel's reference row kernel and is
at the practical DRAM ceiling. Do not re-try ROWS>2 or wider loads.

## Q8_0 eSIMD path (this work, +0.5 t/s)

Q8_0 (13% of bytes, output/lm_head + some attn) was the only non-negligible quant NOT on
the eSIMD DMMV path - it used the reorder-MMVQ kernel (q8_0 x q8_1, dp4a). Moved it to
the eSIMD DMMV path. **Result: +0.5 t/s** (cool 22.87 vs 22.34; warm 22.63 vs 22.08,
back-to-back A/B). pp8 (prompt) is unaffected because multi-token (ne[1] 2..8) still uses
MMVQ; only decode (ne[1]==1) uses the eSIMD DMMV.

Why it's feasible (the machinery already existed):
- `reorder_qw_q8_0` already produces the SoA layout: `[qs: nrows*ncols int8][d:
  nrows*(ncols/32) half]` (d stored as half, min dropped - streams slightly fewer bytes
  than the original block_q8_0).
- The MMVQ side already has a reorder variant selected by `optimized_feature.reorder`.
- The reorder happens on the MMVQ path already (GGML_SYCL_ENABLE_OPT defaults to 1), so
  Q8_0 was ALREADY reordered - the eSIMD path just reads the same SoA layout.

What was added:
- `esimd_reorder_q_traits<GGML_TYPE_Q8_0>` in esimd.hpp: `make_ptrs` (qs + d) and
  `mac_pair`. One mac_pair call = a 256-elem chunk = 8 Q8_0 blocks. Loads qs as two
  `block_load<int8_t,128>` (max 8-bit width in this codebase is 128), 8 half scales,
  dequant = `convert<float>(qs)*d`, FMA. NOTE: `convert<float>()` does NOT accept the
  `.select<32,1>()` temporary - store the select result in a named `simd<int8_t,32>` first.
- `dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd` wrapper in dmmv.cpp (same shape as the
  Q6_K eSIMD wrapper, calls the generic `dequantize_mul_mat_vec_reorder_esimd<Q8_0,false>`).
- Q8_0 case in the DMMV dispatch (dmmv.cpp): use the eSIMD kernel when reordered &&
  `g_ggml_sycl_enable_esimd && g_ggml_sycl_q80_gemv_esimd`.
- Q8_0 in `ggml_sycl_supports_reorder_esimd` (ggml-sycl.cpp), gated by the env var.
- Env var `GGML_SYCL_Q80_GEMV_ESIMD`, **default 1** (opt-out). common.hpp + ggml-sycl.cpp
  (def, env read, log).

Safety: the eSIMD Q8_0 kernel is only reached when `optimized_feature.reorder` is true,
so non-reordered tensors (MoE experts - Q8_0 not in the MoE reorder list; or
GGML_SYCL_ENABLE_OPT=0) fall back to the original-layout DMMV kernel. No corruption risk.

Q5_K (22%, also eSIMD DMMV 2-row) has NO further win - same kernel structure as Q6_K,
already at the DRAM ceiling. IQ4_XS/Q4_K (~1% total) are not worth touching.

## 2026-08-29 session (1-row kernel, 2-matrix merge, iq4_nl SoA)

Profile (GGML_SYCL_PROFILE tensor totals) of the ~23.0 t/s build, by tensor GB/s:
- FFN q6_K/q5_K (fused GLU/add): 516-570 GB/s (saturated, well-optimized).
- attn (attn_gate 6144x5120, ssm_out 5120x6144): 380-500 GB/s.
- **iq4_nl 17408x5120: 92.6 GB/s (biggest single inefficiency, ~2.1% of time).**
- iq4_xs 17408x5120: 260-280 GB/s.
- q8_0 48x5120 (ssm_beta/alpha): 23.6 GB/s (M=48 too small to saturate).
- LM head q8_0 248320x5120: 590 GB/s.

### 1-row small-N eSIMD DMMV kernel (+0.15 t/s, committed)
- `dequantize_mul_mat_vec_reorder_esimd_row1<T,ADD_RES>` in dmmv.cpp: 1 output row per
  work-group (vs 2-row). Below `GGML_SYCL_DMMV_ESIMD_ROW1_MAX_NROWS=8192` rows the 2-row
  layout (2*N subgroups) does not generate enough concurrent subgroups to hide the
  sequential per-subgroup K-load latency; 1-row doubles the subgroup count (4*N).
- `dequantize_mul_mat_vec_reorder_esimd_dispatch<T>` picks 1-row (nrows<=8192) else 2-row.
- f16-convert skip: the eSIMD q8_0 kernel consumes f32 `y` directly, so the src1 f32->f16
  conversion is skipped when the q8_0 eSIMD path is active (DMMV dispatch, dmmv.cpp).

### 2-matrix same-input fused GEMV (Lever 3, WASH - kept, gated)
- Idea: the attn qkv + qkv_gate (and up+gate) are two MUL_MATs on the SAME input vector.
  Merging into one launch was expected to improve DRAM concurrency.
- Implemented: `dequantize_mul_mat_vec_2matrix_reorder_esimd<TA,TB>` in dmmv.cpp (2 rows
  per WG, per-WG A/B branch, 2 separate dst outputs), `ggml_sycl_dmmv_2matrix_reorder_esimd`
  dispatch (Q4_K/Q5_K/Q6_K/Q8_0), detection `ggml_sycl_mul_mat_pair_same_input_fused` in
  ggml-sycl.cpp (finds a matching mmB within 4 nodes, same src[1], same K, only noops
  between; M_A even guard), env var `GGML_SYCL_FUSE_MM_PAIR` (default 1).
- **Result: WASH (22.91-22.92 vs 22.95 baseline, both 1-row and 2-row versions).** The B70
  already pipelines the two separate launches and saturates DRAM; merging does not increase
  effective concurrency. GB/s is not limited by rows-in-a-single-launch. Kept per user (harmless,
  correct, gated).

### iq4_nl SoA reorder + MMVQ kernel (WASH - NOT TRIGGERED on B70)
- Idea: iq4_nl (92.6 GB/s, 32-elem blocks, packed 18B layout) -> SoA reorder
  ([qs: nb*16][d: nb*half]) + MMVQ dp4a kernel (each thread handles a whole block, no
  eSIMD byte-gather). The dp4a dequant (get_int_from_table_16 + dp4a) is more efficient
  for the non-linear iq4_nl lookup than an eSIMD software gather.
- Implemented: `reorder_qw_iq4_nl` (ggml-sycl.cpp), `vec_dot_iq4_nl_q8_1_soa` (vecdotq.hpp),
  `mul_mat_vec_q_iq4_nl_q8_1_soa` + `_soa_sycl` (mmvq.cpp), iq4_nl in
  `ggml_sycl_supports_reorder_mmvq` + `reorder_qw`, reorder-flag check in the iq4_nl MMVQ
  dispatch case. Correctness passes ("Paris").
- **Result: WASH (23.00/22.93 vs 23.0) BECAUSE THE REORDER IS NOT TRIGGERED** (a one-time
  log in `reorder_qw_iq4_nl` never fires). The iq4_nl is not in the GLU fusion whitelist
  (fusion.cpp:38 = Q4_K/Q5_K/Q6_K only), so it goes the main dispatch (which DOES call
  opt_for_reorder(MMVQ)) - yet the reorder still does not fire. Some gating condition
  (likely the iq4_nl is not reaching opt_for_reorder, or should_reorder_tensor fails for it)
  is unverified. Net: the iq4_nl still uses the packed kernel; no gain.
- iq4_nl root cause: 32-elem blocks (160/row for N=5120 vs iq4_xs 20/row) = 8x more
  per-block scale reads + loop overhead. The SoA layout does not change the block size, so
  it cannot fix the per-block overhead. iq4_nl is hard to fix (the small block size is
  inherent to the quant).

Conclusion: the main levers are exhausted. FFN is saturated; attn 2-matrix merge is a wash;
iq4_nl is hard (small blocks) and its SoA reorder does not even trigger on the B70. The
q8_0 48x5120 (M too small) is hard. **~23.0 t/s is the plain-decode ceiling.** The
interleaved-layout path (bottom-line REVISED 2026-08-27) turned out to be a DEAD END (see
below). The route past the ceiling is MTP speculation, which is now verified working:

## MTP (NextN) decode - the win (2026-08-30)

Qwen3.8 (arch `qwen35`) ships a single MTP/NextN block: gguf has
`qwen35.nextn_predict_layers = 1` and `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`
(eh_proj [10240,5120] Q6_K). llama.cpp has full support (`common_speculative_impl_draft_mtp`
in common/speculative.cpp, `graph_mtp` in src/models/qwen35.cpp). It is OFF by default:
the MTP tensors are skipped at load unless `mparams.load_mtp` is set, which happens only
when the speculative type includes `draft-mtp` (common.cpp: `mparams.load_mtp = ...find(
COMMON_SPECULATIVE_TYPE_DRAFT_MTP)`). Without it the loader logs "unused tensor
blk.64.nextn.* -- ignoring".

Enable with:
    llama-cli --spec-type draft-mtp --spec-draft-n-max 4 -m <model> ...
(`--spec-draft-n-max` sets draft.n_max; default 3. qwen35 is a single MTP head so n_max is
NOT clamped to 1 - that clamp only applies to the multi-head "chain_heads"/step35 mode.)

Key facts:
- The MTP draft context SHARED the main model's weights (`shares_model = !has_draft`,
  common.cpp:1317) - it does NOT load a second 20GB copy. Memory = main model + small MTP
  block + KV for both ctxs. Fits the 32GB B70 (no oversubscription).
- Mechanism: each step runs the cheap 1-layer MTP block up to n_max times to draft tokens,
  then ONE main-model pass (20.46 GB) verifies them. Accepted drafts give >1 token per
  20.46 GB read -> beats the DRAM ceiling.
- **Measured (verified 2026-08-30), correct output ("Paris") all runs:**
    - n_max=4 -> Generation **30.0 t/s** (prompt 39.7)
    - n_max=6 -> Generation **30.8 t/s** (prompt 39.4, reproducible x2)  <-- RECOMMENDED
    - n_max=8 -> Generation **4.8 t/s** (COLLAPSE: long draft chains; the single MTP block's
      1-step-ahead prediction degrades over a long horizon so most drafted tokens are
      rejected while you still pay up to 8 cheap-but-real MTP-block draft passes per step.
      Also that run hit box instability - do not use n_max>6.)
  vs plain decode 23.05 t/s -> **+34%** (30.8). This is the new recommended config.
- The box was mildly degraded this session (plain baseline read 22.4 not 23.05), so the
  true MTP numbers on a healthy box are likely a few % higher. Re-confirm on a stable box.

Test bat: i5-test.bat (`--spec-type draft-mtp --spec-draft-n-max 4 -n 128 --temp 0.0
--no-display-prompt < NUL`). The `< NUL` gives the CLI stdin EOF so it exits after
generation (no REPL hang / no orphan).

## MTP verify bottleneck + Q6_K small-batch kernel (2026-08-30)

The MTP win above comes from amortizing the 20.46 GB main-model verify over the accepted
tokens. `common_speculative_print_stats` (speculative.cpp, needs `-v -v -v`) shows:
- **Acceptance is GOOD, not the bottleneck:** `#mean acc len` ~5-6/step, `#acc rate/pos`
  ~(1.0, 0.83, 0.83, 0.67, 0.5, 0.33). Do NOT chase draft quality for t/s.
- **Step breakdown (~145.5 ms/step):** the cheap MTP-block draft (~6 passes) is ~24 ms
  (422 GB/s); the ONE 7-token main-model verify is ~120 ms = **the bottleneck (~86%)**.

The 7-token verify goes the SoA Q6_K small-batch path (ne[1] 2-8):
`reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>` -> `mul_mat_vec_q_reorder_ncols_q6_k<7>`
(mmvq.cpp). It is **compute-bound on the Q6_K dot (2 dp4a + ~6 scalar-FP per token per
block-elem); the weight dequant (~14 ops) is shared and CSE'd by the compiler.** Batch
ridge (llama-bench -pg): tg1=506, pp6=202, pp12=58, pp512=40 GB/s - GB/s collapses as the
batch grows (memory-bound -> compute-bound).

Two kernel changes A/B'd on this path (i12-mtp-hoist.bat, n_max=6, temp default, -n 64,
SYCL0, `GGML_SYCL_Q6K_MMVQ_HOIST` 1 vs 0):
- **Dequant hoist** (default 1): hand-hoist vi0/vi1 out of the per-token loop. **NO-OP** -
  the compiler already CSEs the shared dequant. Kept (harmless, gated).
- **Scale fold** (same kernel): precompute `w0=d*sc0`, `w1=d*sc1` once per (block,elem);
  per-token tail `d80*dp4a(vi0,u0)*w0 + d81*dp4a(vi1,u1)*w1` (drops the `*sc` int-mul and
  the final `*d`). **Worth +~1 t/s** (HOIST=1 28.8-28.9 vs HOIST=0 27.9, reproducible x3,
  correct "Paris"). The compiler will NOT do this regrouping without fast-math.

Verdict (revised below): the small-batch GEMV is **dp4a-throughput-bound**; dequant-hoist
and scale-fold are the last cheap micro-opts on the dp4a kernel. The real win came from
replacing the whole dp4a small-batch path with an eSIMD wide-SIMD M-kernel.

### eSIMD wide-SIMD M-kernel (M=2-8) - the win (+8%, 2026-08-30)

The dp4a MMVQ does a per-token int8 dot (2 dp4a + scale-fold) and is dp4a-throughput-bound
for M=7. Replacing it with the existing fast M=1 eSIMD wide-SIMD design, extended to M
accumulators, wins: `dequantize_mul_mat_vec_reorder_esimd_m_q6k<M>` (dmmv.cpp) dequants ONE
row of the weight into 8x32-wide fp32 `deq` vectors (via the new Q6_K `dequant_block` in
esimd.hpp, factored out of `mac_pair`), then for each of the M tokens does
`acc[m] += y_vec.select<32,1>(32g) * deq[g]` on the **f32 activation directly** (no Q8_1
quantize; `src1_ddf_i` is always allocated). Weight dequant is amortized across M tokens and
the multiply is full-width eSIMD (no int8 dp4a, no scalar scale ops).

A/B on the B70 (i15-mtp-esimd.bat, M=7 MTP verify, n_max=6, temp default, -n 64, SYCL0,
`GGML_SYCL_Q6K_MMVQ_ESIMD` 1 vs 0, HOIST pinned 1), 2 runs each, both 9 tokens, identical
correct output "The capital of France is **Paris**.", same MTP mean acc len 6.0:
- eSIMD M-kernel (ESIMD=1): **31.18 / 31.20 t/s**
- dp4a MMVQ baseline (ESIMD=0): **28.86 / 28.86 t/s**  -> **+2.3 t/s (+8.1%)**, stable

Gate `GGML_SYCL_Q6K_MMVQ_ESIMD` (default 1, on). Requires src0 Q6_K + src1 F32 + reorder (SoA)
+ !interleaved + ncols 2-8; routed at the top of `ggml_sycl_op_mul_mat_vec_q` (mmvq.cpp)
before the per-token loop. lm_head is Q8_0 so it is unaffected; this covers FFN + attention
mat-vecs (the bulk of verify compute). Registers were fine at M=7 (acc[7]+deq[8]+y_vec fit).

So the "dp4a-throughput-bound, exhausted" verdict is SUPERSEDED: for the compute-bound
small-batch verify, the eSIMD wide-SIMD fp32 dequant-amortized-over-M kernel is the faster
design (matches the llm-scaler M-tile idea). Remaining levers: a 2-row variant for the
large-N FFN rows, and the llm-scaler qh pre-shuffle repack.

### fp16-FMA M-kernel variant - NOT a win (2026-08-30)

A fp16 variant (`GGML_SYCL_Q6K_MMVQ_ESIMD_F16`, default 0) dequants to fp16 + converts the
f32 activation to fp16, then does a fp16 FMA (fp16 multiply + fp16 accumulate, fp32 store).
The fp16 FMA is ~1.29x faster in isolation (32-wide eSIMD probe, FMA-bound: fp32 ~20 TF,
fp16 ~25.9 TF), but the **fp32->fp16 convert overhead** (round deq + activation to fp16, once
per token per block) eats the FMA savings. A/B on the B70 (i16-mtp-f16.bat, M=7, 2 runs each,
identical correct output, same acc len 6.00 + acc rate/pos (1,1,1,1,0.5,0.5)):
- fp32 M-kernel (ESIMD=1, F16=0): **31.76 / 31.8 t/s**
- fp16 M-kernel (ESIMD=1, F16=1): **31.57 / 31.6 t/s** -> ~0.2 t/s SLOWER, same accuracy

So fp16 is a net loss here: the B70's fp16 FMA advantage (1.29x) does not offset the convert
overhead for this access pattern (the activation is f32 in memory, so it must be converted
every time). fp16 would only win if the activation were already stored as fp16 (no convert).
Left in the tree behind the gate (default 0), uncommitted.

### bf16-FMA M-kernel variant - WORSE (2026-08-30)

bf16 is a near-free fp32 truncation (same 8-bit exponent, no range handling) - the CONVERT is
cheap, which is the right intuition. But the B70 (Xe2) has **no fast bf16 FMA path**: the
32-wide eSIMD FMA probe (same `acc += a*b` form as the M-kernel, fma_probe.exe, FMA-bound
iters=1e6) reads fp32=20056 GF, fp16=23590 GF (1.18x), **bf16=10405 GF (0.52x)**. So the
bf16 FMA is *half* the fp32 rate, unlike fp16 (1.18x). The bf16 M-kernel loses on both counts:
the slow bf16 FMA (0.52x) + the convert overhead. A/B (i16-mtp-f16.bat, ESIMD=1 F16=0 BF16=1,
M=7, same acc len 6.00): **25.8 t/s** (vs the fp32 M-kernel's 31.8). Far worse than fp16.
Gate `GGML_SYCL_Q6K_MMVQ_ESIMD_BF16` (default 0), uncommitted. On the B70 the fp16 FMA is the
only fast 16-bit FMA; bf16 is not.

NOTE: the M-kernel's low-precision path accumulates in T then reduces in fp32 (the bfloat16/
half simd `operator+` for the reduce is ambiguous; the scalar bfloat16 `operator+=` is
unsupported in the ESIMD context).

### Q5_K + Q8_0 M-kernel (templated) - the win (2026-08-30, committed)

The Q6_K M-kernel above only covered the Q6_K mat-vecs (63% of bytes). Template the kernel
on the quant type (`dequantize_mul_mat_vec_reorder_esimd_m<T,M>`; `using traits =
esimd_reorder_q_traits<T>`) and add a per-quant `dequant_block` (Q5_K: 4-bit base + 5th bit
from qh, factored out of the Q5_K `mac_pair`; Q8_0: `convert<float>(qs)*d` over 8 sub-blocks,
factored out of the Q8_0 `mac_pair`). Same kernel body; only the dequant differs. Routed by
three separate gates at the top of `ggml_sycl_op_mul_mat_vec_q` (mmvq.cpp):
`GGML_SYCL_Q6K_MMVQ_ESIMD`, `GGML_SYCL_Q5K_MMVQ_ESIMD`, `GGML_SYCL_Q80_MMVQ_ESIMD` (all
default 1, on). Committed `1fc95d577` (Q5_K) + `eff4e96e7` (Q8_0).

A/B on the B70 (n_max=6, -n 64, SYCL0, 2 runs each, identical correct output
"The capital of France is **Paris**.", same MTP mean acc len 6.00):
- Q6K M on, Q5K off (baseline): **30.20 / 30.32 t/s**
- Q6K M on, Q5K on:              **39.50 / 39.45 t/s**  -> **+9.2 t/s (+30%)**
- Q6K+Q5K on, Q80 off (baseline): **39.38 / 39.33 t/s**
- Q6K+Q5K on, Q80 on:            **39.85 / 39.91 t/s**  -> **+0.5 t/s (+1.3%)**

The Q5_K win is large (22% of the bytes now on the wide-SIMD M-kernel); the Q8_0 (lm_head)
win is small (the lm_head is a minor fraction of the 7-token verify) but consistent (the two
groups do not overlap; within-run variance ~0.06 t/s). Net MTP decode: **~39.9 t/s** (was
~31.8 fp32-M-only, ~28.9 pre-M-kernel). The fp16/bf16 variants remain stashed (not a win).

NOTE: MTP t/s is flag-dependent. i12 (temp 1.0, -n 64) reads ~27.9-28.85; the earlier i5
(temp 0.0, -n 128) read 30.8. Different sampling/length, not a regression - A/B within one
protocol.

## DRAM ceiling probe (and its traps)

Standalone SYCL read/write/copy probe (bwtest*.cpp in repo). Key traps:
- **Memory compression / caching fakes high read BW.** A "incompressible" fill that is
  actually a linear ramp (e.g. `(i*2654435761)*1e-8f` = constant delta) is trivially
  compressible -> reads served fast from compression, reporting 1000+ GB/s (>spec, bogus).
  Use a real hash (murmur3 finalizer -> random float bits) for the fill.
- A checksum (per-pass sum == analytic total) proves the VALUES are read, NOT that they
  came from DRAM (a cache also has the right values). Combine a hashed fill + checksum.
- Reduce-based DCE guards: a read loop whose result feeds only a never-taken branch is
  elided by the optimizer (0.4ms for a 16GB "read"). Force a live side effect.
- Real measured (hashed fill, clean stream kernel): read ~363 / write ~267 / copy ~324
  GB/s. These micro-probe numbers are LOWER than the 490 GB/s the real GEMV sustains
  (GEMV's coalesced pattern is better than a naive stream kernel). Trust the workload
  number (490 GB/s = 81% of spec), not the micro-probe.

## GEMV-shaped layout probe (bwtest5/6/7, 2026-08-27)

Probes that replicate the eSIMD DMMV access pattern exactly (4-lane WG, 2 rows/WG, lanes
stride 256-blocks by 4, per-block block_loads, y from L2) and compare SoA vs interleaved
layouts, per quant pattern. Traps hit:
- **Compile-time BPR is a trap:** `const int NCOLS = 5120` -> BPR=20 unrolls -> 584 GB/s
  SoA, ~90 GB/s above the real kernel. Pass ncols as a RUNTIME int so the loop trip count
  is not a constant expression. With runtime shape the probe matches the kernel (497 vs 505).
- **L2 residue is a trap:** a per-pass buffer <= ~2x L2 (16MB) reports >spec BW (1034 GB/s
  seen with 25.8MB/pass). Keep per-pass buffer >= ~3x L2 (>=50MB) or rows large enough.
- **Unloaded streams are a trap:** every SoA stream (incl. the 12-16B scales) must be
  actually loaded and consumed (reduce into the scalar weight) or BW is overstated ~7%.
- eSIMD gotchas: `convert<T>()` rejects `.select<>()` temps (materialize first); member
  template calls on dependent types need the `template` keyword (`v.template select<16,1>(0)`).
- Misaligned block_load is OK: the Q6_K interleaved tile is 210B (not 16B-aligned), the
  128B block_load still runs fast.
- Run via a bat that calls `setvars.bat intel64 vs2026` first (exe needs the oneAPI
  runtime DLLs on PATH) + `set ONEAPI_DEVICE_SELECTOR=level_zero:0`.
  `ssh b70 "...run-bwtest6.bat <nrows> <ncols> <passes>"` (bwtest7 adds a 4th arg pattern).

Model shapes (Qwen3.8-27B, 64 blocks = 48 GDN + 16 full-attn, every 4th block full-attn):
- GDN blocks: attn_qkv [5120,10240], attn_gate [5120,6144], ffn_up [5120,17408],
  ffn_gate [5120,17408] (Q5_K/IQ4_XS), ffn_down [17408,5120], ssm_out [6144,5120]
- Full-attn blocks: attn_q [5120,12288], attn_k/v [5120,1024], attn_output [6144,5120]
- output (lm_head) [5120,248320] Q8_0 (1.35GB/token); token_embd [5120,248320] Q6_K
  (only 1 row read at decode). A local copy of the model is at G:\tmp\Qwen3.8-27B-UD-Q6_K.gguf
  (gguf-inspect skill works on it).

## Prefill "reorder poison" (2026-08-27, pre-existing, affects serving not decode)

Any M<=8 pass in a process (decode, or even a 1-token prefill) triggers the in-place
weight REORDER (GGML_SYCL_ENABLE_OPT=1 default). After that, ALL later M>8 prefills in
the same process run 21% slower: standalone pp256 = 752 t/s, but pp8-then-pp256 = 589.
Steady-state serving = first request prefill fast, every later prefill at 589.

Diagnosis (GGML_SYCL_PROFILE per-op breakdown + GGML_SYCL_MKL_FA_DEBUG):
- The whole delta is in the FLASH_ATTN_EXT op: 312ms -> 402ms on the pp256 graph.
  MUL_MAT (dequant+f16 GEMM, 13.8 vs 14.2ms) and everything else is unchanged.
- The reorder-dequant kernels for the f16 path have IDENTICAL access patterns to the
  native-layout ones (same per-block offsets, different base) - not the cause.
- Both FA implementations degrade equally (oneDNN SDPA: 752->589; forced TILE via
  GGML_SYCL_FA_ONEDNN=0: 741->582) -> common per-call machinery, not oneDNN CPU.
- FA at M=256/KV=256 does ~2 GFLOP total but takes 312-402ms wall: per-call CPU/launch
  overhead dominates (~19-25ms per attention-layer call, 16 layers/graph).
- M=16-first (f16 GEMM, no reorder) does NOT poison; ENABLE_OPT=0 (no reorder) does NOT
  poison. The poison correlates exactly with the in-place reorder (20GB permute through
  sycl_ext_malloc_device temp churn). FA does not read weights, so the mechanism is
  GPU/runtime state after the churn, not the layout. Not transient (persists >1.5s of
  other GPU work), not JIT, not host fallback, not the oneDNN graph build.

Bench protocol note: the 750 t/s pp256 baselines were measured in standalone processes
(pp256 first). If a benchmark process runs an M<=8 test first, pp256 reports ~589 -
not a regression. Keep pp256 in its own process to compare against old logs.

Fix directions (not implemented): single reusable reorder temp buffer instead of
per-tensor L0 churn; or defer the first reorder until after the first M>8 prefill.
Either is an upstream-architecture change - needs an issue, not a local hack.

## Interleaved Q6_K: all three layouts lose to SoA (2026-08-30, EXHAUSTED)

Goal was to beat SoA 23.05 with a single-stream interleaved weight layout (the 2026-08-27
probe said 580-594 GB/s was available). Three real layouts were built, A/B'd, and all
lose. `GGML_SYCL_Q6K_INTERLEAVED` (default 0 = SoA) gates the whole family; the SoA
baseline is untouched by the experimental code.

| layout | mechanism | tg128 | verdict |
|--------|-----------|-------|---------|
| **210B** (in-place, `alignment<2>`) | read the native 210B tile, drop the injected `alignment<4>` | **20.66** | SLOW: the Xe2 backend does not lower 128B `alignment<2>` loads fast; misaligned 2-byte-stride tiles cost ~10% |
| **212B** (separate padded buffer) | copy weights into a 212B-stride USM buffer (4B-aligned tiles), eSIMD reads the padded copy | **2.55** | DEAD: 14.1GB padded buffer + 20.5GB model = 34.6GB > 32GB card. `ggml_sycl_malloc_device` (common.cpp:97) uses `ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE` -> Level Zero OVERSUBSCRIBES -> the buffer lands in host-backed (PCIe) memory ~56 GB/s. Non-split SYCL buffer is one big pool (ggml-sycl.cpp:566), so a per-tensor free/swap is impossible. |
| **420B pair210** (in-place supertile) | permute 2-block pairs into 420B tiles (see below); 420%4==0 so every field is 4B-aligned, zero extra VRAM | **22.60** | WASH-TO-LOSS: 420B straddles 7x64B cache lines (448B fetched per 420B useful = 6.7% extra DRAM traffic) vs SoA's zero-waste dense arrays. Structurally ~2% below SoA. |

B70 VRAM facts: 31906 MiB free at startup (32GB card). Model = 19625 MiB GPU + 995 MiB
CPU_Mapped. Any extra buffer > ~11 GiB risks the relaxed-allocation oversubscription above.

### 420B pair210 layout (the most promising, still a loss)
`[b0.ql:128][b1.ql:128][b0.qh:64][b1.qh:64][b0.scales:16][b1.scales:16][b0.d:2][b1.d:2]`
= 420 bytes per 2-block supertile. `reorder_qw_q6_k_pair210` (ggml-sycl.cpp:4419) permutes
in place via a temp buffer (1 supertile/work-item, byte-granular copy, guard
`size%420==0` and `ncols%(2*QK_K)==0`). `esimd_pair210_q6k_traits` (esimd.hpp:595) reads
block `i` from supertile `i/2`, sub `i&1`: ql@+oa*128, qh@+256+oa*64, scales@+384+oa*16,
**d@416+oa*2**.

**d-offset bug (found + fixed 2026-08-30):** the d field must sit at
`QK_K+QK_K/2+QK_K/16+QK_K/16` (=416, after BOTH 16-byte scale arrays). The first version
used `QK_K+QK_K/2+QK_K/16` (=400) in both the permute and the traits, which is self
-consistent (d reads back where it was written) BUT collides with `b1.scales`
(t+400..t+416): `b0.d`/`b1.d` overwrote `b1.scales[0..3]`, corrupting every odd block's
first 4 scales -> garbled output at 4.3 t/s on the 15-token-prompt cli flow (the bench's
8-token flow + the perplexity batch path never hit the decode reorder, so they looked
"fine" and gave no signal). Symptom of a layout-offset bug: garbled text but normal
kernel speed. Fixed at both sites; env=1 is correct again (still 22.60, still < 23.05).

### Reorder trigger gating (in-place layouts)
An in-place permute must only fire on the mat-vec path, or a later batched pass (ne[1]>1)
reads the permuted bytes as the original layout. Gating added: `opt_for_reorder` MMVQ
case skips Q6_K interleaved (ggml-sycl.cpp); the GLU path already requires ne[1]==1; MoE
returns false. A column-loop intercept in `ggml_sycl_mul_mat` serves a ne[1]>1 batch
column-by-column via the DMMV path if a tensor was already reordered (second-prompt
correctness). The one-time reorder cost (~2.5s for the 14GB byte-wise permute across ~200
Q6_K DMMV-side tensors) is amortized over the decode tokens: over 48 tokens (cli) it
dominates (4.3 t/s observed), over 256 (bench) it is ~2% (22.60).

### Log/SSH gotchas hit while debugging this
- llama-cli suppresses GGML_LOG at the default verbosity; this build maps
  `params.verbosity>=LOG_LEVEL_DEBUG ? DEBUG : ERROR` (common.cpp:1326), so you need
  `-v -v` (DEBUG) to see the INFO reorder/dmmv lines. The `GGML_LOG_LEVEL` env var is
  NOT read in this build.
- The perplexity tool (batch, ne[1]>=1) NEVER triggers the ne[1]==1 decode reorder, so it
  is useless for validating a decode-only layout change (env=0 vs env=1 PPL will be
  identical). Validate with a single-token decode flow (llama-cli -p ... -n 48).
- A 15-token prompt goes the MMQ/XMX path (ne[1] 9..32); an 8-token prompt goes MMVQ
  (ne[1] 2..8). They leave different GPU state and different reorder timing - always A/B
  the SAME prompt length.


## Lessons / gotchas

- **Partial deploy after a reconfigure = 0xC0000409 crash misread as a kernel bug.**
  After `cmake-sycl-oneapi.bat`, ALL ggml-*/llama-*.dll change together. Deploy only the
  changed .dll/.exe and the stale companions on the B70 are ABI-inconsistent -> the exe
  prints "Loading model..." then dies EXIT=-1073740791. Deploy the whole bin. (See
  sycl16-build-deploy.)
- `git stash` of dmmv.cpp reverts to the committed state which does NOT compile (Q5_K
  7-arg vs 8-arg template). The working baseline includes the uncommitted Q5_K fix +
  row-kernel block_load. Don't stash to get a "baseline".
- Build wrapper must set `LEVEL_ZERO_V1_SDK_PATH` so any mid-build cmake re-configure
  keeps the Level Zero include (done in build-sycl16-oneapi.bat).
- oneAPI 1.2.1 API: `sycl::reduce_over_group` (not reduce_work_group),
  `atomic_ref<T,order,scope>` (3 params), `malloc_host<T>(n, q)`.
- **SSH to the B70: never pipe llama-cli through `| tail`/`| grep` in the local shell.**
  llama-cli stays at the interactive REPL after `-n` tokens; the ssh session never
  closes, `tail` buffers until EOF -> looks like "no output" for the full timeout, and
  the REMOTE process keeps running after the local ssh dies (orphan holding GPU + CPU).
  Use the detached pattern: stage a .bat (scp it; inline `start /min cmd /c` quoting
  breaks on the -p "prompt with spaces"), then
  `start /min cmd /c "i1-test.bat > i1.log 2>&1"`, poll `type i1.log` from a 2nd ssh.
  Two concurrent 27B loads can wedge the box (sshd starved; ping still answers) - never
  launch a second run while the first may still be alive.
