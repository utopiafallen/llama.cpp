---
name: b70-decode-perf
description: Qwen3.8-27B Q6_K decode performance findings on the Intel Arc Pro B70 (Battlemage) - DRAM ceiling, what was tried, what works. Use when optimizing SYCL decode t/s on the B70 or deciding whether a perf target is reachable on this hardware.
---

# B70 Qwen3.8-27B Q6_K decode perf findings

Model: `Qwen3.8-27B-UD-Q6_K.gguf` (20.46 GiB, dense), B70 = Intel Arc Pro B70
(32GB GDDR6, 256-bit @ 19 Gbps = 608 GB/s spec, 16MB L2). Bench:
`llama-bench -r 2 --device SYCL0 -m <model> -p 8 -n 128` (see sycl16-build-deploy).

## Bottom line

- **Best plain decode = ~23.05 t/s** (Q6_K/Q5_K/Q8_0 eSIMD DMMV + 1-row small-N kernel +
  f16-convert skip, `76f80b445`). ~506 GB/s = 83% of spec; the practical plain-decode ceiling.
- **Best decode = MTP speculation.** `--spec-type draft-mtp --spec-draft-n-max 6` gave ~30.8 t/s
  (+34% vs plain) as of 2026-08-30, pushed to ~39.9 by the eSIMD M-kernel work (see "MTP"
  section). It amortizes the 20.46 GB main-model read over accepted tokens - the only route past
  the DRAM ceiling. n_max=8 collapses to 4.8.
- **Interleaved Q6_K weight layouts = DEAD END** (2026-08-30): every real variant loses to SoA
  23.05; the 2026-08-27 probe that predicted 25-26 t/s over-predicted. Do NOT retry any
  interleaved/padded/pair Q6_K layout for decode.
- **XMX decode FA = a long-context ATTENTION lever, not a headline-decode one.** ~8% faster than
  tile FA at 32K (20.5 vs 18.9 t/s); a wash at short context. Decode FA was COMPUTE-bound (EU FMA
  QK^T/PV), not DRAM-bound; XMX offloads that compute so the attention hits the KV-read floor.
  Decode-only (prefill, the bigger XMX target, is untouched). See the "XMX" section.

## Quant distribution (gguf-inspect)

| type  | bytes   | frac  | kernel path (decode)       |
|-------|---------|-------|----------------------------|
| Q6_K  | 13.89GB | 63.2% | eSIMD DMMV (2-row)         |
| Q5_K  | 4.93GB  | 22.4% | eSIMD DMMV (2-row, #26376) |
| Q8_0  | 2.85GB  | 13.0% | eSIMD DMMV                 |
| IQ4_XS| 142MB   | 0.6%  | (negligible, not touched)  |
| Q4_K  | 100MB   | 0.5%  | (negligible, not touched)  |

output/embed = Q6_K, vocab 248320. Q6_K+Q5_K = 85.6% of bytes, both on the same eSIMD DMMV
structure (generic `dequantize_mul_mat_vec_reorder_esimd` in dmmv.cpp, per-quant `mac_pair` in
esimd.hpp).

## The eSIMD DMMV kernel

- work-group = 4 lanes (`GGML_SYCL_DMMV_ESIMD_WG_SIZE`), owns 2 consecutive output rows.
- loop over K-blocks (QK_K=256): lane `tid` handles blocks `tid, tid+4, ...`; the 4 lanes cover
  4 consecutive blocks = 512B of `ql` per row per iteration (coalesced).
- per block: `block_load<float,256>` of the activation `y` + per-quant dequant-MAC of the 2 rows.
  SoA (reordered) layout: ql/qh/scales are separate contiguous arrays.
- epilogue: per-lane partials -> lmem -> lane 0 reduces -> dst. GLU fusion = same but 4
  accumulators (gate/up x 2 rows) + silu(g)*u epilogue.

## Experiments tried (A/B vs 22.33 t/s baseline)

| change | idea | result | why |
|--------|------|--------|-----|
| **2-row (baseline)** | - | **22.33** | best |
| ROWS=4 (4 rows/WG) | amortize the shared `y` load over 2x rows | 22.13 | `y` is L2-cached (hidden-dim 16-32KB << 16MB L2), so cutting redundant `y` loads cuts L2 traffic, NOT the DRAM weight stream. No DRAM win. |
| 2-block/lane-iter | wider contiguous weight stream (1KB ql vs 512B) | 22.22 | marginal; already well-coalesced |
| row-per-lane (`GGML_SYCL_Q6K_GEMV_ROW=1`) | 1 lane/row, 512-elem tiles (llm-scaler) | 20.39 | slower than our 2-row SoA kernel |
| `GGML_SYCL_ENABLE_ESIMD=0` (MMVQ) | dp4a path | 20.84 | eSIMD DMMV is faster |
| SYCL graph mode | - | 17.93 | -16%, avoid |
| 2-GPU `-ts 54,46` | split layers | 20.36 | worse (cross-GPU sync) |

Conclusion: the 2-row SoA eSIMD DMMV already beats Intel's reference row kernel and is at the
practical DRAM ceiling. Do not re-try ROWS>2 or wider loads.

## Q8_0 eSIMD path (+0.5 t/s)

Q8_0 (13% of bytes) was the only non-negligible quant NOT on the eSIMD DMMV path (used the
reorder-MMVQ dp4a kernel). Moved to eSIMD DMMV: **+0.5 t/s** (22.87 vs 22.34 cool, 22.63 vs 22.08
warm). Only decode (ne[1]==1) uses it; multi-token still MMVQ, so pp8 is unaffected.

Feasible because the machinery existed: `reorder_qw_q8_0` already produces the SoA layout
(`[qs: nrows*ncols int8][d: nrows*(ncols/32) half]`, d stored half, min dropped - streams fewer
bytes than block_q8_0), and Q8_0 was already reordered (GGML_SYCL_ENABLE_OPT=1 default), so the
eSIMD path just reads the same SoA layout.

Added: `esimd_reorder_q_traits<Q8_0>` (make_ptrs + mac_pair; one mac_pair = a 256-elem chunk =
8 Q8_0 blocks; loads qs as two `block_load<int8_t,128>` (max 8-bit width is 128), 8 half scales,
dequant `convert<float>(qs)*d`; NOTE `convert` rejects a `.select<32,1>()` temp - store in a named
`simd<int8_t,32>` first), a dmmv.cpp wrapper (calls generic `dequantize_mul_mat_vec_reorder_esimd
<Q8_0,false>`), the DMMV dispatch case + `ggml_sycl_supports_reorder_esimd` entry, and env
`GGML_SYCL_Q80_GEMV_ESIMD` (default 1, opt-out). Safety: reached only when
`optimized_feature.reorder` is true, so non-reordered tensors (MoE - Q8_0 not in the MoE reorder
list; or ENABLE_OPT=0) fall back to the original-layout DMMV. No corruption risk.

Q5_K (22%, also eSIMD DMMV 2-row) has no further win - same kernel, at the ceiling. IQ4_XS/Q4_K
(~1% total) not worth touching.

## 2026-08-29 session

Profile (GGML_SYCL_PROFILE tensor GB/s) of the ~23.0 t/s build:
- FFN q6_K/q5_K (fused GLU/add): 516-570 (saturated).
- attn (attn_gate 6144x5120, ssm_out 5120x6144): 380-500.
- **iq4_nl 17408x5120: 92.6 (biggest single inefficiency, ~2.1% of time).**
- iq4_xs 17408x5120: 260-280.
- q8_0 48x5120 (ssm_beta/alpha): 23.6 (M=48 too small to saturate).
- LM head q8_0 248320x5120: 590.

### 1-row small-N eSIMD DMMV kernel (+0.15 t/s, committed)
`dequantize_mul_mat_vec_reorder_esimd_row1<T,ADD_RES>` (dmmv.cpp): 1 output row/WG (vs 2-row).
Below `GGML_SYCL_DMMV_ESIMD_ROW1_MAX_NROWS=8192` rows, the 2-row layout (2*N subgroups) can't hide
the sequential per-subgroup K-load latency; 1-row doubles the subgroup count (4*N). Dispatch
(`dequantize_mul_mat_vec_reorder_esimd_dispatch<T>`) picks 1-row (nrows<=8192) else 2-row. Also:
the eSIMD q8_0 kernel eats f32 `y` directly, so the src1 f32->f16 convert is skipped when that
path is active (DMMV dispatch).

### 2-matrix same-input fused GEMV (WASH - kept, gated)
attn qkv + qkv_gate (and up+gate) are two MUL_MATs on the SAME input vector; merging into one
launch was expected to improve DRAM concurrency. Built `dequantize_mul_mat_vec_2matrix_reorder_esimd
<TA,TB>` (dmmv.cpp, 2 rows/WG, per-WG A/B branch, 2 dst outputs), `ggml_sycl_dmmv_2matrix_reorder_esimd`
dispatch (Q4_K/Q5_K/Q6_K/Q8_0), detection `ggml_sycl_mul_mat_pair_same_input_fused` (ggml-sycl.cpp:
matching mmB within 4 nodes, same src[1]/K, only noops between; M_A even guard), env
`GGML_SYCL_FUSE_MM_PAIR` (default 1). **WASH (22.91-22.92 vs 22.95, both 1-row and 2-row):** the B70
already pipelines the two separate launches and saturates DRAM; GB/s is not limited by rows-in-a-
single-launch.

### iq4_nl SoA reorder + MMVQ kernel (WASH - reorder NOT triggered on B70)
iq4_nl (92.6 GB/s, 32-elem blocks, packed 18B) -> SoA (`[qs: nb*16][d: nb*half]`) + MMVQ dp4a
kernel (each thread a whole block, no eSIMD byte-gather). The dp4a dequant (get_int_from_table_16 +
dp4a) beats an eSIMD software gather for the non-linear iq4_nl lookup. Built `reorder_qw_iq4_nl`
(ggml-sycl.cpp), `vec_dot_iq4_nl_q8_1_soa` (vecdotq.hpp), `mul_mat_vec_q_iq4_nl_q8_1_soa` + `_soa_sycl`
(mmvq.cpp), reorder support + flag check in the dispatch. Correct ("Paris") but **WASH (23.00/22.93
vs 23.0) because the reorder NEVER fires** (a one-time log in `reorder_qw_iq4_nl` doesn't trigger).
iq4_nl isn't in the GLU fusion whitelist (fusion.cpp:38 = Q4_K/Q5_K/Q6_K only), so it goes the main
dispatch (which DOES call opt_for_reorder(MMVQ)) - yet the reorder still doesn't fire (some gating
condition is unverified). Root cause of the slowness: 32-elem blocks (160/row for N=5120 vs iq4_xs
20/row) = 8x more per-block scale reads + loop overhead; the SoA layout doesn't change block size,
so it can't fix it. iq4_nl is hard (small blocks are inherent to the quant).

Conclusion: main levers exhausted. FFN saturated; 2-matrix merge wash; iq4_nl hard + its reorder
doesn't even fire; q8_0 48x5120 hard (M too small). **~23.0 t/s is the plain-decode ceiling.** The
interleaved-layout path (bottom line) turned out a DEAD END (see below). The route past the ceiling
is MTP, verified working:

## MTP (NextN) decode - the win (2026-08-30)

Qwen3.8 (arch `qwen35`) ships one MTP block: gguf `qwen35.nextn_predict_layers = 1` +
`blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}` (eh_proj [10240,5120] Q6_K). llama.cpp fully
supports it (`common_speculative_impl_draft_mtp` in common/speculative.cpp, `graph_mtp` in
src/models/qwen35.cpp) but it is OFF by default: the MTP tensors load only when `mparams.load_mtp`
is set (happens only when the spec type includes `draft-mtp`); otherwise the loader logs "unused
tensor blk.64.nextn.* -- ignoring".

Enable:
    llama-cli --spec-type draft-mtp --spec-draft-n-max 6 -m <model> ...
(`--spec-draft-n-max` sets draft.n_max, default 3. Single MTP head, so n_max is NOT clamped to 1 -
that clamp is only the multi-head "chain_heads"/step35 mode.)

Key facts:
- The MTP draft ctx SHARED the main weights (`shares_model = !has_draft`, common.cpp:1317) - no
  second 20GB copy. Memory = main + small MTP block + both KV. Fits the 32GB B70.
- Mechanism: each step runs the cheap 1-layer MTP block up to n_max times to draft, then ONE
  main-model pass (20.46 GB) verifies -> accepted drafts give >1 token per 20.46 GB read.
- **Measured (correct "Paris" all runs):** n_max=4 -> 30.0 t/s; **n_max=6 -> 30.8 t/s
  (RECOMMENDED)**; n_max=8 -> 4.8 (COLLAPSE: long chains degrade the single block's 1-step-ahead
  prediction -> most drafts rejected while still paying up to 8 cheap-but-real MTP passes; that
  run also hit box instability - never use n_max>6). vs plain 23.05 -> **+34%**.
- Box was mildly degraded this session (plain read 22.4 not 23.05), so true MTP on a healthy box
  is a few % higher. Re-confirm on a stable box.

Test bat: i5-test.bat (`--spec-type draft-mtp --spec-draft-n-max 4 -n 128 --temp 0.0
--no-display-prompt < NUL`; the `< NUL` gives stdin EOF so it exits after generation, no REPL hang
/ orphan).

## MTP verify bottleneck + Q6_K small-batch M-kernel (2026-08-30)

The MTP win amortizes the 20.46 GB main-model verify over accepted tokens.
`common_speculative_print_stats` (speculative.cpp, needs `-v -v -v`):
- **Acceptance is GOOD, not the bottleneck:** `#mean acc len` ~5-6/step, `#acc rate/pos`
  ~(1.0, 0.83, 0.83, 0.67, 0.5, 0.33). Do NOT chase draft quality for t/s.
- Step ~145.5 ms: the cheap ~6 MTP draft passes ~24 ms (422 GB/s); the ONE 7-token main-model
  verify ~120 ms = **the bottleneck (~86%)**.

The 7-token verify goes the SoA Q6_K small-batch path (ne[1] 2-8):
`reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>` -> `mul_mat_vec_q_reorder_ncols_q6_k<7>` (mmvq.cpp),
**compute-bound on the Q6_K dot** (2 dp4a + ~6 scalar-FP per token per block-elem; the ~14-op
weight dequant is shared + CSE'd by the compiler). Batch ridge (llama-bench -pg): tg1=506, pp6=202,
pp12=58, pp512=40 GB/s - GB/s collapses as batch grows (memory-bound -> compute-bound).

Micro-opts A/B'd (i12, n_max=6, `GGML_SYCL_Q6K_MMVQ_HOIST` 1 vs 0):
- **Dequant hoist** (default 1): hand-hoist vi0/vi1 out of the per-token loop = **NO-OP** (compiler
  already CSEs the shared dequant). Kept (harmless, gated).
- **Scale fold**: precompute w0=d*sc0, w1=d*sc1 per (block,elem); per-token tail
  `d80*dp4a(vi0,u0)*w0 + d81*dp4a(vi1,u1)*w1` (drops the *sc int-mul and final *d) = **+~1 t/s**
  (28.8-28.9 vs 27.9, reproducible x3, correct "Paris"). Compiler won't regroup without fast-math.

### eSIMD wide-SIMD M-kernel (M=2-8) - the win (+8%)
The dp4a MMVQ does a per-token int8 dot and is dp4a-throughput-bound at M=7. Replace it with the
M=1 eSIMD wide-SIMD design extended to M accumulators: `dequantize_mul_mat_vec_reorder_esimd_m_q6k<M>`
(dmmv.cpp) dequants ONE weight row into 8x32-wide fp32 `deq` (new Q6_K `dequant_block` in esimd.hpp,
factored from `mac_pair`), then per token `acc[m] += y_vec.select<32,1>(32g) * deq[g]` on the
**f32 activation directly** (no Q8_1 quantize; `src1_ddf_i` is always allocated). Weight dequant
amortized across M; full-width eSIMD multiply (no int8 dp4a, no scalar scale ops). A/B (i15, M=7,
n_max=6, -n 64, `GGML_SYCL_Q6K_MMVQ_ESIMD` 1 vs 0): **31.18/31.20 vs 28.86/28.86 = +8.1%** (identical
correct "Paris", same MTP mean acc len 6.0). Gate default 1; requires Q6_K + F32 + reorder +
!interleaved + ncols 2-8, routed at the top of `ggml_sycl_op_mul_mat_vec_q` before the per-token
loop. lm_head is Q8_0 (unaffected); covers FFN + attention mat-vecs (bulk of verify compute);
registers fine at M=7. So the "dp4a-bound, exhausted" verdict is SUPERSEDED: for the compute-bound
small-batch verify, the wide-SIMD fp32 dequant-amortized-over-M kernel is the faster design
(matches the llm-scaler M-tile idea). Remaining levers: a 2-row variant for large-N FFN, and the
llm-scaler qh pre-shuffle repack.

### fp16 / bf16 M-kernel variants - NOT wins (2026-08-30)
- **fp16** (`GGML_SYCL_Q6K_MMVQ_ESIMD_F16`, default 0): fp16 FMA is ~1.29x faster in isolation
  (32-wide eSIMD probe, FMA-bound: fp32 ~20 TF, fp16 ~25.9 TF), but the **fp32->fp16 convert**
  (activation is f32 in memory, converted every time) eats the savings -> **~0.2 t/s SLOWER**
  (31.57/31.6 vs 31.76/31.8, i16, M=7, same acc len 6.00). Would only win if the activation were
  already stored fp16. Left behind the gate, uncommitted.
- **bf16** (`..._BF16`, default 0): bf16 is a near-free fp32 truncation (cheap convert - the right
  intuition), but the B70 (Xe2) has **no fast bf16 FMA**: the probe reads fp32=20056, fp16=23590
  (1.18x), **bf16=10405 GF (0.52x)**. So the bf16 FMA is HALF the fp32 rate -> **25.8 t/s**, far
  worse than fp16 (loses on both the slow FMA and the convert). On the B70 the fp16 FMA is the only
  fast 16-bit FMA; bf16 is not.
- NOTE: the M-kernel low-precision path accumulates in T then reduces in fp32 (the bfloat16/half
  simd `operator+` for the reduce is ambiguous; the scalar bfloat16 `operator+=` is unsupported in
  the ESIMD context).

### Q5_K + Q8_0 M-kernel (templated) - the win (2026-08-30, committed)
Templatize the kernel on the quant (`dequantize_mul_mat_vec_reorder_esimd_m<T,M>`,
`using traits = esimd_reorder_q_traits<T>`) + add a per-quant `dequant_block` (Q5_K: 4-bit base +
5th bit from qh, factored from the Q5_K mac_pair; Q8_0: `convert<float>(qs)*d` over 8 sub-blocks,
factored from the Q8_0 mac_pair). Same body, only dequant differs. Routed by three gates at the top
of `ggml_sycl_op_mul_mat_vec_q`: `GGML_SYCL_Q6K/Q5K/Q80_MMVQ_ESIMD` (all default 1). Committed
`1fc95d577` (Q5_K) + `eff4e96e7` (Q8_0). A/B (n_max=6, -n 64, SYCL0, identical "Paris", same acc len 6.00):
- Q6K on, Q5K off: 30.20/30.32  ->  Q5K on: **39.50/39.45 (+9.2, +30%)**
- +Q5K, Q80 off: 39.38/39.33    ->  Q80 on: **39.85/39.91 (+0.5, +1.3%)**

Q5_K is large (22% of bytes now on wide-SIMD M); Q8_0 (lm_head) small but consistent (groups don't
overlap; within-run variance ~0.06 t/s). Net MTP decode **~39.9 t/s** (was ~31.8 fp32-M-only,
~28.9 pre-M-kernel). fp16/bf16 remain stashed (no win). NOTE: MTP t/s is flag-dependent - i12
(temp 1.0, -n 64) ~27.9-28.85 vs i5 (temp 0.0, -n 128) 30.8; different sampling/length, not a
regression, A/B within one protocol.

## DRAM ceiling probe (and its traps)

Standalone SYCL read/write/copy probe (bwtest*.cpp in repo). Traps:
- **Compression/caching fakes high read BW:** a linear-ramp fill (e.g. `(i*2654435761)*1e-8f`) is
  trivially compressible -> reads served fast, reporting 1000+ GB/s (>spec, bogus). Use a real hash
  (murmur3 finalizer -> random float bits).
- A checksum (per-pass sum == analytic total) proves the VALUES are read, NOT that they came from
  DRAM (a cache also has the right values). Combine a hashed fill + checksum.
- **DCE:** a read loop whose result feeds only a never-taken branch is elided (0.4ms for a 16GB
  "read"). Force a live side effect.
- Real (hashed, clean stream kernel): read ~363 / write ~267 / copy ~324 GB/s - LOWER than the 490
  the real GEMV sustains (GEMV's coalesced pattern beats a naive stream). Trust the workload number
  (490 = 81% of spec), not the micro-probe.

## GEMV-shaped layout probe (bwtest5/6/7, 2026-08-27)

Replicates the eSIMD DMMV access pattern exactly (4-lane WG, 2 rows/WG, lanes stride 256-blocks by
4, per-block block_loads, y from L2); SoA vs interleaved, per quant. Traps:
- **Compile-time BPR:** `const int NCOLS=5120` -> BPR=20 unrolls -> 584 GB/s SoA (~90 above the real
  kernel). Pass ncols as a RUNTIME int so the trip count isn't a constant expression (then 497 vs 505,
  matches).
- **L2 residue:** a per-pass buffer <= ~2x L2 (16MB) reports >spec BW (1034 GB/s seen at 25.8MB/pass).
  Keep per-pass >= ~3x L2 (>=50MB) or large rows.
- **Unloaded streams:** every SoA stream (incl. the 12-16B scales) must be loaded + consumed (reduce
  into the scalar weight) or BW is overstated ~7%.
- eSIMD gotchas: `convert<T>()` rejects `.select<>()` temps (materialize first); member template calls
  on dependent types need the `template` keyword (`v.template select<16,1>(0)`).
- Misaligned block_load is OK: the Q6_K interleaved tile is 210B (not 16B-aligned), the 128B
  block_load still runs fast.
- Run via a bat that calls `setvars.bat intel64 vs2026` first (exe needs the oneAPI runtime DLLs on
  PATH) + `set ONEAPI_DEVICE_SELECTOR=level_zero:0`; `ssh b70 "...run-bwtest6.bat <nrows> <ncols>
  <passes>"` (bwtest7 adds a 4th arg = pattern).

Model shapes (Qwen3.8-27B, 64 blocks = 48 GDN + 16 full-attn, every 4th block full-attn):
- GDN blocks: attn_qkv [5120,10240], attn_gate [5120,6144], ffn_up [5120,17408], ffn_gate [5120,17408]
  (Q5_K/IQ4_XS), ffn_down [17408,5120], ssm_out [6144,5120].
- Full-attn blocks: attn_q [5120,12288] is a JOINT Q+gate projection (2 x 24 heads x 256 head_dim =
  12288; query = first 6144, gate = second 6144, gated attention - qwen35.cpp:266), attn_k/v
  [5120,1024] (4 KV heads x 256), attn_output [6144,5120].
- Attention heads (GGUF kv qwen35.attention.*): head_count=24 (query), head_count_kv=4, head_dim=256,
  gqa=6. DO NOT read 12288/256=48 as the head count - that is the Q+gate width; the query alone is
  6144 (24 heads). This is why the XMX decode FA gate (gqa 1-8) PASSES for this model.
- output (lm_head) [5120,248320] Q8_0 (1.35GB/token); token_embd [5120,248320] Q6_K (only 1 row read
  at decode). Local model copy: G:\tmp\Qwen3.8-27B-UD-Q6_K.gguf (gguf-inspect works on it).

## Prefill "reorder poison" (2026-08-27, pre-existing, affects serving not decode)

Any M<=8 pass in a process (decode, or even a 1-token prefill) triggers the in-place weight REORDER
(GGML_SYCL_ENABLE_OPT=1 default). After that ALL later M>8 prefills in the same process run ~21%
slower: standalone pp256 = 752 t/s, but pp8-then-pp256 = 589. Steady-state serving = first prefill
fast, every later prefill at 589.

Diagnosis (GGML_SYCL_PROFILE per-op + GGML_SYCL_MKL_FA_DEBUG):
- The whole delta is in the FLASH_ATTN_EXT op: 312 -> 402 ms on the pp256 graph. MUL_MAT
  (dequant+f16 GEMM, 13.8 vs 14.2 ms) and everything else is unchanged.
- The reorder-dequant f16 kernels have IDENTICAL access patterns to the native-layout ones (same
  per-block offsets, different base) - not the cause.
- Both FA impls degrade equally (oneDNN SDPA 752->589; forced TILE via GGML_SYCL_FA_ONEDNN=0
  741->582) -> common per-call machinery, not oneDNN CPU.
- FA at M=256/KV=256 does ~2 GFLOP but takes 312-402 ms wall: per-call CPU/launch overhead dominates
  (~19-25 ms per attention-layer call, 16 layers/graph).
- M=16-first (f16 GEMM, no reorder) and ENABLE_OPT=0 (no reorder) do NOT poison. Correlates exactly
  with the in-place reorder (20GB permute through sycl_ext_malloc_device temp churn). FA doesn't read
  weights, so it's GPU/runtime state after the churn, not the layout. Not transient (persists >1.5s of
  other GPU work), not JIT, not host fallback, not the oneDNN graph build.

Bench note: the ~750 t/s pp256 baselines were standalone (pp256 first). A process that runs an M<=8
test first reports ~589 - not a regression. Keep pp256 in its own process to compare against old logs.

Fix (not implemented): single reusable reorder temp buffer instead of per-tensor L0 churn; or defer
the first reorder until after the first M>8 prefill. Either is an upstream-architecture change - needs
an issue, not a local hack.

## Interleaved Q6_K: all three layouts lose to SoA (2026-08-30, EXHAUSTED)

Goal: beat SoA 23.05 with a single-stream interleaved layout (the 2026-08-27 probe said 580-594 GB/s
was available). Three real layouts built + A/B'd; all lose. `GGML_SYCL_Q6K_INTERLEAVED` (default 0 =
SoA) gates the family; the SoA baseline is untouched by the experimental code.

| layout | mechanism | tg128 | verdict |
|--------|-----------|-------|---------|
| **210B** (in-place, `alignment<2>`) | read the native 210B tile, drop the injected `alignment<4>` | **20.66** | SLOW: the Xe2 backend doesn't lower 128B `alignment<2>` loads fast; misaligned 2-byte-stride tiles cost ~10% |
| **212B** (separate padded buffer) | copy weights into a 212B-stride USM buffer (4B-aligned tiles), eSIMD reads the padded copy | **2.55** | DEAD: 14.1GB padded + 20.5GB model = 34.6GB > 32GB. `ggml_sycl_malloc_device` (common.cpp:97) uses `ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE` -> L0 OVERSUBSCRIBES -> buffer lands host-backed (PCIe) ~56 GB/s. Non-split SYCL buffer is one big pool (ggml-sycl.cpp:566), so a per-tensor free/swap is impossible. |
| **420B pair210** (in-place supertile) | permute 2-block pairs into 420B tiles (420%4==0 so every field is 4B-aligned, zero extra VRAM) | **22.60** | WASH-TO-LOSS: 420B straddles 7x64B cache lines (448B fetched per 420B useful = 6.7% extra DRAM) vs SoA's zero-waste dense arrays. Structurally ~2% below SoA. |

VRAM: 31906 MiB free at startup (32GB card); model 19625 MiB GPU + 995 CPU_Mapped. Any extra buffer
> ~11 GiB risks the relaxed-allocation oversubscription above.

### 420B pair210 layout (the most promising, still a loss)
`[b0.ql:128][b1.ql:128][b0.qh:64][b1.qh:64][b0.scales:16][b1.scales:16][b0.d:2][b1.d:2]` = 420 bytes
per 2-block supertile. `reorder_qw_q6_k_pair210` (ggml-sycl.cpp:4419) permutes in place via a temp
buffer (1 supertile/work-item, byte-granular copy, guard `size%420==0` and `ncols%(2*QK_K)==0`).
`esimd_pair210_q6k_traits` (esimd.hpp:595) reads block i from supertile i/2, sub i&1: ql@+oa*128,
qh@+256+oa*64, scales@+384+oa*16, **d@416+oa*2**.

**d-offset bug (found + fixed):** d must sit at `QK_K+QK_K/2+QK_K/16+QK_K/16` (=416, after BOTH
16-byte scale arrays). The first version used `QK_K+QK_K/2+QK_K/16` (=400) in both the permute and
the traits: self-consistent (d reads back where it was written) BUT collides with `b1.scales`
(t+400..t+416): b0.d/b1.d overwrote b1.scales[0..3], corrupting every odd block's first 4 scales ->
**garbled output at 4.3 t/s** on the 15-token-prompt cli flow (the bench's 8-token flow + the
perplexity batch path never hit the decode reorder, so they looked "fine" and gave no signal).
Symptom of a layout-offset bug: garbled text but normal kernel speed. Fixed at both sites; env=1 is
correct again (still 22.60, still < 23.05).

### Reorder trigger gating (in-place layouts)
An in-place permute must fire ONLY on the mat-vec path, or a later batched pass (ne[1]>1) reads the
permuted bytes as the original layout. `opt_for_reorder` MMVQ case skips Q6_K interleaved
(ggml-sycl.cpp); the GLU path already requires ne[1]==1; MoE returns false. A column-loop intercept in
`ggml_sycl_mul_mat` serves a ne[1]>1 batch column-by-column via the DMMV path if a tensor was already
reordered (second-prompt correctness). The one-time reorder cost (~2.5s for the 14GB byte-wise permute
across ~200 Q6_K tensors) is amortized over the decode tokens: over 48 tokens (cli) it dominates
(4.3 t/s), over 256 (bench) it is ~2% (22.60).

### Log/SSH gotchas hit while debugging this
- llama-cli suppresses GGML_LOG at the default verbosity; this build maps
  `params.verbosity>=LOG_LEVEL_DEBUG ? DEBUG : ERROR` (common.cpp:1326), so you need `-v -v` (DEBUG)
  to see the INFO reorder/dmmv lines. The `GGML_LOG_LEVEL` env var is NOT read in this build.
- The perplexity tool (batch, ne[1]>=1) NEVER triggers the ne[1]==1 decode reorder, so it is useless
  for validating a decode-only layout change (env=0 vs env=1 PPL identical). Validate with a
  single-token decode flow (llama-cli -p ... -n 48).
- A 15-token prompt goes the MMQ/XMX path (ne[1] 9..32); an 8-token prompt goes MMVQ (ne[1] 2..8).
  Different GPU state + reorder timing - always A/B the SAME prompt length.

## XMX (joint_matrix) decode flash-attention (2026-09-02)

Decode-only FA that runs QK^T and PV on the Intel XMX matrix engine (joint_matrix) instead of the EU
FMA loop. Files: fattn-xmx-decode.{cpp,hpp}, dispatch in fattn.cpp before the best-kernel switch
(kernel committed `9ad5d30eb`). Enable env `GGML_SYCL_FA_XMX_DECODE` defaults 1, read ONCE at startup
into `g_ggml_sycl_fa_xmx_decode` (not per dispatch). The support gate also requires
`sycl_device_info.has_xmx` (populated at init via `gpu_has_xmx`), so default-on is safe on non-XMX
GPUs (iGPU) - it just falls back to tile/vec. The device banner prints an `XMX|Y/N` column. Correct;
~8% at 32K; a wash at short context.

### B70 XMX hardware (runtime-confirmed)
- The B70 is bmg_g31 (arch low-word 0x00800000 = 8388608), NOT the bmg_g21 the AOT build targets.
  bmg_g21 AOT binaries run fine on bmg_g31 (same Battlemage family).
- 53 matrix_combinations (the iGPU has 0) - XMX present. fp16 tiles: M=1-8 (variable), N=16, K=16;
  M=16, N=16, K=16; M=1/32, N=64, K=16/32. The M=1-8 variable tile is what batches the gqa queries.
- Dense 1024^3 GEMM (16x16x16 tiles): XMX 27 TFLOPS fp16 vs 2.0 naive scalar = 13.2x (JIT == AOT).

### Kernel design
- Main kernel grid (n_splits, n_kv_heads, 16); one 16-lane sub_group per work-group. One work-group =
  one 256-position KV split x one KV head. All gqa queries (Qwen3.8-27B gqa=6) batch into a single
  M=gqa XMX tile so the K/V row loads once and is reused across the group.
- QK^T: scores[gqa][256] = Q16[gqa][256] @ K[256][256]^T, per 16-pos chunk (A=Q16 row_major from LDS;
  B=K global col_major, stride = pos stride, giving B[d][pos]=K[pos][d]). Online softmax over the split
  (unnormalized P + m + l in regs/LDS). PV: O[gqa][256] += P[gqa][16] @ V[16][16], per 16-dim chunk
  (A=P local, B=V global). Combine kernel: flash-decoding merge of the per-split partials (per q_head
  x dim).
- Support gate: decode (Q ne[1]==1), Q F32 / K,V F16, head_dim==256, gqa 1-8, single batch (ne[3]==1),
  no sinks / no logit_softcap. n_splits = ceil(n_kv/256). LDS ~12 KB/WG at gqa=6 (Q16 3KB + scores 6KB
  + P16 3KB).

### joint_matrix API gotchas
- joint_matrix<sub_group, T, use::a/b/accumulator, M, N, layout> + _load/_fill/_mad/_store. a/b load
  layout is a template param; acc store layout is a runtime arg. a/b/store take a
  multi_ptr<T,addr_space,decorated::legacy> built from the raw pointer at the call (xmp_g_h global,
  xmp_l_h LDS).
- **const sycl::half* K/V will NOT convert to the non-const multi_ptr** - cast to (sycl::half*) at
  the load, else the compile fails.
- local_space (LDS) load/store works for staging Q16/P16/scores. In-place accumulate (mad with D=C)
  works for the K-loop.
- One 16-lane sub_group per tile; the 16-lane grid axis must be a multiple of 16.
- reduce_over_group(sg, val, op) (free fn) for the sub_group softmax reductions. A `static` helper
  called from the kernel lambda is fine (cf. dmmv.cpp), BUT any compile error inside it cascades into
  "SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute" at every call site -
  fix the root error, the cascade clears.

### FA mask layout (the correctness bug)
- For decode the mask dim-1 is the query-position dim, which is 1 (n_q==1): purely per-position,
  value = mask[pos] (pos stride = mask->nb[0]/2 = 1). The q-head does NOT index the mask. Measured:
  n_kv=256 -> mask ne=256x1, nb=2/512.
- Bug: indexing the mask with the q-head (0..23) as dim-1 reads OUT OF BOUNDS -> garbage scores -> NaN
  -> sampler assert after one token ("Assertion failed: found", llama-sampler.cpp). Symptom of an OOB
  mask read.

### Benchmark method: the -pg prefill trap (cost a 2x error this session)
- llama-bench `tg` (from -n) runs with n_prompt=0 -> decodes over a TINY context (~256 KV); attention
  is negligible there, so XMX vs tile is a wash (~22 both). It CANNOT measure 32K-context decode.
- To decode at a big context use -pg 32768,N (prefill 32K to build the cache, then generate N); it
  reports a COMBINED pp+tg t/s.
- **The prefill in a -pg run is SLOWER than a standalone -p run** (measured 819 vs 973 t/s; the -pg run
  has a larger n_ctx). So do NOT subtract a standalone pp time from a -pg time. Subtract WITHIN one
  run: `llama-bench -pg 32768,0 -pg 32768,N` (same session), then
  tg = N / ((N+32768)/combined - 32768/pp0). A cross-run prefill gave 9.7 t/s vs the true ~19 (2x low).

### Gains (verified 2026-09-02, -pg 32768,128 same-run subtraction)
- 32K-context decode: XMX 20.5 vs tile FA 18.9 t/s = **+8%** (18.9 confirms the prior 19.07 tile
  baseline; the ~1.6 t/s delta is ~50x the std dev).
- Small-context decode (the normal tg test): wash, ~22 t/s both (attention negligible).
- Prefill: identical 819.6 t/s both - the kernel is decode-only, never runs in prefill.
- Why the gain is ~8% overall (not bigger): the tile FA decode was COMPUTE-bound, not DRAM-bound (prior
  probes confirmed removing the QK^T/PV work pushes decode to the KV-read floor). XMX offloads that
  compute, so the 32K attention drops to its KV-read (DRAM) floor - a large win WITHIN the attention,
  but the attention is only a slice of the GEMV-dominated token, so whole-token is ~8% (18.9 -> 20.5).
- Prefill attention is the COMPUTE-bound case (big Q@K^T over many tokens) with far more XMX headroom.
  Extending the XMX path to prefill is the bigger potential win but a much larger change (full
  online-softmax tiling, large M tiles). Not done.

### Correctness
- Coherent output ("Paris") at 1-split (n_kv=256) and 2-split (n_kv=512) via llama-cli. n_splits =
  ceil(n_kv/256); the combine kernel is exercised at >=2 splits.

### MTP in llama-bench + the verify path = next lever (2026-09-03)
- MTP is now a first-class llama-bench mode (`--mtp`, committed `191548555`): `-d` sets the untimed
  established context, `-n` the timed generation, `--mtp-n-max` / `--mtp-p-min` the draft params. It
  runs the canonical common_speculative path (same as speculative-simple) on a single SYCL context.
  **Baseline: `tg128 @ d32768` = 41.12 t/s** (n_max=4, p_min=0.6) vs non-MTP 32K tg128 = 21.58 t/s
  -> ~1.9x. This matches/exceeds the cli MTP number (~39.9) and confirms the MTP path is correct.
- **Why the current XMX decode FA does NOT help MTP t/s:** the draft is a chain of single-token
  decodes (ne[1]==1 -> XMX decode FA runs), but the ONE main-model verify decodes the whole
  [id_last + draft] batch in a single graph pass (ne[1]=K+1, e.g. 5) -> the ne[1]==1 gate skips XMX,
  so verify attention runs on tile FA. The verify is ~86% of an MTP step, so it is the lever.
- **Next step: extend the XMX decode FA M-tile to ne[1]=2..8** (the verify batch size). The kernel
  already batches the gqa queries into one M=gqa tile; the extension batches the K+1 verify positions
  too (M = gqa*(K+1) or a separate dim). Covers verify attention (the compute-bound slice); the GEMVs
  already use the eSIMD M-kernel. See fattn-xmx-decode.cpp:191 (the ne[1]!=1 gate) + the support gate
  (Q ne[1]==1).

## Lessons / gotchas

- **Partial deploy after a reconfigure = 0xC0000409 crash misread as a kernel bug.** After
  `cmake-sycl-oneapi.bat` ALL ggml-*/llama-*.dll change together; deploying only the changed one leaves
  stale companions ABI-inconsistent -> the exe prints "Loading model..." then dies
  EXIT=-1073740791 (0xC0000409 fastfail). Deploy the whole bin (see sycl16-build-deploy).
- `git stash` of dmmv.cpp reverts to a committed state that does NOT compile (Q5_K 7-arg vs 8-arg
  template). The working baseline includes the uncommitted Q5_K fix + row-kernel block_load. Don't
  stash to get a "baseline".
- Build wrapper must set `LEVEL_ZERO_V1_SDK_PATH` so any mid-build cmake re-configure keeps the Level
  Zero include (done in build-sycl16-oneapi.bat).
- oneAPI 1.2.1 API: `sycl::reduce_over_group` (not reduce_work_group), `atomic_ref<T,order,scope>`
  (3 params), `malloc_host<T>(n, q)`.
- **SSH to the B70: never pipe llama-cli through `| tail`/`| grep` in the local shell.** llama-cli
  stays at the interactive REPL after `-n` tokens; the ssh session never closes, `tail` buffers until
  EOF -> looks like "no output" for the full timeout, and the REMOTE process keeps running after the
  local ssh dies (orphan holding GPU + CPU). Use the detached pattern: stage a .bat (scp it; inline
  `start /min cmd /c` quoting breaks on a -p "prompt with spaces"), then
  `start /min cmd /c "i1-test.bat > i1.log 2>&1"`, poll `type i1.log` from a 2nd ssh. Two concurrent
  27B loads can wedge the box (sshd starved; ping still answers) - never launch a second run while the
  first may still be alive.
