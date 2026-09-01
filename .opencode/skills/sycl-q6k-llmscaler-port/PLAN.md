# Q6_K llm-scaler comprehensive fix - WORKING PLAN

Status: COMPLETE (2026-08-27). All steps done; final A/B shows -1.8% decode vs the SoA
baseline -> port validated but retired (gate stays OFF). See "FINAL RESULT" below.
This file is the single source of truth across context compactions.

## Goal
Gate-ON build (`GGML_SYCL_Q6K_GEMV_LLMSCALER`) produces garbled text. Fix ALL packed-layout
bugs in one go: ONE layout everywhere under the gate, established ONCE at load time, all ops
read it. Use the TRUE llm-scaler layout + kernels (full perf unlock), A/B vs SoA eSIMD baseline.
Gate-OFF stays 100% upstream (untouched).

## Key decisions (user-approved)
- Target layout = llm-scaler reference layout (512-tile qh shuffle), NOT the SoA layout.
- Repack ONCE at load time, IN-PLACE (same 210B/block size, memory unchanged), NO fp16
  scale precompute (base gate: int8 sc + half d, combined on GPU).
- Port the REFERENCE kernels (q6_k_GEMV.h): M=1 for decode, M=2/4/8 tiled for multi-token
  (their kernels already read the repacked layout correctly -> no adaptation bug for MMVQ).
  Arbitrary M (incl. >8, MMQ range) handled by their chunk-8/4/2/1 launcher pattern.
- Everything compile-time gated; gate OFF = upstream behavior.

## CRITICAL FINDING (why the old port was wrong)
The previously ported kernel (dmmv.cpp dequantize_mul_mat_vec_q6_K_llmscaler) is NOT the
reference kernel. It uses a per-256-block self-contained qh convention (byte=e%64,
field=2*(e/64)); the reference uses a per-512-tile CROSS-BLOCK shuffle with one 128B qh
load + 4 stride-1 planes. The old KDBG check only verified ported-kernel vs ported-repack
self-consistency, so it passed. Re-port faithfully.

Reference (q6_k_GEMV.h, verified line-by-line):
  - VL=512 tiles (= 2 x QK_K=256 blocks). Requires K % 512 == 0.
  - ql: PAIR-PACKED, 256B per tile. tile byte B holds element 2B (low nibble) and
    2B+1 (high nibble). NOT a raw copy.
  - qh: 128B per tile. shuffled byte t2 (0..127), field p (0..3, bits 2p..2p+1) ->
    element e = p*128 + t2 (within tile). One block_load<uint8,128>; then
    ext = (qh_data >> (2*p)) & 3 -> elements p*128..p*128+127.
  - scale: 32 per tile (per-16 groups). Base gate: int8 sc[16]/block + half d/block,
    w = d*sc*(v6-32). (FP16 variant would pre-combine - SKIPPED, memory.)
  - Kernels: Q6_K_gemv_kernel (M=1) and Q6_K_gemv_M_kernel<M> (M=2,4,8) + host fallback.
    grid: ceil(N/4) groups x 4 lanes; row = group*4 + lane; 1 output row per lane.
    M kernel: weight tile loaded+dequantized ONCE, reused across all M act rows
    (M accumulators). Act for row m at input + m*K + k (contiguous in k).
    Launcher q6_k_gemv_M_host: tiles M in chunks 8/4/2/1 (arbitrary M).
  - Reference uses fp16 act + fp16 out; OUR ADAPTATION: fp32 act (llama.cpp src1 is f32
    on device; column-major [K,M] so act[m*K + k] is contiguous in k - matches as-is),
    fp32 dst write (dst[n + m*N] - matches as-is), int8 sc + half d.
  - Reduce: use reduce<float>(v, std::plus<>()) (codebase style), not esimd_detail::sum.

## Layout spec (base gate, in-place, == SoA size)
Flat (row, block) order as in SoA:
  [ql: nb*(QK_K/2) PAIR-PACKED][qh: nb*(QK_K/4) REFERENCE-SHUFFLED per 512-tile][scales int8: nb*16 raw][d: nb*half raw]
  nb = nrows * (K/256); K%512==0 required so 512-tile = 2 consecutive flat blocks.
Per 512-tile (blocks ib0=2t, ib1=2t+1), element e in 0..511, ob=e/256, oe=e%256:
  ql: tile byte B=e/2; val = source blk[ob].ql[64*(oe/128) + (oe%32) + 32*((oe%128)/32 & 1)],
      low nibble if (oe%128) < 64 else high. (pair packing of the source-mapped values)
  qh: tile byte t2, field p, e = p*128 + t2; bits = (blk[ob].qh[32*(oe/128) + (oe%32)]
      >> (2*((oe%128)/32))) & 3.
  scales: blk0.scales[16] then blk1.scales[16] (raw); d: blk0.d, blk1.d (raw halves).
Reader per tile element E: ql byte E/2 (low iff E even); qh byte E%128 field E/128;
  scale sb=E/16 -> sc[sb]*d[sb/16]; v6 = ql | qh<<4; w = (v6-32)*sc*d.

## Dispatch map (ggml_sycl_mul_mat, ggml-sycl.cpp ~4633)
  ne11 == 1  -> DMMV  (can_use_dequantize_mul_mat_vec, 4618)
  2..8       -> MMVQ  (can_use_mul_mat_vec_q, 4628; MMVQ_MAX_BATCH_SIZE=8, common.hpp:211)
  >8         -> MMQ   (ggml_sycl_supports_mmq; Q6_K kernel EXISTS at mmq.cpp:1766, reads AoS, NO reorder variant)
  else       -> generic ggml_sycl_op_mul_mat_sycl
CHANGES under gate (src0 Q6_K + reorder flag set; DONE):
  - DMMV: reference M=1 kernel. view_src extra fallback (tied lm_head has no extra).
  - MMVQ (2..8): wired into the Q6_K case of ggml_sycl_op_mul_mat_vec_q -> M-tiled
    launcher (reads fp32 act from src1_ddf_i directly, q8_1 buffer unused).
  - MMQ range (>8): wired into the Q6_K case of ggml_sycl_op_mul_mat_q -> same launcher,
    dst_stride = nrows_dst. Correct for all M; SLOW for large M (re-reads weights per
    8-row chunk) - documented limitation.
  - get_rows (token_embd is Q6_K, 5120x248320): new reference-layout kernel
    (getrows.cpp, scalar per-element mapping, contiguous src0 asserted).
  - convert/cpy/set_rows: hard abort under gate if a repacked Q6_K hits them
    (unreachable in this graph; fail loudly, never silent garbage).
  - Dispatch resolves extra via src0->extra, falling back to src0->view_src->extra.

## Change list
1. ggml-sycl.cpp
   - reorder_qw_q6_k_llmscaler (4416): rewrite repack kernel = reference 512-tile shuffle
     above (base in-place branch; keep FP16 branch structure for later but it stays OFF).
   - LOAD-TIME REPACK HOOK in ggml_backend_sycl_buffer_set_tensor (648): after the upload
     memcpy, if gate && tensor->type == Q6_K && ne[2]==1 && ne[3]==1 && ne[0] % 512 == 0
     && size % sizeof(block_q6_K) == 0: run repack on tensor->data+offset with `size`,
     set extra->optimized_feature.reorder = true. Skip MoE (ne[2]>1) and non-512-aligned
     tensors (they stay AoS, flag off, upstream paths - safe fallback, no garbage).
   - REMOVE the lazy mid-compute repack for Q6_K under the gate: opt_for_reorder (4610)
     gate early-return currently repacks during first 1-token graph compute - replace with
     no-op for Q6_K (flag already set at load). Delete `should_reorder_tensor`'s ne[1]<=8
     relevance for Q6_K under gate (keep function for gate-OFF path).
   - Fusions under gate: GLU already declines Q6_K (fusion.cpp:39). CHECK + DECLINE MM+ADD
     fusion for Q6_K too (ggml_sycl_mul_mat_add_fused -> dispatches SoA-reading
     ggml_sycl_q6_k_dmmv_reorder_esimd_add, dmmv.cpp:2004).
2. dmmv.cpp
   - Replace dequantize_mul_mat_vec_q6_K_llmscaler (2228) with faithful reference M=1 port
     (fp32 act, int8 sc + half d, fp32 dst, 4-lane WGs, VL=512).
   - Add M=2/4/8 ports + launcher (new op function, e.g. ggml_sycl_op_q6_k_llmscaler_gemv_m)
     usable for both the MMVQ (2..8) and MMQ-range (>8) branches.
   - Q6_K case (2496): under gate, reorder branch -> M=1 kernel (unchanged shape);
     #else SoA/AoS branches stay for gate-OFF.
   - Strip ALL debug blocks (Q6K-DISP/XCHK/OPMAT/OPPOST/canary) AFTER verification passes.
3. mmq.cpp: Q6_K case wired to the M-tiled launcher under gate (see dispatch map above).
4. convert.cpp (both to_fp16/to_fp32 Q6_K cases) / cpy.cpp (Q6_K->Q6_K) / set_rows.cpp
   (Q6_K dst): hard abort under gate - unreachable in this graph (step 1), fail loudly.
5. MoE Q6_K (reorder_qw_q6_k_moe, moe fuse): untouched - 4D tensors are NOT repacked at
   load (set_tensor hook requires ne[2]==1), opt_for_reorder_id skips Q6_K under gate,
   so MoE stays upstream AoS. Qwen3.8 is dense anyway.

## Current debug instrumentation (strip at step 6)
- dmmv.cpp: [Q6K-XCHK] (act 4 + dst 8, cap 2, ffn_up-0) - SHARED code, fires in BOTH builds.
- ggml-sycl.cpp: [Q6K-OPMAT] (split info at startup), [Q6K-OPPOST] (post-op dst, cap 40),
  [Q6K-REPACK] (load-time, gate-ON only; first 6 + any token_embd).
- getrows.cpp: [Q6K-EMB] (post-switch, keys on src0 name token_embd, cap 2) - NEVER fired:
  token_embd get_rows runs on CPU (host buffer), not the SYCL backend.

## Session: run56-60 isolation results (DONE)
Model is an SSM HYBRID (not Llama attention): per block attn_qkv+attn_gate+ssm_alpha/beta+
ssm_out are Q4_K/Q5_K/Q6_K/Q8_0 (dynamic per block, unsloth UD), ffn_gate IQ4_XS/Q5_K/Q6_K,
ffn_up Q5_K/Q6_K/Q8_0, ffn_down Q5_K/Q6_K/IQ4_XS/Q8_0. blk.0: attn/SSM all Q8_0, ffn_up Q6_K.

1. token_embd (Q6_K) is NOT repacked: no [Q6K-REPACK] line, and the SYCL get_rows probe
   (post-switch, keys on src0 name) NEVER fires -> token_embd sits on the HOST buffer, its
   get_rows runs on CPU (upstream AoS, correct). Consequence: the get_rows llmscaler kernel
   is dead code in this config (single-GPU SYCL0, server places the 1GB embd on host), but
   it was fixed + verified anyway (see 2). lm_head = tied view of token_embd -> also host ->
   its DMMV falls through (extra NULL) to the upstream AoS kernel reading host data.
2. get_rows llmscaler BUG FOUND + FIXED: sc section stride was QK_K/2 (128B/tile), correct is
   QK_K/8 (32B/tile int8; 32 halves/tile in FP16); d_b base followed sc_sz (was 218MB OOB
   past the buffer). Fixed in getrows.cpp (dequant_elem + sc_sz, variant-aware). Host-verified
   bit-exact: q6k_hosttest.cpp phase 2 (multi-row buffer, 7 rows x 13 tiles x 512, maxdiff=0).
   (The host test itself had a d_sz missing sizeof(float) that heap-corrupted its own ground
   truth - fixed; that was the false 766-mismatch scare, NOT the kernel.)
3. M=1 DMMV + M-tile MMVQ verified at runtime: warmup ffn_up-0 (M=1) and prefill ffn_up-0
   (batch 8 -> M-tile) outputs match gate-OFF to ~1e-3 in EVERY run. M-tile scale/d indexing
   re-verified against the repack by inspection (sc stride ntile*32, d_i[2*it], sb<16?d0:d1).
4. The ~1e-3 per-op ON/OFF delta is INHERENT to the reference layout: block scale d is stored
   fp16 (2B) to keep the in-place 210B/block size; upstream reads f32 d. ~5e-4 relative weight
   error -> ~1e-3 abs op-output delta. Not a bug; the llm-scaler layout has the same property.
5. DECODE is non-deterministic run-to-run, PRE-EXISTING (upstream, gate-OFF too): within one
   build, warmup+prefill probe values are bit-identical across runs, but decode XCHK/OPPOST
   values differ run-to-run (ON: run56 vs run59; OFF: run57 vs run60 - different both times).
   Unmodified SSM state/decode path is the only prefill-vs-decode delta -> upstream issue
   (suspect: SSM state init/persistence or non-deterministic decode SSM kernel). OUT OF SCOPE.
   Consequence: the earlier "act diverges 0.235 between ON and OFF" was run-to-run noise, NOT
   an ON/OFF difference. Verification criterion (adjusted): compare the DETERMINISTIC ops
   (warmup + prefill ffn_up-0) between ON/OFF at ~1e-3 (fp16-d) tolerance; decode is
   non-deterministic in BOTH builds.
6. Text is coherent in both builds (random seed -1, so per-run answers differ by design).
   Single-request t/s with probes: ON 22.19, OFF 22.59-22.61 (8 tokens, not a benchmark).

## Verification gate (adjusted by run56-60 findings)
1. [x] Host unit test phase 1: repack -> scalar dequant == ggml ground truth, bit-exact.
2. [x] Host unit test phase 2: get_rows multi-row offsets (post-fix), bit-exact.
3. [x] K % 512: all 241 Q6_K tensors pass.
4. [x] Deterministic-op parity ON vs OFF: warmup + prefill ffn_up-0 within ~1e-3 (fp16-d),
   stable across runs (runs 56-60). Decode excluded (non-deterministic in both builds).
5. [x] Text coherent in both builds; no convert/cpy/set_rows abort fired.
6. [x] token_embd placement resolved (host buffer; SYCL get_rows/DMMV never touch it).

## Benchmark (after verification)
- Strip debug, rebuild ON + OFF, deploy whole bin.
- llama-bench -p 256 -n 32 (tg32) + tg128, env: GGML_SYCL_ENABLE_ESIMD=1
  GGML_SYCL_FUSE_MM_ADD=1 GGML_SYCL_FUSE_MM_GLU=1 GGML_SYCL_FUSE_GDN_DT=1 (runtest.bat).
- Baseline (gate OFF, SoA eSIMD) ~22.7 t/s tg32. Expectations per skill: DRAM-bound
  ~490 GB/s floor; win may be small. Report uplift.

## Step 1 results (DONE)
- K%512: ALL 241 Q6_K tensors pass (K in {5120, 6144, 10240, 17408}, all %512==0). No fallback case.
- Q6_K tensors (per block, 64 blocks, mixed quants): attn_gate(Q6_K 26), attn_k(3),
  attn_output(12), attn_q(9), attn_qkv(35), ffn_down(45), ffn_gate(41), ffn_up(38), ssm_out(30),
  nextn.eh_proj(1, unused MTP), token_embd(1).
- **token_embd.weight is Q6_K** (5120 x 248320) -> GET_ROWS on Q6_K IS REACHABLE (every prompt
  embedding lookup). getrows.cpp:355 uses dequantize_q6_K (AoS, dequantize.hpp:381). MUST PORT.
  No output.head tensor in GGUF -> lm_head is TIED to token_embd (likely transposed view).
  RISK: views have no extra of their own (init_tensor early-returns for view_src, ggml-sycl.cpp:604)
  -> dispatch must resolve extra via src0->view_src->extra fallback, else the lm_head DMMV
  takes the AoS path on repacked data = garbage. Verify in step 5 (log the output DMMV path).
- MM+ADD fusion: ALREADY declines Q6_K under the gate (ggml-sycl.cpp:4852-4855). No change.
- Convert (convert.cpp:684-693 Q6_K branch) / cpy (cpy.cpp:1049): assume unreachable in this
  graph; under gate use the shared scalar dequant helper (convert) or hard assert (cpy).
- set_tensor (648) runs on the device DEFAULT queue; compute uses ctx.stream() (different queue).
  -> repack in set_tensor must end with a queue wait (load is serial; safe).
- tensor->extra is allocated in buffer_init_tensor (619) BEFORE set_tensor for Q6_K
  (g_ggml_sycl_enable_optimize). Views have no extra.

## Execution order
- [x] 0. Diagnose garbled text (runs 44-54): root causes = (a) MMVQ prompt chunk read
      repacked weights wrong, (b) ported kernel not reference-faithful, (c) out-of-band
      overwrite artifact gate-ON-specific (stable in gate-OFF: run54 XCHK==OPPOST).
- [x] 1. Verify: K%512 per Q6_K tensor; graph op reachability (DEQUANTIZE/GET_ROWS/CPY on
      Q6_K); MM+ADD fusion condition for Q6_K. (no build)
- [x] 2. Repack (reference shuffle) + set_tensor load-time hook + flag-at-load + remove lazy
      repack for Q6_K under gate.
- [x] 3. Faithful M=1 kernel port (replaces current llmscaler kernel) + host unit test.
- [x] 4. M=2/4/8 ports + launcher + wire MMVQ (2..8) and MMQ-range (>8) dispatch under gate
      + decline MM+ADD fusion for Q6_K.
- [x] 5. Build ON; verify (adjusted criteria, runs 56-60): host tests bit-exact, deterministic
      ops within fp16-d tolerance of gate-OFF, text coherent, decode non-determinism confirmed
      pre-existing (both builds).
- [x] 6. Strip debug instrumentation ([Q6K-XCHK], [Q6K-OPMAT], [Q6K-OPPOST], [Q6K-REPACK],
      [Q6K-EMB]); rebuild ON+OFF; deploy.
- [x] 7. tg32/tg128 A/B benchmark (llama-bench -r 2 --device SYCL0, 2 rounds each,
      interleaved order): ON 22.44-22.45 vs OFF 22.87 - 1.8% slower. NO WIN. See below.

## FINAL RESULT (2026-08-27): no perf win, port retired
llama-bench -r 2 --device SYCL0, Qwen3.8-27B-UD-Q6_K.gguf, 32GB B70:

| test  | ON round1 | ON round2 | OFF round1 | OFF round2 |
|-------|-----------|-----------|------------|------------|
| tg128 | 22.44     | 22.45     | 22.87      | 22.87      |
| tg32  | 22.45     | 22.45     | 22.87      | 22.87      |
| pp8   | 65.74     | 65.82     | 63.48      | 63.35      |
| pp256 | 558.20    | 558.25    | 750.64     | 750.29     |

- Decode: the reference-layout row kernel is 1.8% (0.42 t/s) SLOWER than the SoA 2-row
  eSIMD DMMV. Both builds rock-stable across rounds/order; the delta is real.
- Prefill M>8: -26%. ggml_sycl_supports_mmq() is hardcoded false (upstream "accuracy
  issues in MMQ"), so M>8 batches dequant weights to f16 and run f16 GEMM. The llmscaler
  scalar dequant (4 elem/item, no SIMD) loses to the vectorized SoA
  dequantize_block_q6_K_reorder there.
- Confirms the earlier skill finding (row-per-lane sglang port = 20.39 vs 22.33 SoA):
  on B70 the 2-row SoA eSIMD DMMV is the better design. The faithful llm-scaler port
  reproduces it, but better kernels do not come from that layout.
- Correctness of the port was fully validated (host tests bit-exact, deterministic ops
  within fp16-d ~1e-3 of OFF, coherent text) - the loss is a design/latency-bandwidth
  tradeoff, not a bug.
- M>8 f16-fallback path: added dequantize_row_q6_K_llmscaler_reorder in convert.cpp
  (returned by ggml_get_to_fp16_sycl / ggml_get_to_fp32_sycl under the gate); without it
  the gate-ON build aborts on any batch > 8 (server with default n_batch included).
- Decision: leave the gate code in tree (compile-time OFF, zero cost when OFF); do not
  enable it. If a future B70-class GPU shows a different bandwidth/latency balance, the
  gate can be re-benchmarked without re-porting.

## Environment / commands (see SKILL.md for details)
- Build: cmd.exe /C "G:\llama-cpp-src\build-sycl16-gate.bat ON|OFF OFF"
  (args: LLMSCALER FP16; FP16 stays OFF).
- Deploy: kill the user's server on SYCL1 FIRST (by PID, never by image name), then
  scp build-x64-windows-sycl-release-f16/bin/ggml-sycl.dll
  b70:"C:/Users/LocalAdmin/Desktop/llama-cpp-sycl16/ggml-sycl.dll".
- Run: setsid ssh b70 'cd /d "C:/Users/LocalAdmin/Desktop/llama-cpp-sycl16" && runtest.bat > runNN.log 2>&1'
  >/dev/null 2>&1 </dev/null & disown; sleep 45; then
  ssh b70 "curl -s http://127.0.0.1:18081/completion -d @C:/Users/LocalAdmin/Desktop/llama-cpp-sycl16/req51.json"
  (req51.json = {"prompt":"The largest planet in the solar system is","max_tokens":8};
  ALWAYS use -d @file - Windows cmd does not strip single quotes; inline JSON breaks).
- runtest.bat: --device SYCL0 (pin to the free B70; the user keeps SYCL1), port 18081.
  SYCL_DEVICE_FILTER is IGNORED by the 2026.1 runtime; ZE_AFFINITY_MASK crashes the
  level_zero driver. --device is the only working pin.
- Box: 3 SYCL devices (SYCL0=B70 32GB free, SYCL1=B70 [user server], SYCL2=iGPU).
  --list-devices shows per-device free VRAM. 24GB Q6_K model fits one GPU, so with
  --device SYCL0 the weights are NOT split: ffn_up-0 op split=0, ctx_dev=0.
- Host unit test (dev box): cmd.exe /C "G:\llama-cpp-src\q6k_hosttest.bat" - round-trips
  repack vs ggml ground truth over 2000 tiles; PASSES bit-exact (maxdiff=0).

## Open items / risks
- simd<float,512> / block_load<uint8,128>: reference uses them on same arch (bmg_g21) -
  should compile; if not, split VL=512 into 2x256 (keeps semantics).
- MMQ-range Q6_K under gate is slow (chunk-8 GEMV loop) - documented, follow-up: port MMQ
  Q6_K qh handling to 512-tile.
- 3-device mode: repack runs per tensor on the tensor's own device (set_tensor has
  ctx->device) - no cross-device work needed since layout is per-device-buffer.
- If step 5 shows DISP != OPPOST after all of the above: it is a genuine concurrency bug,
  isolate with a full-buffer canary right after the kernel (whole 17408-float marker) +
  per-stream logging; do NOT benchmark until resolved.
