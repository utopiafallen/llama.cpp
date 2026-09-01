---
name: sycl-q6k-llmscaler-port
description: Port Intel llm-scaler's host-repacked Q6_K row GEMV into llama.cpp SYCL behind a COMPILE-TIME gate, A/B vs the SoA 2-row eSIMD DMMV on the B70. Use when working on the Q6_K row-kernel layout (llm-scaler repack vs SoA) or the GGML_SYCL_Q6K_GEMV_LLMSCALER path.
---

# SYCL Q6_K llm-scaler port (compile-time gate)

## FINAL OUTCOME (2026-08-27) - port complete, NO perf win, gate stays OFF
Faithful port implemented behind GGML_SYCL_Q6K_GEMV_LLMSCALER (+_FP16 variant), fully
validated, A/B'd on B70 (llama-bench -r 2 --device SYCL0, 2 rounds, Qwen3.8-27B-UD-Q6_K):

| test  | gate ON | gate OFF (SoA 2-row eSIMD) |
|-------|---------|---------------------------|
| tg128 | 22.44/22.45 | 22.87/22.87 |
| tg32  | 22.45/22.45 | 22.87/22.87 |
| pp256 | 558.20/558.25 | 750.64/750.29 |

Decode -1.8%, prefill M>8 -26%. Conclusion: on the B70 the existing SoA 2-row eSIMD
DMMV beats the llm-scaler row design; the repack's dequant savings do not overcome the
DRAM-bound floor. Code kept in tree (compile-time OFF, zero runtime cost); detail +
findings in PLAN.md in this directory (incl. decode run-to-run non-determinism which is
PRE-EXISTING upstream, and the ~1e-3 per-op fp16-d inherent ON/OFF delta).

Final code state (uncommitted on branch qwen3.8-27b-opt, base 00d351449):
- ggml-sycl.cpp: in-place repack reorder_qw_q6_k_llmscaler (210B/block, d stored fp16
  2/tile) + set_tensor load-time hook; GLU/MM+ADD fusions decline Q6_K under the gate.
- dmmv.cpp: M=1 row kernel (512-tile, 1 lane/row) + M-tile <M=1,2,4,8> launcher
  (dequantize_mul_mat_vec_q6_K_llmscaler_mt); DMMV/MMVQ/MMQ dispatch branches.
- getrows.cpp: get_rows_q6_K_llmscaler (scale stride was a 4x bug, fixed + host-tested).
- convert.cpp: dequantize_row_q6_K_llmscaler_reorder for the f16/f32 M>8 fallback
  (ggml_sycl_supports_mmq is hardcoded false upstream, so M>8 ALWAYS dequants to f16;
  without this kernel the ON build aborts on any batch > 8).
- cpy.cpp / set_rows.cpp: aborts on Q6_K reference-layout hits (unreachable in practice).
- q6k_hosttest.cpp/.bat: 2-phase host test, both bit-exact vs ggml ground truth.

---

Goal: test Intel llm-scaler's Q6_K row GEMV (host-repacked layout) vs the current SoA
2-row eSIMD DMMV on the B70. Full port behind a COMPILE-TIME gate
`GGML_SYCL_Q6K_GEMV_LLMSCALER` (NO runtime dual-path handling). A/B = two builds
(gate ON vs OFF), deploy, tg32 bench. User explicitly chose "full port, compile-time
gate" over a runtime probe.

Reference (local copy): /mnt/g/llm-scaler/sglang/custom-esimd-kernels/csrc/xpu/esimd_kernels/
  q6_k_GEMV.h  (q6_K)   and  q5_k_GEMV.h  (q5_K)

## The "optimal layout" (llm-scaler q6_K) -- a HOST-SIDE REPACK, not a reorder

Per output row (N rows total), row-major, zero extra memory vs GGUF:
  ql    [K/2]  uint8  4 low bits, nibble (byte j: low->elem 2j, high->elem 2j+1). SAME as SoA.
  qh    [K/4]  uint8  upper 2 bits, 2-bit packed AND PRE-SHUFFLED on host (below).
  scale [K/16] fp16   PRE-COMBINED = d * sc_int8 (per 16-elem group, may be negative).
  out   [N]    fp16

Dequant (SYMMETRIC, group=16, NO min):
  v6 = ql_nibble | (qh_2bit << 4)   (0..63)
  w  = scale * (v6 - 32)            (scale = d*sc_int8, pre-combined fp16)

Kernel: VL=512-elem tiles (=2x QK_K=256), 4 lanes/WG, each owns 1 output row
(row = group*4 + lane). The 4 lanes read 4 DIFFERENT rows (rows nblk*128 B apart).
This IS how the reference is written and it is fast -> NOT a coalescing problem.
The win is the repack (pre-shuffle + pre-combine), not the access pattern.

### qh PRE-SHUFFLE (the big win) -- exact mapping
Per VL=512 tile, shuffled byte t (t=0..127) holds 4 2-bit fields (p=0..3):
  field p of shuffled byte t  ->  element e = p*128 + t
i.e. shuffled_qh[t] = f0(elem t) | f1(elem 128+t)<<2 | f2(elem 256+t)<<4 | f3(elem 384+t)<<6
Repack (orig qh is 2-bit packed, 4-per-byte, orig byte b covers elems 4b..4b+3):
  for t in 0..127:
    for p in 0..3:
      e    = p*128 + t
      bits = (orig_qh[e/4] >> (2*(e%4))) & 3
      shuffled_qh[t] |= bits << (2*p)
GPU then reads the 128-B tile as 4 contiguous 2-bit planes (STRIDE-1 add, not stride-4):
  qh_data = block_load<uint8,128>; for p in 0..3:
    ext = (qh_data >> (2*p)) & 3;   weight_f[p*128 .. p*128+127] += ext * 16

### scale PRE-COMBINE (the other win)
  for each 16-elem group g in a 256-elem block:  scale[g] = (fp16)( (float)d * (float)sc[g] )
GPU does NO per-block sc/d combine (current SoA stores int8 sc + half d separately and
combines on GPU).

## Current llama.cpp Q6_K SoA (baseline, gate OFF) -- DO NOT BREAK

Layout: [ql: nb*(QK_K/2)] [qh: nb*(QK_K/4)] [scales int8: nb*(QK_K/16)] [d: nb*half]
  nb = nrows*nblk, nblk = K/QK_K, ordered by (row, block). ql identical to llm-scaler;
  qh NOT shuffled; scale = int8 sc + half d SEPARATE.
Kernel: esimd_reorder_q_traits<GGML_TYPE_Q6_K>::mac_pair, esimd.hpp:404-498 (2-row eSIMD).
  dequant w = sc*d*(v6-32), strided 2-bit extract (case 0-3 switch on qh masks 03/0C/30/C0).
reorder: ggml-sycl.cpp:4343 reorder_qw_q6_k (dispatch from reorder_qw:4432).

## Plan (TWO compile-time gates, 3-way A/B) -- user chose "implement both"

Gates (CMake options, default OFF, defined on ggml-sycl target):
  GGML_SYCL_Q6K_GEMV_LLMSCALER      : row kernel + qh pre-shuffle. IN-PLACE (same SoA size).
  GGML_SYCL_Q6K_GEMV_LLMSCALER_FP16 : additionally fp16 pre-combined scale. LARGER buffer
                                      (needs separate repack buffer, ptr in extra->repacked_data).
  (FP16 requires the base gate.)

3 builds for the A/B (isolate the fp16-scale uplift):
  A (off)                : current SoA + eSIMD 2-row DMMV  -> baseline ~22.7 t/s
  B (base gate only)     : row kernel + qh pre-shuffle, in-place (int8 sc + half d on GPU)
  C (base + _FP16)       : row kernel + qh pre-shuffle + fp16 pre-combined scale
  uplift(qh) = B - A ; uplift(fp16 scale) = C - B

Layouts:
  B (in-place, same size as SoA): [ql nb*(QK_K/2)] [qh nb*(QK_K/4) PRE-SHUFFLED]
                                   [scales int8 nb*(QK_K/16)] [d half nb]
  C (larger, separate buffer):   [ql nb*(QK_K/2)] [qh nb*(QK_K/4) PRE-SHUFFLED]
                                   [scale fp16 nb*(QK_K/16)]   (no separate d)
Steps:
 1. CMake: 2 options + target_compile_definitions.
 2. Repack: qh pre-shuffle per 512-tile (B,C). Under _FP16 also fp16 scale + separate
    buffer stored in extra->repacked_data (C only).
 3. Row kernel: 512-tile, stride-1 high-bit, fp32 act. #ifdef _FP16 to read fp16 scale
    vs int8-sc+half-d. Reads extra->repacked_data when _FP16, else src0->data.
 4. GLU: decline Q6_K GLU when base gate on (ffn_gate/up fall to row DENSE kernel).
    eSIMD path intact when base gate off.
 5. Add `void* repacked_data=nullptr` to ggml_tensor_extra_gpu (gated _FP16); free in
    release_extra_gpu.

## Dispatch map (where the gate goes)

- reorder_qw_q6_k:        ggml-sycl.cpp:4343 (from reorder_qw:4432; MoE:4156 reorder_qw_q6_k_moe)
- DENSE dispatch:         dmmv.cpp:2265 dequantize_mul_mat_vec_q6_K_sycl_reorder
                          (row:2417 if g_ggml_sycl_q6k_gemv_row, else esimd:2419)
- existing row kernel:    dmmv.cpp:2138 dequantize_mul_mat_vec_q6_K_sycl_reorder_row (SoA, pre-port)
- eSIMD 2-row:            dmmv.cpp:1983 -> dequantize_mul_mat_vec_reorder_esimd<Q6_K,ADD> (esimd.hpp)
- add path:               dmmv.cpp:2109 ggml_sycl_q6_k_dmmv_reorder_esimd_add; ggml-sycl.cpp:4751
- GLU fused:              ggml-sycl.cpp:4619 ggml_sycl_mul_mat_glu_mmvq_fused
                          (Q6_K/Q5_K eSIMD 4637-4659) -> dmmv.cpp:2123 ..._esimd_glu<Q6_K>
- env (runtime, existing):g_ggml_sycl_q6k_gemv_row (GGML_SYCL_Q6K_GEMV_ROW) ggml-sycl.cpp:103,327

## Build / deploy / A-B (see sycl16-build-deploy + b70-decode-perf)

- build OFF (baseline): no define.  build ON: -DGGML_SYCL_Q6K_GEMV_LLMSCALER=ON.
- cmd.exe /C "G:\llama-cpp-src\build-sycl16-oneapi.bat"; deploy WHOLE bin (else 0xC0000409).
- bench: llama-bench -p 256 -n 32 (tg32); env GGML_SYCL_ENABLE_ESIMD=1 GGML_SYCL_FUSE_MM_ADD=1
  GGML_SYCL_FUSE_MM_GLU=1 GGML_SYCL_FUSE_GDN_DT=1. Baseline ~22.7 t/s.
- correctness: "Paris" llama-cli on the gated build.

## Caveats / expectations

- eSIMD 2-row already reads the same bytes with shared 128B fetches; the row kernel's edge
  would come ONLY from pre-shuffle/pre-combine dequant savings, which may be small vs the
  ~490 GB/s DRAM-bound floor. May land within noise. (Why we leaned probe-first, but user
  chose full port.)
- repack is host-side one-time (amortized). llama.cpp act y is fp32 (llm-scaler uses fp16
  input) -> adapt the kernel's activation load to fp32.
- MoE Q6_K (reorder_qw_q6_k_moe) is NOT in this dense model; ignore for now.

## llm-scaler q5_K (if needed, q5_k_GEMV.h)

  ql [K/2] uint8; qh [K/8] uint8 1-bit (5th bit) PRE-SHUFFLED (per 512: bit b of byte t ->
  elem b*64+t, stride-1 not stride-8); scale [K/32] fp16 = d*sc6; min [K/32] fp16 = dmin*mn6;
  v5 = ql_nibble | (qh_bit<<4) (0..31); w = scale*v5 - min. (q5_K is ASYMMETRIC: has min.)
