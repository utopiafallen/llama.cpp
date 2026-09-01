---
name: sycl-fusion-debug
description: Diagnose why a SYCL backend graph fusion silently declines (or add a new one). Use when a fused op never fires, when adding a fusion pattern, or when a pattern gate rejects valid candidates.
---

# Debug SYCL graph fusion declines

Symptom: a fusion env flag is on, the startup log shows the flag, but the fused op never
activates (no activation log, no t/s delta, t/s flat). The decline is intentional - some
gate checked false. This skill finds which gate, without guessing.

## How the dispatch works

`ggml_backend_sycl_graph_compute_impl` (ggml/src/ggml-sycl/ggml-sycl.cpp, loop starts ~5793)
iterates `cgraph->nodes` and, per node, tries in order:

1. `ggml_sycl_fuse` (generic, topk MoE only)
2. GDN fused-cache cpy
3. `ggml_sycl_can_fuse` pattern checks: `{RMS_NORM, MUL}` -> rms_norm_fused,
   `{UNARY, MUL}` -> unary_mul_fused, `{ADD, UNARY(softplus), MUL}` -> add_softplus_mul_fused
4. `{MUL_MAT, ADD}` -> `ggml_sycl_mul_mat_add_fused`, `{MUL_MAT, MUL_MAT, GLU}` ->
   `ggml_sycl_mul_mat_glu_mmvq_fused`

A matched pattern launches the fused kernel and skips the consumed nodes (`i += n; continue`).
On any decline the nodes run unfused - so a silently-declined fusion is always bit-correct,
just not faster.

## The two traps that cause silent declines

1. `ggml_sycl_can_fuse` (ggml/src/ggml-sycl/fusion.cpp) returns true ONLY for patterns with an
   explicit case in its if-chain. After the generic chain check it falls through to an
   unconditional `return false`. Adding a new pattern (e.g. `{MUL_MAT, ADD}`) requires adding
   its case there too - the entry function can be fully correct and still never run.
2. The inference loader NEVER sets `GGML_TENSOR_FLAG_PARAM` (training-only flag, zero uses in
   src/ and common/). Gates that identify "which operand is the model constant" via the PARAM
   flag always reject. In inference graphs the constant is a graph leaf: identify operands by
   `GGML_TENSOR_FLAG_COMPUTE` (the computed chain input has it, GGUF tensors and their views do
   not). Views also carry no COMPUTE flag, so this works through view-based loading.

## Chain-form fusability (what `ggml_can_fuse` requires)

`ggml_can_fuse_ext` (ggml/src/ggml-impl.h:669), applied to sequential node indices:

- every node: op matches, `GGML_TENSOR_FLAG_COMPUTE` set
- every node except the last: `ggml_node_get_use_count() == 1`, no `view_src`, no OUTPUT flag
- each link: node's `src[0]` or `src[1]` is the previous node, and `ggml_are_same_shape`
- use counts come from the `cgraph->use_counts` table filled at graph build
  (`ggml_build_forward_expand`); a tensor not visited by the build reads as 0 uses

Tensor-level gates (weight type, mat-vec `src1->ne[1] == 1`, contiguity, split buffers,
DMMV reorder installed, per-fusion env flags) live in the entry functions in ggml-sycl.cpp,
AFTER the can_fuse call.

## Diagnosis workflow (one-shot rejection dump)

1. In the entry/dispatch site, filter true candidates first (e.g. for `{MUL_MAT, ADD}` only run
   the diag when `nodes[i+1]->op == GGML_OP_ADD`), so non-candidate nodes don't consume the
   diag budget.
2. On rejection of a candidate, dump EVERY gate value in one line, first N=4 candidates per
   process (`static int n_diag` guard, `GGML_LOG_INFO`):
   - chain: use count, view_src, OUTPUT flag, COMPUTE flag (for each node), src-link, same-shape
   - tensor: types of weight/act/dst/residual, `ne[1]`, `ggml_is_contiguous`, nelements,
     split-buffer, reorder state, relevant global flags
3. Have the user rebuild and run a short bench (e.g. `llama-bench -m ... -b 2048 -p 512 -n 8
   --device SYCL0` with the env flags on) and paste the reject lines. The values localize the
   failing clause directly.
4. If all dumped values pass, the failure is outside the dumped clauses - suspect the
   `ggml_sycl_can_fuse` case fall-through (trap 1) or a master flag (e.g.
   `g_ggml_sycl_enable_fusion`, check another pattern in the same run fires to prove it is on).

Keep the diag lines one-shot: per-op dispatch lines only under `GGML_SYCL_DEBUG=1`.

Activation check while debugging: have each fused entry log once on first fire
(`...: active, first on '<tensor name>'`) - the tensor name also confirms WHICH op type fired.
The activation log and the whole reject-dump scaffold are session-only: strip them out
(before the final commit) and keep only the `scope_op_debug_print` line under
`GGML_SYCL_DEBUG=1`, which is the codebase's normal dispatch pattern.
(e.g. ffn_down vs attn out proj).

## Cross-checking ggml graph behavior locally

If the doubt is about `ggml_can_fuse` itself rather than a gate, do not burn a full build:
build a tiny C/C++ repro against the ggml core and run it, see the `wsl-windows-tinybuild`
skill (this is exactly how a false "can_fuse" decline was settled: 2-node `{MUL_MAT, ADD}`
repro returned 1, proving the wrapper case was missing).

## Worked example (Qwen3-Next GDN/FA, 64-layer hybrid)

- GDN layer tail: ssm_out GEMV -> `reshape_2d` -> residual add. The reshape breaks
  `{MUL_MAT, ADD}` for GDN layers; additionally `ssm_out` is Q8_0, not Q6_K.
- FA layer tail: gate `MUL` sits BEFORE the wo GEMV, so `{MUL_MAT, ADD}` holds for FA.
- Dense FFN tail: down GEMV is the last FFN node (no trailing view), `{MUL_MAT, ADD}` holds.
- GDN dt chain (48 v-heads, per layer): `cont_3d(alpha)` -> ADD (+ssm_dt.bias [48]) ->
  softplus -> MUL (* ssm_a [48]) -> reshape_4d. Matches `{ADD, UNARY, MUL}` with x [48,1,1],
  dt/a 1-D leaves.
- `build_lora_mm(weight, x, scale)`: a non-null scale inserts an extra `MUL` after the GEMV -
  check the GGUF for scale tensors (names ending `_s`) before assuming GEMV is directly linked
  to its consumer (`gguf-inspect` skill).

## Performance expectations on Arc B-series (B770 measured, Qwen3.5/3.8-Next 27B Q6_K)

Decisive A/B (tg8, baseline 22.09 t/s, PPL unchanged 5.398):
- {MUL_MAT,ADD} epilogue (80 adds/decode): +0.4%
- {MUL_MAT,MUL_MAT,GLU} dense-FFN epilogue (128/decode): +0.5%
- {ADD,softplus,MUL} GDN dt chain (48 x 3 tiny ops/decode): +0.4%
- all three (~352 nodes/decode): +0.9% (22.09 -> 22.28)

Rule of thumb: ~1 us per removed decode node, on the in-order queue; launch-count and
byte-cut savings are comparable. A ~50-130-node fusion is worth ~0.4-0.5%, so budget
loader-level merges (e.g. merging two GEMV weights into one) at that same scale before
doing them.

Caution: a flat result for an env-gated fusion is NOT evidence launch savings are free -
first prove the pattern actually fires (activation log / rejection diag) before drawing
perf conclusions. (This exact misdiagnosis burned one round-trip.)
- Profile op-times from these runs are distorted by the per-command floor; use them for shape,
  not shares.
