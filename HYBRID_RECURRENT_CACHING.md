# Hybrid/Recurrent Models and Prompt Caching in llama.cpp

Research findings on why Qwen3.6 (and similar hybrid/recurrent models) fail to reuse
restored KV cache from disk, despite successful `/slots/{id}?action=restore` calls.

Date: 2026-07-20

## Standard Transformer Caching (Baseline)

In a standard transformer, each token position has a Key and Value vector in the KV cache.
The cache is a random-access lookup table:

```
Position 0:  K0, V0  (system prompt start)
Position 1:  K1, V1  (system prompt cont.)
...
Position 99: K99, V99
```

When reusing a prefix, llama.cpp skips over cached positions — reads existing K/V and only
computes attention for new tokens. The cache is stateless: you can read any position without
affecting others, and you can freely truncate (delete positions 50–99) without corrupting
positions 0–49.

This is why standard transformers support simple prefix caching:
`memory_seq_rm(slot, 50, -1)` to truncate, remaining cache at 0–49 is still valid.

## Recurrent State (The Problem)

Recurrent models (SSMs like Mamba, or recurrent layers in Qwen3.6) maintain hidden state
that flows sequentially through the sequence. Unlike the KV cache (random-access), recurrent
state is cumulative:

```
Position 0:  recurrent_state = f(input_0, initial_state)
Position 1:  recurrent_state = f(input_1, state_from_0)
Position 2:  recurrent_state = f(input_2, state_from_1)
...
Position 99: recurrent_state = f(input_99, state_from_98)
```

Key difference: you cannot jump to position 50 without having processed 0–49. The recurrent
state at position 50 depends on everything before it. To resume from position 50, you need
the exact recurrent state that existed at that point.

A simple KV cache restore is insufficient — you've restored the attention K/V vectors, but
the recurrent layers have no saved state to resume from. They reprocess everything from scratch.

## Checkpoints: The Bridge

llama.cpp solves this with context checkpoints — snapshots of the full model state
(KV cache + recurrent hidden state) at specific positions:

```
Position 0–511:   computed normally
Checkpoint @ 511: saves KV + recurrent state to disk/memory

Position 512–1023: computed normally
Checkpoint @ 1023: saves KV + recurrent state

Position 1024–1535: computed normally
Checkpoint @ 1535: saves KV + recurrent state
```

When a new request arrives with a 1200-token prefix:
1. Find common prefix: 1200 tokens match
2. Search checkpoints for one at or before position 1200 → found checkpoint @ 1023
3. Restore the checkpoint (KV + recurrent state)
4. Only recompute positions 1024–1200 (176 tokens, not 1200)

Without checkpoints, there's no saved recurrent state to restore, so the model must
recompute from position 0.

## Hybrid Models (Qwen3.6)

Qwen3.6 is "hybrid" because it mixes both architectures in the same model:
- Attention layers: standard transformer self-attention (KV cache works normally)
- Recurrent layers: SSM-style state space models (need checkpoints for state rollback)

When llama.cpp processes a hybrid model, it must handle both:
- The KV cache for attention layers (random-access, easily truncated)
- The recurrent state for SSM layers (sequential, requires checkpoints)

Critical code path in `server-context.cpp:3284-3318`:

```cpp
// For hybrid/recurrent models, can we find a checkpoint to roll back to?
const auto it = std::find_if(
    slot.prompt.checkpoints.rbegin(),
    slot.prompt.checkpoints.rend(),
    [](const auto& cp) { return cp.pos_max <= pos_next; });

bool do_reset = (it == slot.prompt.checkpoints.rend());
if (do_reset) {
    // No checkpoint found — must reprocess everything
    n_past = 0;
}
```

## Why Slot Save/Restore Breaks for Hybrid Models

### Bug 1: Save Never Writes Checkpoints

`server-context.cpp:2529-2531` — the save handler only serializes the token list and
the sequence's KV cells:

```cpp
const llama_tokens tokens = slot->prompt.tokens.get_text_tokens();
const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(),
                                                slot->id, tokens.data(), token_count);
```

Checkpoint metadata is never written to disk.

### Bug 2: Restore Clears Checkpoints

`server-context.cpp:2576-2577` — the restore handler clears everything:

```cpp
slot->prompt.clear();              // clears tokens AND checkpoints
slot->prompt.tokens.insert(tokens);
```

Even if checkpoints existed in the slot before restore, they're deleted by `clear()`.

### Result

After a restore, `slot.prompt.checkpoints` is always empty. The next request finds the
token prefix match correctly (`get_common_prefix()` returns the full overlap), but the
checkpoint search fails → `do_reset = true` → `n_past = 0` → full recompute.

## Why In-Memory Cache Works But Disk Doesn't

The in-memory prompt cache (PR #16391) stores the entire `server_prompt` object, including
checkpoints. The disk save/restore only saves token IDs + KV cache, not checkpoint state.

This is why `cache_n > 0` with the in-memory cache, but `cache_n = 0` after disk restore.

## Checkpoint Min-Step Regression (b9354+)

A separate regression (issue #24055) broke `--checkpoint-min-step` in build b9354+.
Checkpoints are created but the checkpoint search logic fails to match them, producing
the same "forcing full prompt re-processing" message.

The older `--checkpoint-every-n-tokens` flag (pre-b9354) works correctly.

First bad commit: e98cb51

## Summary Table

| Architecture           | Cache Type           | Can Truncate? | Needs Checkpoints? | Disk Save/Restore Works? |
|------------------------|----------------------|---------------|-------------------|-------------------------|
| Pure transformer       | KV cache only        | Yes           | No                | Yes                     |
| (Llama, Mistral)       |                      |               |                   |                         |
| Pure recurrent         | Recurrent state only | No            | Yes               | **No** (upstream bug)   |
| (Mamba, RWKV)          |                      |               |                   |                         |
| Hybrid                 | KV + recurrent state | Partially     | Yes               | **No** (upstream bug)   |
| (Qwen3.6, Jamba)       |                      |               |                   |                         |

## Relevant Issues and PRs

| Reference | Title | Status |
|-----------|-------|--------|
| Issue #25913 | /slots save/restore silently loses all prompt reuse on hybrid/recurrent models — checkpoints are never persisted | Open |
| Issue #24055 | Context checkpoints always invalidated on hybrid/recurrent models | Open (bug-unconfirmed) |
| PR #16382 | Extended checkpoint requirement from SWA-only to hybrid/recurrent models | Merged |
| PR #16391 | In-memory prompt cache (stores full server_prompt with checkpoints) | Merged |
| PR #20955 | Recover from checkpoint on truncation failure for hybrid/recurrent models | Closed |
| ik_llama.cpp checkpoint fix | Community patch for checkpoint search logic (3 changes to server-context.cpp) | External |

## Proposed Fixes (from Issue #25913)

### Option 1: Sidecar File (No libllama Format Change)

In the save handler, write `filepath + ".ckpt"` containing each `common_prompt_checkpoint`'s
`n_tokens`, `id_task`, `pos_min`, `pos_max` and the length-prefixed `data_tgt` / `data_dft` /
`data_spec` blobs, behind a magic + version. In restore, rebuild `slot->prompt.checkpoints`
after the `prompt.clear()` at :2576. A missing or version-mismatched sidecar degrades to
today's behavior.

### Option 2: Unify the Two Paths

Serialize a whole `server_prompt_cache_state` so the on-disk format is a superset of the
in-memory cache entry. Removes the divergence that produced this bug, at the cost of a
larger change.

## Caveats

- Synthesizing a checkpoint at `pos_min = 0` in the restore handler is NOT a valid shortcut.
  The matcher would take the `cur.pos_min == 0` branch and decode on top of a recurrent
  state that has already consumed that token — silently corrupting output rather than
  merely being slow.
- Reuse after a fix will be bounded by checkpoint granularity (`--ctx-checkpoints`,
  `--checkpoint-min-step`), i.e., it should match in-memory behavior, not necessarily
  produce a full-prefix hit.
- `ctx_dft` state is also not saved, so `--model-draft` setups will desync after restore.

## Workarounds for Proxycache Users

1. **Use llama.cpp's in-memory prompt cache** — start with `--cache-ram <GB>` and let
   llama.cpp manage caching natively. This preserves checkpoints and works correctly for
   hybrid models.

2. **Patch llama.cpp** — the ik_llama.cpp checkpoint fix demonstrates a working patch for
   the checkpoint search logic. Three changes to `server-context.cpp`:
   - Fix checkpoint search to match on `pos_max <= pos_next`
   - Lower minimum token threshold from 64 to 4 for hybrid/recurrent cases
   - Rewrite interval guard to use last checkpoint's `pos_max` as reference

3. **Use pre-b9354 builds** — the `--checkpoint-every-n-tokens` flag works correctly
   before the b9354 regression.

4. **Accept partial restores** — for pure transformer models (non-hybrid), disk save/restore
   works correctly. The issue is specific to hybrid/recurrent architectures.
