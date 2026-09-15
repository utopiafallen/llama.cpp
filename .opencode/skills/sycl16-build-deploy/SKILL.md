---
name: sycl16-build-deploy
description: Build the llama.cpp SYCL f16 (bmg_g21) targets on the dev box and deploy them to the BattleMatrix B70 bench machine, then smoke-test. Use when the user asks to rebuild or redeploy the SYCL build, update the B70 binaries, or benchmark after a kernel change.
---

# SYCL f16 build + deploy to BattleMatrix

Loop: build on dev box (ChibiGamer, WSL2) -> scp changed binaries to B70 -> smoke-test there.
Connection details for the B70 live in the `battlematrix-ssh` skill.

## 1. Build (dev box, from WSL)

- oneAPI env is required. WSL interop mangles double quotes in argv to Windows processes
  (they arrive as `\"`), so do NOT inline `setvars.bat` in the ssh/cmd command line.
  Use the wrapper bat: `G:\llama-cpp-src\build-full-oneapi.bat`
  (calls setvars.bat intel64 vs2026, then cmake --build with all targets).
- Run: `cmd.exe /C "G:\llama-cpp-src\build-full-oneapi.bat"` with a long bash timeout
  (SYCL recompile of ggml-sycl.dll is slow; CMake handles incremental builds natively,
  no-op if nothing changed).
- Targets: llama-server, llama-cli, llama-bench (all in one cmake --build invocation).
- Output: `G:\llama-cpp-src\build-x64-windows-sycl-release-f16\bin\`
- **NEVER delete .obj files or individual build artifacts to force a "targeted" rebuild.**
  CMake already performs correct incremental builds. Deleting objects causes link errors,
  stale ABI mismatches, and wastes the full recompile time you were trying to save.
- Reconfigure ONLY when build flags change: `cmake-sycl-oneapi.bat` (needs oneAPI env;
  sets LEVEL_ZERO_V1_SDK_PATH, preset x64-windows-sycl-release-f16,
  GGML_SYCL_DEVICE_ARCH=bmg_g21, GGML_RPC=ON). After a reconfigure, ALL dlls change.

## 2. Deploy (dev -> B70, via scp)

- B70 deploy dir: `C:\Users\LocalAdmin\Desktop\llama-cpp-sycl16`
- **Always copy the FULL binary set.** Never pick individual files you think changed.
  Stale companion DLLs cause ABI mismatches (exe loads, prints "Loading model...", then
  fastfail 0xC0000409). The full set is ~20 files and copies in seconds:
  ```
  scp /mnt/g/llama-cpp-src/build-x64-windows-sycl-release-f16/bin/* b70:"C:/Users/LocalAdmin/Desktop/llama-cpp-sycl16/"
  ```
- Before deploying, kill ONLY your own llama-server (Services session). See
  `battlematrix-ssh` skill for the PID identification rules.

## 3. Smoke test (B70)

Model: `D:\model.md` (symlink to Qwen3.8-27B-UD-Q6_K.gguf, 20.5 GiB dense Q6_K)

```
ssh b70 "cd /d C:\Users\LocalAdmin\Desktop\llama-cpp-sycl16 & llama-bench.exe --device SYCL0 -m D:\model.md -p 256 -n 32"
```

Baseline (2026-09-14, Q8_0 FA tile load, single GPU): pp32768 ~931 t/s, tg32 ~22.7 t/s.
Pre-Q8_0-fa-fix baseline: pp256 ~753 t/s, tg32 ~21.5 t/s.
The trailing `build: <commit>` line confirms which binary actually ran.

- Device pinning: `--device SYCL0` / `--device SYCL0,SYCL1`. This build REJECTS
  `--devices` (error: invalid parameter). The flag is singular, comma-separated list.
- Machine has 2x B70 - always pin with --device, or results are confounded.
- A/B env vars: GGML_SYCL_ENABLE_ESIMD, GGML_SYCL_Q6K_GEMV_ROW,
  GGML_SYCL_PRIORITIZE_DMMV, GGML_SYCL_FUSE_MM_ADD, GGML_SYCL_FUSE_MM_GLU,
  GGML_SYCL_FUSE_GDN_DT, GGML_SYCL_PROFILE, GGML_SYCL_PROFILE_FILE. In cmd:
  `set VAR=x && llama-bench.exe ...`
- Model load takes ~30-60 s; give ssh bash calls a 600000 ms timeout.

## 3b. llama-server for decode perf testing

Full parameter list for a decode performance test server:

```
llama-server.exe --device SYCL0 -m D:\model.md -c 170000 \
  -ctk q8_0 -ctv q8_0 \
  --jinja --port 8082 -np 1 --no-ui \
  --gpu-heartbeat 5 --load-mode none \
  --slot-save-path D:\slot-save\ \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --presence-penalty 1.0 --min-p 0.00 --repeat-penalty 1.0
```

**Slot save/restore workflow (do NOT use `slot_save` in the request body - it is ignored):**

The server is STATELESS. Each completion request must carry the full conversation history.
The slot cache only persists the KV cache, NOT the prompt/token state.

1. **First run (establishes baseline + creates slot save):**
   - Send the full chat/completion request (e.g. `TestPrompt152k.json`) with a small
     `max_tokens` (e.g. 32-512). This does the full prefill AND generates decode tokens.
     The decode timing from THIS run is valid.
   - **Use a long bash timeout (600000+ ms)** - initial prefill at 144K takes minutes.
   - Save the slot: `curl POST /slots/0?action=save` with `{"filename":"my-save"}`
2. **After server restart (re-test with changes):**
   - Restore: `curl POST /slots/0?action=restore` with `{"filename":"my-save"}`
     (loads KV cache only, takes seconds instead of minutes)
   - Send the SAME full prompt JSON again (same messages/conversation). The prefix hits
     the restored KV cache (fast), then decode begins. Measure the decode timing.
3. **Do NOT send a short/" " prompt after restore** - that only has 1 token of input,
   the server doesn't know about the 144K context from the request's perspective.

Read results from server log `print_timing` lines, NOT the API response
(`eval_tokens_per_second` in the JSON is often 0.0 for speculative/chat requests).

Decode baseline (2026-09-15, fresh 144K prefill, SYCL1): **8.59 t/s** (116 ms/tok).
Test prompts on desktop: `TestPrompt152k.json` (full), `TestPrompt152k-512.json`,
`TestPrompt152k-32.json`, `test-64k.json` (~25K tokens).

## 4. Correctness sanity check (after any kernel change)

Throughput and perplexity both fail to flag garbled output; verify real text:

```
ssh b70 "cd /d C:\Users\LocalAdmin\Desktop\llama-cpp-sycl16 & llama-cli.exe --device SYCL0 --simple-io --no-warmup -c 2048 --reasoning off -m D:\model.md -st -p \"The capital of France is\" -n 48"
```

Expected: a coherent answer starting "The capital of France is **Paris**.", then
`[ Prompt: xx t/s | Generation: ~21 t/s ]` and a clean exit. Any random tokens /
repetitions / "Exiting" without text = broken kernels.

- `-st` (single-turn) is REQUIRED: this model has a chat template, so conversation mode
  auto-enables and the CLI otherwise drops into interactive mode, hanging on stdin over
  ssh (and exhausting machine RAM while it sits there).
- `--reasoning off`: the model thinks by default; a thinking trace can eat the whole
  -n budget before any answer appears.
- `--simple-io` for ssh console compatibility; `-c 2048` keeps the KV cache small.

## Phase 0 findings (Qwen3.8-27B-UD-Q6_K, single SYCL0, 2026-08-21)

Decode is weight-streaming bound. Do not re-run these A/Bs without a kernel change:

| config | tg128 t/s |
| --- | --- |
| eager, all fusions on (DEFAULT, best) | 21.47 |
| GGML_SYCL_ENABLE_ESIMD=0 (MMVQ q8_1/dp4a) | 20.84 |
| GGML_SYCL_Q6K_GEMV_ROW=1 (gather variant) | 16.29 |
| GGML_SYCL_FUSE_MM_ADD=0 (alone) | 21.44 (no-op) |
| GGML_SYCL_FUSE_MM_GLU=0 (alone) | 21.43 (no-op) |
| GGML_SYCL_ENABLE_GRAPH=1 + ONEAPI_DEVICE_SELECTOR=level_zero:0 | 17.93 (-16%) |
| 2-GPU --device SYCL0,SYCL1 -ts 54,46 | 20.36 (worse than 1-GPU) |

- pp32768 baseline: 953 t/s (-b 2048). Model loads fully on ONE 32GB Arc Pro B70
  (all 64 layers, no CPU offload); 31.9 GiB free before load.
- 21.47 t/s x 22.0 GB = 472 GB/s = 74% of the 640 GB/s GDDR6 spec. Decode wall time
  is GEMV DRAM streaming; op count, fusions, launch gaps all negligible (ablated).
- SYCL graph mode is a net -16% as implemented: per-call begin_recording +
  exec_graph->update() runs while the GPU is idle (after the logits sync), while
  eager enqueues overlap under GPU execution. Graph also needs ONEAPI_DEVICE_SELECTOR
  to one GPU (check_graph_compatibility bails when device_count>1) and the CONCAT
  rejection is stale (concat.cpp is enqueue-only; the model's 54 CONCAT/token go the
  non-cont kernel path). Real fix = capture-once/replay-many with stable pointer
  indirection (CUDA-graph style) - not implemented, needs discussion.
- llama-bench hides INFO logs without -v; llama-bench has no -lv flag.
- Model quirks: 54 CONCAT/token (GDN conv_input), nextn_predict_layers=1 (MTP, unused
  by llama-bench), mixed quants (q6_K/q8_0/q5_K/q4_K/iq4_xs, unsloth dynamic).
- SYCL builds are NOT bit-reproducible: same source rebuilds give different MD5s.
