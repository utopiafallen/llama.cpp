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
  Use the wrapper bat (created 2026-08-21): `G:\llama-cpp-src\build-sycl16-oneapi.bat`
  (calls setvars.bat intel64 vs2026, then build-sycl16.bat).
- Run: `cmd.exe /C "G:\llama-cpp-src\build-sycl16-oneapi.bat"` with a long bash timeout
  (SYCL recompile of ggml-sycl.dll is slow; no-op if nothing changed).
- Targets (build-sycl16.bat): llama-server, ggml-rpc-server, llama-bench, llama-perplexity,
  llama-cli (llama-cli was added 2026-08-21 for output-correctness checks;
  llama-bench/perplexity both miss garbled output).
- Output: `G:\llama-cpp-src\build-x64-windows-sycl-release-f16\bin\`
- Reconfigure ONLY when build flags change: `cmake-sycl.bat` (needs oneAPI env; it sets
  LEVEL_ZERO_V1_SDK_PATH, preset x64-windows-sycl-release-f16,
  GGML_SYCL_DEVICE_ARCH=bmg_g21, GGML_RPC=ON). It has no wrapper bat yet - make one the
  same way as build-sycl16-oneapi.bat if needed.

## 2. Deploy (dev -> B70, via scp)

- B70 deploy dir: `C:\Users\LocalAdmin\Desktop\llama-cpp-sycl16`
- From WSL, using the `b70` ssh alias (key auth, see battlematrix-ssh skill):
  ```
  scp /mnt/g/llama-cpp-src/build-x64-windows-sycl-release-f16/bin/<changed>.exe \
      /mnt/g/llama-cpp-src/build-x64-windows-sycl-release-f16/bin/<changed>.dll \
      b70:"C:/Users/LocalAdmin/Desktop/llama-cpp-sycl16/"
  ```
- Verify with MD5 on both sides: dev box `cmd.exe /C "certutil -hashfile <f> MD5"`,
  B70 `ssh b70 "certutil -hashfile <f> MD5"`.
- IMPORTANT: after a CMake RECONFIGURE (cmake-sycl-oneapi.bat) the whole build is
  recompiled, so ALL the ggml-*/llama-*.dll change together. Deploying only the
  changed .dll/.exe then leaves stale companion dlls on the B70 -> ABI mismatch ->
  the exe runs, prints "Loading model...", then dies with EXIT=-1073740791
  (0xC0000409 fastfail). Symptom is easy to misread as a kernel bug. Fix: after any
  reconfigure, deploy the ENTIRE bin (`scp bin/*.dll bin/*.exe b70:...`), not just
  the changed files. (The runtime dlls sycl9/mkl/tbb/ur_* are NOT in bin/ - leave them.)
- Do NOT use the SMB share (\\epycdesktop\media\...) from an ssh logon: access denied.
  The user's interactive RDP session can reach it, but the key-based ssh logon has no
  stored credential (cmdkey empty) and cannot. scp is the working deploy path.

## 3. Smoke test (B70)

Model: `D:\huggingface_cache\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-Q6_K.gguf` (20.5 GiB, dense Q6_K)

```
ssh b70 'cd /d C:\Users\LocalAdmin\Desktop\llama-cpp-sycl16 & llama-bench.exe --device SYCL0 -m "D:\huggingface_cache\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-Q6_K.gguf" -p 256 -n 32'
```

Baseline (2026-08-21, pre-optimization, single GPU): pp256 ~753 t/s, tg32 ~21.5 t/s.
The trailing `build: <commit>` line confirms which binary actually ran.

- Device pinning: `--device SYCL0` / `--device SYCL0,SYCL1`. This build REJECTS
  `--devices` (error: invalid parameter). The flag is singular, comma-separated list.
- Machine has 2x B70 - always pin with --device, or results are confounded.
- A/B env vars (from run-llama-bench.ps1): GGML_SYCL_ENABLE_ESIMD, GGML_SYCL_Q6K_GEMV_ROW,
  GGML_SYCL_PRIORITIZE_DMMV, GGML_SYCL_FUSE_MM_ADD, GGML_SYCL_FUSE_MM_GLU,
  GGML_SYCL_FUSE_GDN_DT, GGML_SYCL_PROFILE, GGML_SYCL_PROFILE_FILE. In cmd:
  `set VAR=x && llama-bench.exe ...`
- Model load takes ~30-60 s; give ssh bash calls a 600000 ms timeout.
- Formal benchmark script on the B70: run-llama-bench.ps1
  (`-b 2048 -p 32768 --device SYCL0` + the env vars above).

## 4. Correctness sanity check (after any kernel change)

Throughput and perplexity both fail to flag garbled output; verify real text:

```
ssh b70 'cd /d C:\Users\LocalAdmin\Desktop\llama-cpp-sycl16 & llama-cli.exe --device SYCL0 --simple-io --no-warmup -c 2048 --reasoning off -m "D:\huggingface_cache\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-Q6_K.gguf" -st -p "The capital of France is" -n 48'
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
