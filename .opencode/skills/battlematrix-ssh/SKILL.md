---
name: battlematrix-ssh
description: SSH from WSL to BattleMatrix (battlematrix.lan), the Windows box with 2x Intel Arc B70 GPUs, and run SYCL benchmark commands on it. Use when the user asks to run tests or benchmarks on the B70 / BattleMatrix machine, or to check its GPUs.
---

# BattleMatrix SSH (B70 bench box)

Windows box `BattleMatrix` (`battlematrix.lan`, 192.168.1.54) with **2x Intel Arc B70**
(Battlemage). Used for SYCL decode/prefill benchmarking. The dev box where opencode runs
(ChibiGamer, WSL2) is a DIFFERENT machine; this skill covers the SSH hop and running
commands remotely.

## Connection

From WSL: `ssh b70 <cmd>`. Alias lives in `~/.ssh/config` on the dev box (user `<devbox-user>`):

```
Host b70
    HostName battlematrix.lan
    User localadmin
    IdentityFile ~/.ssh/id_ed25519_b70
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 4
```

Recreate if missing: `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_b70 -N ""` then have the
user add the new public key on the B70 (see auth gotcha below).
Public key in use (2026-08-21):
`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINDzUm8H6Q+cFo4y4hdJJzLEOryp+esQkAZpcqo9w9Mv opencode-b70-bench`

Never put a private key in this skill or in the repo.

## Hard-won gotchas (all actually hit during setup)

- The default shell is **cmd** (OpenSSH for Windows 9.5). Use cmd syntax: `&` to separate
  commands, `set VAR=x && <cmd>` for env vars. `;` is echoed literally. An HKCU
  `SOFTWARE\OpenSSH` `DefaultShell=powershell.exe` override did NOT take effect; do not
  fight it, just use cmd.
- `localadmin` is in the administrators group, so sshd's `Match Group administrators`
  block REPLACES AuthorizedKeysFile with `C:\ProgramData\ssh\administrators_authorized_keys`.
  The personal `C:\Users\LocalAdmin\.ssh\authorized_keys` is silently ignored for this
  user. The key was added to the administrators file (in an elevated PowerShell):
  `Add-Content -Path "C:\ProgramData\ssh\administrators_authorized_keys" -Value '<pubkey>' -Encoding ASCII`
- Windows PowerShell 5.1 `Add-Content`/`Set-Content` default to UTF-16LE; OpenSSH cannot
  parse UTF-16 key files. Always pass `-Encoding ASCII`.
- The Windows firewall allows only TCP 22 inbound by default. For a temp debug sshd on
  another port: `New-NetFirewallRule -Name tmp-<port> -Direction Inbound -Protocol TCP -LocalPort <port> -Action Allow`
  (delete after: `Remove-NetFirewallRule -Name tmp-<port>`).
- A non-elevated `sshd.exe -ddd` cannot load the host private keys (exit "no hostkeys
  available"); run it elevated if you ever need server-side debug traces.
- WSL does not forward env vars to the remote command; set them in the remote command line.
- `icacls` grant syntax `user:(R)` gets mangled by PowerShell native-arg passing
  ("Invalid parameter (R)"); use `icacls --% <literal path> ...` or skip it (default
  inherited ACLs already give SYSTEM/Administrators what sshd needs).

## Running commands

**CRITICAL: Before ANY `taskkill` of llama-server, check which PIDs are running and
exclude the user's instance.** The user's long-running server is on SYCL1 (RDP-Tcp#0
session). Always run `ssh b70 "tasklist | findstr /i llama-server"` first, identify
the user's PID (the one on RDP-Tcp#0 or with ~15GB+ RAM), then use
`taskkill /f /im llama-server.exe /fi "PID ne <user_pid>"`. NEVER use bare
`taskkill /f /im llama-server.exe` - it will kill the user's server.
The user's PID changes when they restart, so re-check every time.

```
# simple
ssh b70 "hostname & whoami"

# GPU pin + benchmark (see dual-GPU note)
ssh b70 "C:\Users\LocalAdmin\Desktop\llama-cpp-sycl16\llama-bench.exe --device SYCL0 -m model.gguf -n 32"
```

- **Dual B70: always pin the device** with `--device SYCL0` (or `SYCL1`, or `SYCL0,SYCL1`
  for both). The flag is singular and comma-separated; this build REJECTS `--devices`
  (error: invalid parameter). `set ONEAPI_DEVICE_SELECTOR=level_zero:0 && <cmd>` is the
  process-level alternative.
- **Device ownership convention (the box has 2 B70s, one instance per GPU is fine):**
  the USER's long-running llama-server lives on **SYCL1**; MY benchmark runs always pin
  **SYCL0**. Never put two instances on the same GPU (2 x 20 GB = 40 GB > 32 GB card ->
  Level Zero oversubscribes, buffers go host-backed ~56 GB/s, sshd starves).
- **NEVER launch a second 27B load while the user's server is already resident**, even on
  the other GPU: two concurrent 20 GB model loads wedge the box (disk/CPU/host-RAM pressure)
  - sshd resets ("Connection reset by peer") while ping still answers. Check
  `ssh b70 "tasklist | findstr /i llama"` FIRST; if any llama process is up, stop and ask
  before launching another. (Note: `wmic` is gone on this Windows build; use `tasklist`.)
- **Avoid VRAM idle drain in harnesses:** the user wants `--gpu-heartbeat 5` passed to
  llama-cli / llama-server, BUT the deployed `llama-cpp-sycl16` build does NOT have that flag
  (it errors `invalid argument: --gpu-heartbeat`). It only has `--warmup,--no-warmup` (warmup
  on by default). Do NOT pass `--gpu-heartbeat` to this build; if a rebuild adds it, pass
  `--gpu-heartbeat 5` then.
- Long runs: the local bash tool has its own timeout, so pass an explicit `timeout`
  (e.g. 600000 ms) for multi-minute llama-bench runs. For very long jobs, detach on the
  remote: `start /min cmd /c "<cmd> > G:\bench.log 2>&1"` then poll the log with a second
  ssh.
- Backslashes: single-quote the ssh arg in bash or escape `\` where bash would eat it;
  `%` in the remote string is safe (bash does not touch it).

## Build layout

- llama.cpp SYCL f16 build + runtime dlls: `C:\Users\LocalAdmin\Desktop\llama-cpp-sycl16`
  (llama-bench.exe, llama-server.exe, ggml-sycl.dll, sycl9.dll, ...). Build + deploy
  workflow: see the `sycl16-build-deploy` skill.
- Model symlink: `D:\model.md` -> Qwen3.8-27B-UD-Q6_K.gguf (20.5 GiB dense Q6_K).
  Use this short path in all commands - avoids quote-escaping issues over SSH with the
  long huggingface_cache path. If missing, recreate:
  `mklink D:\model.md "D:\huggingface_cache\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\<hash>\Qwen3.8-27B-UD-Q6_K.gguf"`
- Slot saves: `D:\slot-save\fa-decode-142k` (142K KV), `D:\slot-save\kv-64k` (64K KV).
- Helper scripts in the deploy dir: run-llama-bench.ps1, sycl.conf.

## Launching llama-server over SSH

`start /B`, `Start-Process`, and `wmic` all fail or hang over SSH. The working pattern
is: background the ssh in local bash, sleep, health-check from a second ssh, then kill
the first ssh PID:

```bash
# Start server (no MTP) on SYCL0, port 8082, with Q8_0 KV cache:
ssh b70 "cmd.exe /C \"cd /d C:\Users\LocalAdmin\Desktop\llama-cpp-sycl16 && set GGML_SYCL_PROFILE=0 && llama-server.exe --device SYCL0 -m D:\model.md -c 170000 -ctk q8_0 -ctv q8_0 --jinja --port 8082 -np 1 --no-ui > server-test.log 2>&1 &\"" &
SSH_PID=$!
sleep 55
ssh b70 "curl -s http://localhost:8082/health"   # expect {"status":"ok"}
# ... run tests ...
# Kill only my PID (never /IM):
ssh b70 "taskkill /F /PID <my_pid>"
kill $SSH_PID 2>/dev/null
```

With MTP speculative decoding (post-rebase flag names):
```
--spec-type draft-mtp --spec-draft-n-max 4
```
(Old names `-sp mtp` and `--draft-max N` no longer exist.)

Completion request with slot-save load:
```bash
ssh b70 "curl -s http://localhost:8082/completion -d \"{\\\"prompt\\\":\\\"hello\\\",\\\"stream\\\":false,\\\"max_tokens\\\":32,\\\"slot_save\\\":\\\"D:\\\\slot-save\\\\fa-decode-142k\\\"}\""
```

## GPU per-op profiling (GGML_SYCL_PROFILE)

Set `GGML_SYCL_PROFILE=1` in the server launch env. The profiler calls `stream->wait()`
after each op, serializing GPU execution and attributing wall time to each op type.
Output is written to `sycl-profile.log` (relative to CWD) via `ggml_sycl_profile_write`.

Warning: this adds ~10ms/step overhead (serialization). Use for breakdowns, not for
throughput numbers. For throughput, use `GGML_SYCL_PROFILE=0`.

## Network share (\\epycdesktop\media)

- Hosts the CORSAIR AI drive; `\\epycdesktop\media\CORSAIRAI Backup\llama-cpp-sycl16` is
  the dev->B70 staging dir (dev box pushes via copy-sycl16.bat, B70 pulls via its own
  copy-sycl16.bat).
- The dev box authenticates with a stored cmdkey credential
  (`Domain:target=epycdesktop`, user 122abalone\administrator).
- The B70's `localadmin` has NO stored credential (cmdkey empty). The user's interactive
  RDP session (RDP logon as LocalAdmin, session 2) holds the password in its token and
  sees mapped drive Z:; my key-based ssh logon gets Access denied on the share no matter
  what, because a key logon has no password to piggyback on.
- Therefore deploy via scp from the dev box (see sycl16-build-deploy), not via the share.
