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
- Model: `D:\huggingface_cache\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots\27af057ecb382ddfea5d12837360a8980560e3ed\Qwen3.8-27B-UD-Q6_K.gguf`
  (20.5 GiB dense Q6_K).
- Helper scripts in the deploy dir: run-llama-bench.ps1 (formal bench, -b 2048 -p 32768
  --device SYCL0), run-qwen3.8-multi.ps1 (2-GPU server, 524k ctx, MTP draft on SYCL1),
  sycl.conf (device allowlist).

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
