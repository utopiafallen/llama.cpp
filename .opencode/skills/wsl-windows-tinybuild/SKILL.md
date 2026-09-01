---
name: wsl-windows-tinybuild
description: Build and run a small native Windows test (e.g. a ggml graph-behavior repro) from WSL using only icx.exe and link.exe. Use when WSL has no gcc/clang but the source is on the G: drive and a full SYCL CMake build is overkill.
---

# WSL -> Windows tiny build

WSL (Ubuntu, python3 only, no gcc/clang/venv) + Windows with Intel oneAPI and VS 2026. Goal:
compile a few objects with `icx.exe`, link with `link.exe`, run the .exe from WSL. Working
example from a real session: `/mnt/g/tmp_repro/` (fuse_repro.exe, libs/, .obj, .cpp).

## Hard-won constraints

- WSL interop does NOT forward env vars to Windows processes. `LIB=... INCLUDE=... cmd.exe
  /c "echo %LIB%"` prints unexpanded. Do not rely on env; use command-line flags or copied libs.
- icx.exe (clang-cl) mangles flag args containing spaces when crossing WSL argv:
  `-LIBPATH:"C:\Program Files (x86)\..."` arrives truncated at the space and is dropped
  silently. icx also does not accept `-LIBPATH:` at all (warns: unknown argument).
  Solution: never pass space-containing paths on the command line. Use 8.3 short names where
  needed (`C:\PROGRA~2\...`) or, better, link directly with link.exe against a space-free
  lib dir on G:.
- WSL globs are case-sensitive. `*.lib` misses `kernel32.Lib` / `uuid.Lib`.
  (`shopt -s nocaseglob` helps.)
- `oldnames.lib` is referenced by icx-compiled objects but does not exist anywhere on a stock
  oneAPI install - stub it.
- link with icx's own driver cannot find the Intel CRT libs without the env it never sees, so
  call `link.exe` directly.

## Toolchain locations (this machine)

- icx.exe:        /mnt/c/PROGRA~2/Intel/oneAPI/compiler/latest/bin/icx.exe
- link/lib .exe:  "/mnt/c/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/"
- oneAPI libs:    /mnt/c/PROGRA~2/Intel/oneAPI/compiler/2026.1/lib
- MSVC libs:      ".../MSVC/14.51.36231/lib/x64"
- ucrt libs:      "/mnt/c/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0/ucrt/x64"
- um libs:        ".../Lib/10.0.26100.0/um/x64"

Look up current versions from any existing CMakeCache.txt (CMAKE_CXX_COMPILER, CMAKE_LINKER)
if paths differ.

## Steps

1. Space-free workdir: `mkdir -p /mnt/g/tmp_repro/libs`
2. Copy libs (all into /mnt/g/tmp_repro/libs):
   - oneAPI: libircmt.lib libmmt.lib svml_dispmt.lib and the libirc*.lib set
   - MSVC `lib/x64`: all `*.lib` (libcmt, libcpmt, libvcruntime, libucrt, vcruntime, ...)
   - ucrt `x64`: all `*.lib`
   - um `x64`: all `*.lib` plus the differently-cased ones (kernel32.Lib)
3. Stub oldnames.lib:
   ```
   echo "" > /mnt/g/tmp_repro/empty.c
   icx.exe /nologo -O1 -x c -c G:/tmp_repro/empty.c -o G:/tmp_repro/empty.obj
   lib.exe /nologo /out:G:/tmp_repro/libs/oldnames.lib G:/tmp_repro/empty.obj
   ```
   (lib.exe with no members exits 0 but creates nothing - always give it an object.)
4. Compile. /TP for C++, `-x c` for C files. Defines for the ggml core:
   `-DGGML_STATIC -DGGML_VERSION=0 -DGGML_COMMIT=0`. Add `/EHsc` where the file uses try
   (ggml-backend-reg.cpp, ggml-backend-meta.cpp).
   ggml-core object chain needed to link graph-API code (each missing one shows as LNK2019):
   ggml.c, ggml-quants.c, ggml-backend.cpp, ggml-backend-reg.cpp, ggml-backend-meta.cpp,
   ggml-backend-dl.cpp (for dl_get_sym/dl_error in backend-reg), ggml-threading.cpp,
   ggml-alloc.c. Includes: `-IG:/llama-cpp-src/ggml/include -IG:/llama-cpp-src/ggml/src`
   (both the `/mnt/g/...` and `G:/...` spellings worked).
5. Link directly, one libpath with no spaces:
   ```
   link.exe /nologo /out:G:/tmp_repro/test.exe G:/tmp_repro/*.obj /LIBPATH:G:/tmp_repro/libs
   ```
   Iterate on LNK1104: each "cannot open file 'X.lib'" is one more lib to copy (the chain in
   a real session: libircmt -> libcpmt -> libcmt -> libvcruntime -> libucrt -> uuid.Lib ->
   kernel32.Lib, then oldnames via the stub).
6. Run `/mnt/g/tmp_repro/test.exe` straight from WSL.

## ggml graph-behavior repro template (the use case)

Answer "does ggml consider this node pattern fusable / what are the flags and use counts"
without a SYCL build:

```cpp
#include "ggml.h"
#include "ggml-impl.h"   // raw ggml_can_fuse(cgraph, idx, ops, n), ggml_node_get_use_count
...
struct ggml_init_params p = {1024*1024*1024, NULL, false};
struct ggml_context * ctx = ggml_init(p);
struct ggml_cgraph *  gf  = ggml_new_graph(ctx);
// build the exact op topology, e.g.
struct ggml_tensor * mm  = ggml_mul_mat(ctx, w, x);
struct ggml_tensor * add = ggml_add(ctx, mm, res);
ggml_build_forward_expand(gf, add);
// find mm in gf->nodes, print op/flags/use of the chain, then:
enum ggml_op ops[2] = {GGML_OP_MUL_MAT, GGML_OP_ADD};
printf("%d\n", ggml_can_fuse(gf, i, ops, 2));
```

Notes: `ggml_init_params` is {mem_size, mem_buffer, no_alloc}; cgraph flags (COMPUTE=16) and
`use_counts` are already populated by `ggml_build_forward_expand`, matching what backends see.
Keep tensor sizes small (F32) - the repro tests graph bookkeeping, not numerics.
