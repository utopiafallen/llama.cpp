---
name: sycl-xmx-joint-matrix
description: Intel XMX (joint_matrix) optimization reference for SYCL kernels on Battlemage (B70). Tile sizes, layout rules, FA patterns, LDS staging, performance data. Use when writing or debugging any joint_matrix kernel, choosing XMX tile parameters, or diagnosing XMX JIT link errors.
---

# Intel XMX / joint_matrix on B70 (Battlemage)

## Hardware

- B70 = bmg_g31 at runtime (AOT targets bmg_g21; same family, AOT binaries run fine).
- 53 matrix_combinations (iGPU has 0 - check `has_xmx` before dispatching XMX kernels).
- Sub-group width: **16 lanes** (all XMX tiles are per-sub_group).
- Dense 1024^3 GEMM benchmark: ~27 TFLOPS fp16 (13.2x vs naive scalar FMA).
- LDS: work-group shared memory via `syclex::work_group_static`. B70 has 1MB LDS per EU partition.

## Tile constraints (oneAPI 2026.1, confirmed by JIT)

| tile | A layout | B layout | status |
|------|----------|----------|--------|
| M=8/16, N=16, K=16 | row_major | col_major | WORKS |
| M=8/16, N=16, K=16 | row_major | row_major | WORKS |
| M=32, N=64, K=16 | row_major | **row_major** | **WORKS** (the big win) |
| M=32, N=64, K=16 | row_major | col_major | **MISSING** (`OpJointMatrixLoadINTEL_PackedB_ColumnMajor_SG16_32x64` undefined) |
| M=32, N=64, K=32 | row_major | col_major | **MISSING** (same builtin, different K) |

### CRITICAL: BMG uses `layout::row_major` for B, NOT `col_major`

This is the single most important finding. The oneAPI 2026.1 backend for Battlemage
does NOT have a 32x64 col_major B-load builtin. However it DOES have row_major variants.
The dkhaldi/sycl_joint_matrix_kernels reference (220 TFlops on BMG) uses row_major for both A and B.

For VNNI architectures (DG2, SPR/AMX), B is loaded as `layout::ext_intel_packed` (VNNI layout).
For BMG without VNNI, everything is row_major.

## Layout semantics

### A matrix: `joint_matrix<sg, T, use::a, M, K, layout::row_major>`
- Memory access: `A[m][k] = ptr[m * ldm + k]`
- `ldm` = element stride between consecutive A rows (passed to `joint_matrix_load`)
- For Q in FA: `ldm = head_dim` (Q stored as `[row][dim]`)

### B matrix: `joint_matrix<sg, T, use::b, K, N, layout::row_major>`
- Memory access: `B[k][n] = ptr[k * ldm + n]`
- `ldm` = element stride between consecutive K-rows
- For V in FA PV: `ldm = pos_stride` (V stored as `[pos][dim]`, so B[k][n] = V[pos_c+k][dim_dc+n])

### B matrix: `joint_matrix<sg, T, use::b, K, N, layout::col_major>`
- Memory access: `B[k][n] = ptr[n * ldm + k]`
- Only available for N=16 on BMG (the 32x64 variant is missing)
- For K in FA QK^T (old approach): B[d][pos] with `ldm = pos_stride`

## Flash attention patterns

### QK^T: scores[M][SPLIT] = Q[M][D] @ K[SPLIT][D]^T

We need `B[k][n] = K[pos_n][dim_k]` i.e. B in [dim][pos] order. K is stored as [pos][dim].

**Option A (N=16, col_major from global):** Works for 16x16 tiles only.
```cpp
// B[k][n] = ptr[n*ldm + k] = K_h[(pos_c+n)*pos_stride + dim_k]
joint_matrix_load(sg, B_jm, xmp_g_h(K + ...), k_pos_stride);
```
This reads K directly from global, strided by pos_stride. No transpose needed but limited to N=16.

**Option B (N=64, row_major from LDS with transposed K):** The fast path.
```cpp
// 1. Stage K transposed into LDS: K_T[k*NTILE + n] = K[pos_c+n][dim_dc+k]
const sycl::half * Ksrc = K + kv_head*k_head_stride + (pos_base + c*NTILE)*k_pos_stride + dc*KTILE;
for (int i = lane; i < KTILE*NTILE; i += 16) {
    const int k = i / NTILE, n = i % NTILE;
    K_T[i] = Ksrc[n * k_pos_stride + k];
}
sg.barrier();

// 2. Load B from LDS with row_major: B[k][n] = K_T[k*NTILE + n]
joint_matrix_load(sg, B_jm, xmp_l_h(K_T), NTILE);
```

The transpose access pattern is coalesced: lane i reads `Ksrc[(i%NTILE)*pos_stride + i/NTILE]`.
Within each 64-element row of K_T, the 16 lanes read 16 consecutive halfs from the same K row.
LDS write stride is 16 elements per lane (minor bank conflicts, acceptable).

Cost: 16x64 = 2KB LDS buffer per K tile chunk, one barrier per chunk. Negligible vs the XMX MAD.

### PV: O[M][D] += P[M][SPLIT] @ V[SPLIT][D]

V is stored as [pos][dim] which IS the natural [K][N] row-major order. No transpose needed!
Load directly from global:
```cpp
// B[k][n] = V[(pos_c+k)*v_pos_stride + dim_dc+n], ldm = v_pos_stride
joint_matrix_load(sg, B_jm, xmp_g_h(V + kv_head*v_head_stride + (pos_base+c*KTILE)*v_pos_stride + dc*NTILE), v_pos_stride);
```

This works at any N tile size (N=16, 32, 64) since it's row_major global.

### A matrix from LDS (Q or P)

Both Q and P are staged into LDS as [M][stride] row-major:
```cpp
joint_matrix_load(sg, A_jm, xmp_l_h(Q16 + dc*KTILE), D);  // ldm = D (head_dim)
joint_matrix_load(sg, A_jm, xmp_l_h(P16 + c*KTILE), SPLIT); // ldm = SPLIT
```

## API reference

```cpp
#include <sycl/ext/intel/experimental/matrix.h>
namespace mx = sycl::ext::oneapi::experimental::matrix;

// Types
mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, M, K, mx::layout::row_major> A;
mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, K, N, mx::layout::row_major> B;
mx::joint_matrix<sycl::sub_group, float,    mx::use::accumulator, M, N> C;

// Core operations
mx::joint_matrix_fill(sg, C, 0.0f);                        // zero accumulator (uniform value)
mx::joint_matrix_load(sg, A, ptr, ldm);                     // load A tile from memory
mx::joint_matrix_load(sg, B, ptr, ldm);                     // load B tile (layout from template)
mx::joint_matrix_mad(sg, C, A, B, C);                       // C = A*B + C (in-place accumulate)
mx::joint_matrix_store(sg, C, ptr, ldm, mx::layout::row_major); // store C
mx::joint_matrix_copy(sg, src, dst);                        // convert between use types/layouts

// Element-wise (no coordinates - position-agnostic op)
mx::joint_matrix_apply(sg, C, [](float &x) { x *= alpha; });              // unary
mx::joint_matrix_apply(sg, A, B, [](float &x, float &y) { y = x*alpha; });// binary

// Element-wise WITH coordinates (Intel extension, aspect::ext_intel_matrix)
namespace imx = sycl::ext::intel::experimental::matrix;
imx::joint_matrix_apply(sg, C, [](float &x, size_t row, size_t col) { ... });
// Callback gets (T& element, size_t row, size_t col) - can index lookup tables

// Prefetch
mx::joint_matrix_prefetch<M,K>(sg, ptr, stride, layout);   // prefetch into L1/L2

// Checked load/store (aspect::ext_intel_matrix_checked, OOB-safe)
imx::joint_matrix_load_checked(sg, A, base, ldm, global_rows, global_cols, row_idx, col_idx);

// Pointer helpers (local/global)
auto xmp_l_h = [](const sycl::half *p) -> auto { /* multi_ptr to local_space */ };
auto xmp_g_h = [](const sycl::half *p) -> auto { /* multi_ptr to global_space */ };
```

Key notes:
- `multi_ptr` is built from raw pointer at the call site (no pre-built handles)
- `const sycl::half*` will NOT convert to non-const `multi_ptr` - cast at the load call
- In-place accumulate (`mad(sg, C, A, B, C)`) works for K-loop accumulation
- `barrier()` on sub_group is deprecated (use `sycl::group_barrier`) but still works
- One 16-lane sub_group per XMX tile; grid axis must be multiple of 16
- **XMX is a PURE MAC unit**: it only does C += A*B. No ALU ops (no shifts, masks, subtract, add).
  Dequantization MUST happen in SIMD registers before data reaches XMX via memory.
- **`joint_matrix_fill` sets ALL elements to one value** - cannot inject per-element data from SIMD
- **`joint_matrix_apply` (base)**: lambda gets `(T& x)` with NO index - uniform ops only
- **`imx::joint_matrix_apply` (Intel ext)**: lambda gets `(T& x, size_t row, size_t col)` -
  position-dependent ops possible (per-element scale lookup, etc.)
- **No SIMD-register-to-XMX-register path exists**: data MUST pass through memory (LDS minimum)

## JIT link errors

When a tile/layout combination doesn't exist, the error appears at RUNTIME (JIT compile):
```
undefined reference to `OpJointMatrixLoadINTEL_PackedB_ColumnMajor_SG16_32x64<...>'
```
There is NO runtime API to query which combinations are available. You must try and catch.
The `matrix_combinations` device info lists what the hardware supports, but the backend
(driver) may not have implemented all of them.

## Performance data (B70, Qwen3.8-27B Q6_K, MTP verify at 32K context)

| config | t/s | notes |
|--------|------|-------|
| tile FA (no XMX verify) | 35.6 | baseline, larger native tiles |
| XMX verify M=32/N=64/K=16, SPLIT=256 | 31.0 | best XMX config, -13% vs tile FA |
| XMX verify M=32/N=64/K=16, SPLIT=128 | 30.1 | more splits = more overhead, confirms flash-decoding cost |
| XMX verify M=32/N=64/K=16, SPLIT=512 | crash | DEVICE_LOST (LDS pressure or kernel timeout) |
| XMX verify M=16/N=16/K=16, SPLIT=256 | 28.7 | old, more iterations |
| XMX decode at 32K | 20.5 | vs tile FA 18.9 = +8% |

**Conclusion: XMX verify is not competitive with tile FA for MTP.** The flash-decoding approach
(split KV across WGs, combine partials) has inherent overhead (redundant Q reads, combine kernel
launch) that outweighs the XMX compute advantage for small-M verify (MACT=12-30). SPLIT tuning
confirms: smaller splits = more overhead = slower. SPLIT=512 crashes. SPLIT=256 is optimal but
still 13% behind tile FA. The decode path (M=1-8, single token) IS a net +8% win because the
per-WG overhead is amortized differently there.

SPLIT=512 crash: likely the 8 K-chunks x 16 dim-chunks = 128 barriers per WG times out, or LDS
(114KB at SPLIT=512) exceeds the per-WG allocation limit on bmg_g31.

## LDS staging pattern

```cpp
constexpr int LDS_BYTES = M*D*2 + KTILE*NTILE*2 + M*SPLIT*4 + M*SPLIT*2;
syclex::work_group_static<char[LDS_BYTES]> lsm;
sycl::half * Q16  = (sycl::half *)&lsm;           // [M][D]
sycl::half * K_T  = Q16 + M*D;                     // [KTILE][NTILE] transposed
float    * scores = (float *)(K_T + KTILE*NTILE);  // [M][SPLIT]
sycl::half * P16  = (sycl::half *)(scores + M*SPLIT); // [M][SPLIT]

// Write, then barrier before XMX reads:
for (int i = lane; i < N; i += 16) buf[i] = ...;
sg.barrier();
mx::joint_matrix_load(sg, A_jm, xmp_l_h(buf), ldm);
```

Typical LDS usage at M=32/N=64/D=256/SPLIT=256:
- Q16: 32*256*2 = 16KB
- K_T: 16*64*2 = 2KB (reused per chunk)
- scores: 32*256*4 = 32KB
- P16: 32*256*2 = 16KB
- Total: ~66KB (well within 1MB LDS)

## Gotchas

1. **FA output tensor is [D][heads][positions]**, NOT [D][positions][heads] like Q.
   `ne[1]=heads`, `ne[2]=positions`. Strides: `nb[1]/4` for head, `nb[2]/4` for pos.
   Swapping them -> garbled output while first (head=0,pos=0) looks correct.

2. **col_major B only works at N=16 on BMG.** For N=64 you MUST use row_major + LDS transpose.

3. **K-transpose coalescing:** read pattern `Ksrc[n*pos_stride + k]` with `i = k*NTILE+n`
   means lanes 0-15 read 16 consecutive halfs (coalesced). The pos_stride jump happens every
   NTILE/16 iterations. Good.

4. **NCHUNK:** with M=32, gqa=6, n_q_pos<=5: MACT<=30 fits in one tile. NCHUNK=1 always.
   With M=16, MACT>16 required NCHUNK=2 (K/V re-loaded per chunk = slower).

5. **The `barrier()` deprecation warning** is benign; the old API still works fine in 2026.1.

6. **Work-group size:** 1 sub_group of 16 lanes. Use `reqd_sub_group_size(16)`. The 3rd grid
   axis must be exactly 16 (one sub_group per work-group).

## XMX Q6_K GEMV experiment (2026-09-04)

Attempted to offload Q6_K GEMV weight dequant+dot to XMX. **Result: 2.3 t/s vs 22.5 eSIMD
baseline - a 10x LOSS. XMX is not competitive for quantized GEMV on B70.**

Architecture conflict:
- eSIMD: 1 row/WG, `block_load<uint8_t, 128>` covers entire 256-elem block in ONE instruction.
  Sequential access = perfect coalescing = DRAM bandwidth bound (506 GB/s).
- XMX: needs M rows simultaneously in a tile (minimum M=1). Even at M=1, the K-dim must be loaded
  as 1D into A, B as K x N. For Q6_K, each lane reads scattered bytes (ql[0..31], qh[0..15],
  scales[0..15]) per 32-elem chunk instead of one 128B block_load. Plus 2 LDS barriers per K-tile
  x 320 K-iterations. The memory access pattern is inherently worse.

Correct dequant IS achievable (verified: all 256 values match CPU reference). The bottleneck is
purely the memory access pattern + barrier overhead, not compute.

**LDS B-load bug (BMG-specific, non-FA kernels):** `xmp_l_h` for `joint_matrix_load(sg, B_jm, ...)`
produces wrong results in the Q6_K GEMV kernel even with correct data in LDS. The same pattern works
correctly in the FA kernel. Cause unknown (possibly LDS banking, tile-size interaction, or a driver
bug specific to non-template / differently-shaped kernels). Workaround: use `xmp_g_h` (global) for B.

**Kernel MUST be a template function.** Non-template kernels calling XMX API on BMG cause DEVICE_LOST
crashes (the same pattern that crashes non-template LDS code). The FA kernel works because it is
already a template (`template <int GQA>`).

Verdict: **Do not retry XMX for quantized GEMV on B70.** XMX's 2D-tile access model is
fundamentally incompatible with the sequential block-based layouts that make eSIMD saturate DRAM.
XMX wins where data is already in natural row-major FP16 (FA: Q/K/V are contiguous half arrays).

## Reference

- dkhaldi/sycl_joint_matrix_kernels: https://github.com/dkhaldi/sycl_joint_matrix_kernels/
  - Confirms BMG = row_major B, tM=32/tN=64/tK=16 for large tiles, ~220 TFlops
  - VNNI flag only for DG2/SPR; absent for BMG
- llama.cpp files: `ggml/src/ggml-sycl/fattn-xmx-decode.cpp` (decode + verify FA),
  `ggml/src/ggml-sycl/dmmv-q6k-xmx.cpp` (Q6_K GEMV experiment - stashed, do not retry)
