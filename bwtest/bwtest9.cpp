// Real-kernel DRAM probe: runs the PRODUCTION eSIMD DMMV kernel (copied verbatim
// from dmmv.cpp + esimd.hpp) against SoA and interleaved weight buffers.
// usage: bwtest8.exe [nrows] [ncols] [passes] [quant]  quant: 0=Q6_K 1=Q5_K 2=Q8_0
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <utility>

#define QK_K 256
#define QK8_0 32
#define K_SCALE_SIZE 12

enum ggml_type {
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
};

namespace ggml_sycl_esimd {

#ifndef GGML_SYCL_DMMV_ESIMD_WG_SIZE
constexpr int GGML_SYCL_DMMV_ESIMD_WG_SIZE = 4;
#endif

template <ggml_type T> struct esimd_reorder_q_traits;

// === verbatim from esimd.hpp (helpers) ===

template <ggml_type T> struct esimd_reorder_q_traits;

// build a 32-lane vector whose low 16 lanes are `lo` and high 16 are `hi`
// (a super-chunk splits into two 16-wide halves with distinct scale/min codes).
static ESIMD_INLINE sycl::ext::intel::esimd::simd<float, 32> splat_lo_hi(float lo, float hi) {
    using namespace sycl::ext::intel::esimd;
    simd<float, 32> v;
    v.select<16, 1>(0)  = lo;
    v.select<16, 1>(16) = hi;
    return v;
}

// unpack one block of Q4_K/Q5_K scale/min codes (get_scale_min_k4 layout) into 8
// float scales (dall * sc) and 8 float mins (-dmin * m); the min carries the
// negation so the dequant epilogue adds.
static ESIMD_INLINE void unpack_scale_min_k4(
        sycl::ext::intel::esimd::simd<uint8_t, 12> scales, float dall, float dmin,
        sycl::ext::intel::esimd::simd<float, 8> & scale_f,
        sycl::ext::intel::esimd::simd<float, 8> & min_f) {
    using namespace sycl::ext::intel::esimd;
    simd<uint8_t, 8> sc = 0;
    simd<uint8_t, 8> m  = 0;
    simd<uint8_t, 4> scale_lo = scales.select<4, 1>(0);
    simd<uint8_t, 4> min_lo   = scales.select<4, 1>(4);
    simd<uint8_t, 4> hi_bits  = scales.select<4, 1>(8);
    sc.select<4, 1>(0) = scale_lo & simd<uint8_t, 4>(0x3F);
    sc.select<4, 1>(4) = (hi_bits & simd<uint8_t, 4>(0x0F)) |
                         ((scale_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    m.select<4, 1>(0)  = min_lo & simd<uint8_t, 4>(0x3F);
    m.select<4, 1>(4)  = (hi_bits >> simd<uint8_t, 4>(4)) |
                         ((min_lo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    scale_f = convert<float>(sc) * dall;
    min_f   = convert<float>(m) * (-dmin);
}

template <> struct esimd_reorder_q_traits<GGML_TYPE_Q5_K> {
    // per-stream byte strides: SoA uses the stream widths, the interleaved
    // layout uses the 176 byte tile for all streams
    struct ptrs {
        const uint8_t *    qs;
        const uint8_t *    qh;
        const uint8_t *    scales;
        const sycl::half * dm;
        size_t             qs_stride;
        size_t             qh_stride;
        size_t             scales_stride;
        size_t             dm_stride;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb, int interleaved) {
        const uint8_t * base = (const uint8_t *) vx;
        if (interleaved) {
            const size_t tile = QK_K / 2 + QK_K / 8 + K_SCALE_SIZE + 2 * sizeof(sycl::half);
            return { base,
                     base + QK_K / 2,
                     base + QK_K / 2 + QK_K / 8,
                     (const sycl::half *) (base + QK_K / 2 + QK_K / 8 + K_SCALE_SIZE),
                     tile, tile, tile, tile / (2 * sizeof(sycl::half)) };
        }
        const uint8_t *    qs     = base;
        const uint8_t *    qh     = qs + nb * (QK_K / 2);
        const uint8_t *    scales = qh + nb * (QK_K / 8);
        const sycl::half * dm     = (const sycl::half *) (scales + nb * K_SCALE_SIZE);
        return { qs, qh, scales, dm, QK_K / 2, QK_K / 8, K_SCALE_SIZE, 2 };
    }

    // extract bit `bit` (0..7) of each lane and move it to bit position 4,
    // e.g. for the 4-bit base quant's 5th (high) bit. `bit` is always a
    // compile-time-known unrolled loop constant at call sites, so this folds
    // to a single mask (bit==4), mask+left-shift (bit<4), or mask+right-shift
    // (bit>4) instead of the shift+mask+shift a naive `(qh>>bit & 1) << 4` emits.
    static ESIMD_INLINE sycl::ext::intel::esimd::simd<uint16_t, 32> extract_bit_to_pos4(
            sycl::ext::intel::esimd::simd<uint8_t, 32> qh, int bit) {
        using namespace sycl::ext::intel::esimd;
        simd<uint16_t, 32> masked = convert<uint16_t>(qh & simd<uint8_t, 32>((uint8_t) (1u << bit)));
        if (bit < 4) {
            return masked << simd<uint16_t, 32>((uint16_t) (4 - bit));
        } else if (bit > 4) {
            return masked >> simd<uint16_t, 32>((uint16_t) (bit - 4));
        }
        return masked;
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> qs_a     = block_load<uint8_t, 128>(pa.qs + bia * pa.qs_stride);
        simd<uint8_t, 128> qs_b     = 0;
        simd<uint8_t, 32>  qh_a     = block_load<uint8_t, 32>(pa.qh + bia * pa.qh_stride);
        simd<uint8_t, 32>  qh_b     = 0;
        simd<uint8_t, 12>  scales_a = block_load<uint8_t, 12>(pa.scales + bia * pa.scales_stride);
        simd<uint8_t, 12>  scales_b = 0;

        const float dall_a = (float) pa.dm[bia * pa.dm_stride + 0];
        const float dmin_a = (float) pa.dm[bia * pa.dm_stride + 1];
        float dall_b = 0.0f;
        float dmin_b = 0.0f;
        if (has_b) {
            qs_b     = block_load<uint8_t, 128>(pb.qs + bib * pb.qs_stride);
            qh_b     = block_load<uint8_t, 32>(pb.qh + bib * pb.qh_stride);
            scales_b = block_load<uint8_t, 12>(pb.scales + bib * pb.scales_stride);
            dall_b = (float) pb.dm[bib * pb.dm_stride + 0];
            dmin_b = (float) pb.dm[bib * pb.dm_stride + 1];
        }

        simd<float, 8> scale_f_a, min_f_a, scale_f_b, min_f_b;
        unpack_scale_min_k4(scales_a, dall_a, dmin_a, scale_f_a, min_f_a);
        unpack_scale_min_k4(scales_b, dall_b, dmin_b, scale_f_b, min_f_b);

        simd<uint8_t, 128> qs_lo_a = qs_a & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_a = qs_a >> simd<uint8_t, 128>(4);
        simd<uint8_t, 128> qs_lo_b = qs_b & simd<uint8_t, 128>(0x0F);
        simd<uint8_t, 128> qs_hi_b = qs_b >> simd<uint8_t, 128>(4);

#pragma unroll
        for (int sb = 0; sb < 8; sb += 2) {
            const int q_offset = sb * 16;
            simd<float, 32> y_lo = y_vec.select<32, 1>(sb * 32);
            simd<float, 32> y_hi = y_vec.select<32, 1>((sb + 1) * 32);

            const float scale_a_lo = scale_f_a[sb];
            const float scale_a_hi = scale_f_a[sb + 1];
            const float min_a_lo   = min_f_a[sb];
            const float min_a_hi   = min_f_a[sb + 1];
            const float scale_b_lo = scale_f_b[sb];
            const float scale_b_hi = scale_f_b[sb + 1];
            const float min_b_lo   = min_f_b[sb];
            const float min_b_hi   = min_f_b[sb + 1];

            simd<uint8_t, 32> qa_lo_u8 = qs_lo_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qa_hi_u8 = qs_hi_a.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_lo_u8 = qs_lo_b.select<32, 1>(q_offset);
            simd<uint8_t, 32> qb_hi_u8 = qs_hi_b.select<32, 1>(q_offset);
            simd<uint16_t, 32> qa_lo = convert<uint16_t>(qa_lo_u8);
            simd<uint16_t, 32> qa_hi = convert<uint16_t>(qa_hi_u8);
            simd<uint16_t, 32> qb_lo = convert<uint16_t>(qb_lo_u8);
            simd<uint16_t, 32> qb_hi = convert<uint16_t>(qb_hi_u8);

            // add the 5th bit: chunk sb uses qh bit sb, chunk sb+1 uses qh bit sb+1;
            // qh always indexes the same 32 bytes regardless of chunk
            qa_lo += extract_bit_to_pos4(qh_a, sb);
            qa_hi += extract_bit_to_pos4(qh_a, sb + 1);
            qb_lo += extract_bit_to_pos4(qh_b, sb);
            qb_hi += extract_bit_to_pos4(qh_b, sb + 1);

            simd<float, 32> deq_a_lo = convert<float>(qa_lo) * scale_a_lo + min_a_lo;
            simd<float, 32> deq_a_hi = convert<float>(qa_hi) * scale_a_hi + min_a_hi;
            simd<float, 32> deq_b_lo = convert<float>(qb_lo) * scale_b_lo + min_b_lo;
            simd<float, 32> deq_b_hi = convert<float>(qb_hi) * scale_b_hi + min_b_hi;

            acc_a += y_lo * deq_a_lo;
            acc_b += y_lo * deq_b_lo;
            acc_a += y_hi * deq_a_hi;
            acc_b += y_hi * deq_b_hi;
        }
    }
};

// ---------------------------------------------------------------------------
// Q6_K, SOA reorder layout:
//   [ql: nb*(QK_K/2)] [qh: nb*(QK_K/4)] [scales(int8): nb*(QK_K/16)] [d: nb*half]
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q6_K> {
    // per-stream byte strides: SoA uses the stream widths, the interleaved
    // layout uses the 210 byte tile for all streams
    struct ptrs {
        const uint8_t *    ql;
        const uint8_t *    qh;
        const int8_t *     scales;
        const sycl::half * d;
        size_t             ql_stride;
        size_t             qh_stride;
        size_t             scales_stride;
        size_t             d_stride;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb, int interleaved) {
        const uint8_t * base = (const uint8_t *) vx;
        if (interleaved) {
            const size_t tile = QK_K / 2 + QK_K / 4 + QK_K / 16 + sizeof(sycl::half);
            return { base,
                     base + QK_K / 2,
                     (const int8_t *) (base + QK_K / 2 + QK_K / 4),
                     (const sycl::half *) (base + QK_K / 2 + QK_K / 4 + QK_K / 16),
                     tile, tile, tile, tile / sizeof(sycl::half) };
        }
        const uint8_t *    ql     = base;
        const uint8_t *    qh     = ql + nb * (QK_K / 2);
        const int8_t *     scales = (const int8_t *) (qh + nb * (QK_K / 4));
        const sycl::half * d      = (const sycl::half *) (scales + nb * (QK_K / 16));
        return { ql, qh, scales, d, QK_K / 2, QK_K / 4, QK_K / 16, 1 };
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        simd<uint8_t, 128> ql_a     = block_load<uint8_t, 128>(pa.ql + bia * pa.ql_stride);
        simd<uint8_t, 128> ql_b     = 0;
        simd<uint8_t, 64>  qh_a     = block_load<uint8_t, 64>(pa.qh + bia * pa.qh_stride);
        simd<uint8_t, 64>  qh_b     = 0;
        simd<int8_t, 16>   scales_a = block_load<int8_t, 16>(pa.scales + bia * pa.scales_stride);
        simd<int8_t, 16>   scales_b = 0;

        const float d_a = (float) pa.d[bia * pa.d_stride];
        float d_b = 0.0f;
        if (has_b) {
            ql_b     = block_load<uint8_t, 128>(pb.ql + bib * pb.ql_stride);
            qh_b     = block_load<uint8_t, 64>(pb.qh + bib * pb.qh_stride);
            scales_b = block_load<int8_t, 16>(pb.scales + bib * pb.scales_stride);
            d_b = (float) pb.d[bib * pb.d_stride];
        }

        simd<float, 16> sc_a = convert<float>(scales_a);
        simd<float, 16> sc_b = convert<float>(scales_b);

#pragma unroll
        for (int im = 0; im < 2; ++im) {
            simd<uint8_t, 32> ql_lo_a   = ql_a.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_a   = ql_a.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_a = qh_a.select<32, 1>(32 * im);
            simd<uint8_t, 32> ql_lo_b   = ql_b.select<32, 1>(64 * im);
            simd<uint8_t, 32> ql_hi_b   = ql_b.select<32, 1>(64 * im + 32);
            simd<uint8_t, 32> qh_bits_b = qh_b.select<32, 1>(32 * im);

            // reconstruct each 32-wide 6-bit group (matches dequantize_row_q6_K)
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                simd<float, 32> y_g = y_vec.select<32, 1>(32 * (4 * im + g));

                const float scale_a_lo = sc_a[8 * im + 2 * g + 0] * d_a;
                const float scale_a_hi = sc_a[8 * im + 2 * g + 1] * d_a;
                const float scale_b_lo = sc_b[8 * im + 2 * g + 0] * d_b;
                const float scale_b_hi = sc_b[8 * im + 2 * g + 1] * d_b;

                simd<float, 32> scale_vec_a = splat_lo_hi(scale_a_lo, scale_a_hi);
                simd<float, 32> scale_vec_b = splat_lo_hi(scale_b_lo, scale_b_hi);

                simd<uint8_t, 32> qa;
                simd<uint8_t, 32> qb;
                switch (g) {
                    case 0:
                        qa = (ql_lo_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        qb = (ql_lo_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x03)) << simd<uint8_t, 32>(4));
                        break;
                    case 1:
                        qa = (ql_hi_a & simd<uint8_t, 32>(0x0F)) | ((qh_bits_a & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        qb = (ql_hi_b & simd<uint8_t, 32>(0x0F)) | ((qh_bits_b & simd<uint8_t, 32>(0x0C)) << simd<uint8_t, 32>(2));
                        break;
                    case 2:
                        qa = (ql_lo_a >> simd<uint8_t, 32>(4)) | (qh_bits_a & simd<uint8_t, 32>(0x30));
                        qb = (ql_lo_b >> simd<uint8_t, 32>(4)) | (qh_bits_b & simd<uint8_t, 32>(0x30));
                        break;
                    default:
                        qa = (ql_hi_a >> simd<uint8_t, 32>(4)) | ((qh_bits_a & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        qb = (ql_hi_b >> simd<uint8_t, 32>(4)) | ((qh_bits_b & simd<uint8_t, 32>(0xC0)) >> simd<uint8_t, 32>(2));
                        break;
                }

                simd<float, 32> deq_a = (convert<float>(qa) - 32.0f) * scale_vec_a;
                simd<float, 32> deq_b = (convert<float>(qb) - 32.0f) * scale_vec_b;

                acc_a += y_g * deq_a;
                acc_b += y_g * deq_b;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Q8_0, SOA reorder layout (reorder_qw_q8_0):
//   [qs: nrows*ncols int8] [d: nrows*(ncols/QK8_0) half]
// one mac_pair call handles a 256-element chunk == QK_K/QK8_0 Q8_0 blocks
// ---------------------------------------------------------------------------
template <> struct esimd_reorder_q_traits<GGML_TYPE_Q8_0> {
    // per-stream byte strides: SoA uses the stream widths, the interleaved
    // layout uses the 272 byte tile for all streams
    struct ptrs {
        const int8_t *     qs;
        const sycl::half * d;
        size_t             qs_stride;
        size_t             d_stride;
    };

    static ESIMD_INLINE ptrs make_ptrs(const void * vx, size_t nb, int interleaved) {
        const int8_t * base = (const int8_t *) vx;
        if (interleaved) {
            const size_t tile = QK_K + (QK_K / QK8_0) * sizeof(sycl::half);
            return { base, (const sycl::half *) (base + QK_K), tile, tile / sizeof(sycl::half) };
        }
        const int8_t *     qs = base;
        const sycl::half * d  = (const sycl::half *) (qs + nb * QK_K);
        return { qs, d, QK_K, QK_K / QK8_0 };
    }

    static ESIMD_INLINE void mac_pair(
            const ptrs & pa, size_t bia,
            const ptrs & pb, size_t bib, bool has_b,
            sycl::ext::intel::esimd::simd<float, 256> & y_vec,
            sycl::ext::intel::esimd::simd<float, 32> & acc_a,
            sycl::ext::intel::esimd::simd<float, 32> & acc_b) {
        using namespace sycl::ext::intel::esimd;

        constexpr int NSUB = QK_K / QK8_0; // 8 sub-blocks of 32

        simd<int8_t, 128> qs_a_lo = block_load<int8_t, 128>(pa.qs + bia * pa.qs_stride);
        simd<int8_t, 128> qs_a_hi = block_load<int8_t, 128>(pa.qs + bia * pa.qs_stride + 128);
        simd<int8_t, 128> qs_b_lo = 0;
        simd<int8_t, 128> qs_b_hi = 0;
        if (has_b) {
            qs_b_lo = block_load<int8_t, 128>(pb.qs + bib * pb.qs_stride);
            qs_b_hi = block_load<int8_t, 128>(pb.qs + bib * pb.qs_stride + 128);
        }

        // sub-blocks 0..NSUB/2-1 live in the low 128 int8, the rest in the high 128
#pragma unroll
        for (int s = 0; s < NSUB / 2; ++s) {
            const float d_a = (float) pa.d[bia * pa.d_stride + s];
            const float d_b = has_b ? (float) pb.d[bib * pb.d_stride + s] : 0.0f;

            simd<float, 32> y_s = y_vec.select<32, 1>(32 * s);
            simd<int8_t, 32> qa = qs_a_lo.select<32, 1>(32 * s);
            simd<int8_t, 32> qb = qs_b_lo.select<32, 1>(32 * s);
            simd<float, 32> deq_a = convert<float>(qa) * d_a;
            simd<float, 32> deq_b = convert<float>(qb) * d_b;
            acc_a += y_s * deq_a;
            acc_b += y_s * deq_b;
        }
#pragma unroll
        for (int s = NSUB / 2; s < NSUB; ++s) {
            const int     sl  = s - NSUB / 2;
            const float   d_a = (float) pa.d[bia * pa.d_stride + s];
            const float   d_b = has_b ? (float) pb.d[bib * pb.d_stride + s] : 0.0f;

            simd<float, 32> y_s = y_vec.select<32, 1>(32 * s);
            simd<int8_t, 32> qa = qs_a_hi.select<32, 1>(32 * sl);
            simd<int8_t, 32> qb = qs_b_hi.select<32, 1>(32 * sl);
            simd<float, 32> deq_a = convert<float>(qa) * d_a;
            simd<float, 32> deq_b = convert<float>(qb) * d_b;
            acc_a += y_s * deq_a;
            acc_b += y_s * deq_b;
        }
    }
};

} // namespace ggml_sycl_esimd
#ifndef GGML_SYCL_DMMV_ESIMD_WG_SIZE
using ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_SIZE;
#endif

// generic reordered dequantize-matvec: each work-group owns a pair of
// consecutive output rows and updates one 32-wide accumulator per row
template <ggml_type T, bool ADD_RES>
ESIMD_INLINE void dequantize_mul_mat_vec_reorder_esimd(
        const void * vx, const float * y, const float * res, float * dst,
        const int ncols, const int nrows,
        sycl::local_accessor<float, 1> lmem,
        const sycl::nd_item<1> & it,
        int interleaved) {
    using namespace sycl::ext::intel::esimd;
    using traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    const int    num_blocks_per_row = ncols / QK_K;
    const size_t nb = (size_t) nrows * num_blocks_per_row;
    const auto   ps = traits::make_ptrs(vx, nb, interleaved);

    const int  tid      = it.get_local_id(0);
    const int  row_pair = it.get_group(0);
    const int  row0     = row_pair * 2; // two consecutive output rows
    const bool has_row1 = row0 + 1 < nrows;

    // one 32-wide accumulator per output row (small footprint, no spill)
    simd<float, 32> acc0 = 0.0f;
    simd<float, 32> acc1 = 0.0f;

    for (int ib = tid; ib < num_blocks_per_row; ib += GGML_SYCL_DMMV_ESIMD_WG_SIZE) {
        simd<float, 256> y_vec = block_load<float, 256>(y + (size_t) ib * QK_K);

        const size_t bi0 = (size_t) (row0 + 0) * num_blocks_per_row + ib;
        const size_t bi1 = (size_t) (row0 + 1) * num_blocks_per_row + ib;

        traits::mac_pair(ps, bi0, ps, bi1, has_row1, y_vec, acc0, acc1);
    }

    lmem[tid * 2 + 0] = reduce<float>(acc0, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<float>(acc1, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_SIZE; ++p) {
            sum0 += lmem[p * 2 + 0];
            sum1 += lmem[p * 2 + 1];
        }
        if constexpr (ADD_RES) {
            dst[row0 + 0] = sum0 + res[row0 + 0];
            if (has_row1) {
                dst[row0 + 1] = sum1 + res[row0 + 1];
            }
        } else {
            dst[row0 + 0] = sum0;
            if (has_row1) {
                dst[row0 + 1] = sum1;
            }
        }
    }
}
#include <vector>
#include <algorithm>

static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static void fill_bytes(sycl::queue & q, uint8_t * base, size_t n, uint32_t seed) {
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(n / 4),
                       [=](sycl::id<1> i) {
                           ((uint32_t *) base)[i[0]] =
                               hash32((uint32_t) ((size_t) i[0] * 2654435761ull) + seed);
                       });
    });
}

// wall-clock per kernel, layouts alternate every pass (S,I,S,I,...) so both see
// the same thermal/clock state; the CPU submit+wait overhead is calibrated out
// with no-op kernels. each pass reads a DIFFERENT window of the weight buffer
// (8 windows, rotated), so no L2 state survives between passes: every measured
// byte comes from DRAM
template <ggml_type T, size_t TILE>
static void run_rr(sycl::queue & q, uint8_t * w, float * y, float * dst,
                   int NROWS, int NCOLS, int NP, const char * name) {
    const size_t NB  = (size_t) NROWS * (NCOLS / 256);
    const size_t WGS = ((size_t) NROWS + 1) / 2;
    const size_t WIN = NB * TILE;

    auto submit_one = [&](int ilv, uint8_t * base) {
        q.submit([&](sycl::handler & h) {
            sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(WGS * GGML_SYCL_DMMV_ESIMD_WG_SIZE),
                                  sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
                [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    dequantize_mul_mat_vec_reorder_esimd<T, /*ADD_RES=*/false>(
                        base, y, nullptr, dst, NCOLS, NROWS, lmem, it, ilv);
                });
        });
        q.wait();
    };

    const int NCAL = 50;
    auto c0 = std::chrono::steady_clock::now();
    for (int i = 0; i < NCAL; ++i) { q.submit([](sycl::handler &) {}); q.wait(); }
    auto c1 = std::chrono::steady_clock::now();
    const double c_ns = std::chrono::duration<double, std::nano>(c1 - c0).count() / NCAL;

    std::vector<double> t_soa, t_ilv;
    for (int r = 0; r < NP; ++r) {
        const int ilv = r % 2;
        uint8_t * base = w + (r % 8) * WIN;
        auto t0 = std::chrono::steady_clock::now();
        submit_one(ilv, base);
        auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() - c_ns;
        if (ilv) t_ilv.push_back(ns); else t_soa.push_back(ns);
    }
    std::sort(t_soa.begin(), t_soa.end());
    std::sort(t_ilv.begin(), t_ilv.end());
    const double ms_soa = t_soa[t_soa.size() / 2] / 1e6;
    const double ms_ilv = t_ilv[t_ilv.size() / 2] / 1e6;
    const double gb     = (double) WIN / 1e9;
    printf("  %-10s %6.2f GB  SoA %9.2f ms -> %8.1f GB/s | interleaved %9.2f ms -> %8.1f GB/s | ratio %+.1f%% (cal %.0f us)\n",
           name, gb, ms_soa, gb * 1000.0 / ms_soa, ms_ilv, gb * 1000.0 / ms_ilv,
           100.0 * (ms_soa / ms_ilv - 1.0), c_ns / 1000.0);
}

template <ggml_type T>
static void warm(sycl::queue & q, uint8_t * w, float * y, float * dst, int NROWS, int NCOLS, size_t WIN) {
    const size_t WGS = ((size_t) NROWS + 1) / 2;
    for (int ilv = 0; ilv <= 1; ++ilv) {
        q.submit([&](sycl::handler & h) {
            sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(WGS * GGML_SYCL_DMMV_ESIMD_WG_SIZE),
                                  sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
                [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    dequantize_mul_mat_vec_reorder_esimd<T, /*ADD_RES=*/false>(
                        w, y, nullptr, dst, NCOLS, NROWS, lmem, it, ilv);
                });
        });
    }
    q.wait();
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    sycl::device dev(sycl::gpu_selector_v);
    sycl::queue q(dev, sycl::property::queue::in_order());
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int NROWS = 5120, NCOLS = 17408, NP = 16, QUANT = 0;
    if (argc >= 5) {
        NROWS = atoi(argv[1]);
        NCOLS = atoi(argv[2]);
        NP    = atoi(argv[3]);
        QUANT = atoi(argv[4]);
    }
    printf("probe: nrows=%d ncols=%d (bpr=%d) passes=%d quant=%s\n",
           NROWS, NCOLS, NCOLS / 256, NP,
           QUANT == 0 ? "Q6_K" : QUANT == 1 ? "Q5_K" : "Q8_0");

    const size_t NB = (size_t) NROWS * (NCOLS / 256);

    size_t tile = 0;
    if (QUANT == 0)      tile = QK_K/2 + QK_K/4 + QK_K/16 + 2;   // 210
    else if (QUANT == 1) tile = QK_K/2 + QK_K/8 + K_SCALE_SIZE + 4; // 176
    else                 tile = QK_K + (QK_K/QK8_0) * 2;          // 272
    const size_t WIN  = NB * tile;
    const size_t NWIN = (NP >= 8) ? 8 : (size_t) NP;
    printf("tile: %zu bytes/block, window: %.2f GB, buffer: %.2f GB (%zu windows)\n",
           tile, (double) WIN / 1e9, (double) WIN * NWIN / 1e9, NWIN);

    auto w   = sycl::malloc_device<uint8_t>(WIN * NWIN, q);
    auto y   = sycl::malloc_device<float>(NCOLS, q);
    auto dst = sycl::malloc_device<float>(NROWS, q);

    fill_bytes(q, w, WIN * NWIN, 0x165667b1u);
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(NCOLS),
                       [=](sycl::id<1> i) {
                           y[i[0]] = ((float) (hash32((uint32_t) i[0] + 0x1234abcd) & 0xFFFF) / 65536.0f) - 0.5f;
                       });
    }).wait();

    if (QUANT == 0) {
        warm<GGML_TYPE_Q6_K>(q, w, y, dst, NROWS, NCOLS, WIN);
        run_rr<GGML_TYPE_Q6_K, 210>(q, w, y, dst, NROWS, NCOLS, NP, "Q6_K");
    } else if (QUANT == 1) {
        warm<GGML_TYPE_Q5_K>(q, w, y, dst, NROWS, NCOLS, WIN);
        run_rr<GGML_TYPE_Q5_K, 176>(q, w, y, dst, NROWS, NCOLS, NP, "Q5_K");
    } else {
        warm<GGML_TYPE_Q8_0>(q, w, y, dst, NROWS, NCOLS, WIN);
        run_rr<GGML_TYPE_Q8_0, 272>(q, w, y, dst, NROWS, NCOLS, NP, "Q8_0");
    }

    sycl::free(dst, q);
    sycl::free(y, q);
    sycl::free(w, q);
    return 0;
}
