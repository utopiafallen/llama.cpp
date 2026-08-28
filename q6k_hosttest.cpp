// Host-side verification of the Q6_K llm-scaler reference-layout repack + dequant.
// Round trip: random blocks -> repack (exact copy of reorder_qw_q6_k_llmscaler math)
// -> scalar port of the kernel's 512-tile interpretation vs the GGML ground truth.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

#define QK_K 256

struct block_q6_K {
    float  d;
    int8_t scales[16];
    uint8_t ql[128];
    uint8_t qh[64];
};

// Reference dequant (scalar port of ggml dequantize_block_q6_K, QK_K==256) -- ground truth
static void dequant_ref(const block_q6_K & x, float * y) {
    const float d = x.d;
    for (int ip = 0; ip < 2; ++ip) {
        for (int il = 0; il < 32; ++il) {
            const int is = 8*ip + il/16;
            const uint8_t * qlp = x.ql + 64*ip + il;
            const uint8_t   qh  = x.qh[32*ip + il];
            const int8_t  * scp = x.scales + is;
            y[128*ip + il +  0] = d * scp[0] * ((int8_t)((qlp[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
            y[128*ip + il + 32] = d * scp[2] * ((int8_t)((qlp[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
            y[128*ip + il + 64] = d * scp[4] * ((int8_t)((qlp[ 0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
            y[128*ip + il + 96] = d * scp[6] * ((int8_t)((qlp[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
        }
    }
}

// EXACT copy of the tile repack math from reorder_qw_q6_k_llmscaler (base gate)
static void repack_tile(const block_q6_K * blks, uint8_t * out_ql, uint8_t * out_qh,
                        int8_t * out_sc, float * out_d) {
    for (int B = 0; B < QK_K; ++B) {
        uint8_t v = 0;
        for (int f = 0; f < 2; ++f) {
            const int e    = 2*B + f;
            const int ob   = e / QK_K;
            const int oe   = e % QK_K;
            const int ig   = (oe % 128) / 32;
            const int il   = oe % 32;
            const int byte = 64*(oe/128) + il + 32*(ig & 1);
            const uint8_t b = blks[ob].ql[byte];
            const uint8_t val = ((oe % 128) >= 64) ? (b >> 4) : (b & 0x0F);
            v |= (uint8_t) (val << (4*f));
        }
        out_ql[B] = v;
    }
    for (int t2 = 0; t2 < QK_K/2; ++t2) {
        uint8_t v = 0;
        for (int p = 0; p < 4; ++p) {
            const int e    = p*(QK_K/2) + t2;
            const int ob   = e / QK_K;
            const int oe   = e % QK_K;
            const int byte = 32*(oe/128) + (oe%32);
            const int fld  = 2*((oe%128)/32);
            const uint8_t bits = (blks[ob].qh[byte] >> fld) & 3;
            v |= (uint8_t) (bits << (2*p));
        }
        out_qh[t2] = v;
    }
    for (int g = 0; g < QK_K/16; ++g) {
        out_sc[g]          = blks[0].scales[g];
        out_sc[16 + g]     = blks[1].scales[g];
    }
    out_d[0] = blks[0].d;
    out_d[1] = blks[1].d;
}

// Scalar port of the kernel's 512-tile interpretation (base gate), per tile element E.
static float dequant_tile_elem(const uint8_t * out_ql, const uint8_t * out_qh,
                               const int8_t * out_sc, const float * out_d, int E) {
    const int B  = E / 2;
    const int f  = E % 2;
    const uint8_t b = out_ql[B];
    const int qlv = f ? (b >> 4) : (b & 0x0F);
    const int p  = E / 128;
    const int t2 = E % 128;
    const int qhv = (out_qh[t2] >> (2*p)) & 3;
    const int sb = E / 16;
    const float s = (float) out_sc[sb] * out_d[sb / 16];
    return ((float)(qlv | (qhv << 4)) - 32.0f) * s;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> u8(0, 255);
    std::uniform_int_distribution<int> i8(-100, 100);

    int   nbad   = 0;
    float maxdiff = 0.0f;
    const int NTEST = 2000;
    for (int it = 0; it < NTEST; ++it) {
        block_q6_K blks[2];
        for (int ob = 0; ob < 2; ++ob) {
            for (int j = 0; j < 128; ++j) blks[ob].ql[j] = (uint8_t) u8(rng);
            for (int j = 0; j < 64;  ++j) blks[ob].qh[j] = (uint8_t) u8(rng);
            for (int g = 0; g < 16;  ++g) blks[ob].scales[g] = (int8_t) i8(rng);
            blks[ob].d = 0.05f + 0.01f * (it % 50) * (ob ? -1.0f : 1.0f);
        }

        float y_ref[512];
        dequant_ref(blks[0], y_ref);
        dequant_ref(blks[1], y_ref + 256);

        uint8_t r_ql[256], r_qh[128];
        int8_t  r_sc[32];
        float   r_d[2];
        repack_tile(blks, r_ql, r_qh, r_sc, r_d);

        for (int E = 0; E < 512; ++E) {
            const float w_new = dequant_tile_elem(r_ql, r_qh, r_sc, r_d, E);
            float diff = fabsf(y_ref[E] - w_new);
            if (diff > maxdiff) maxdiff = diff;
            if (diff > 1e-3f) {
                if (nbad < 12) printf("MISMATCH[tile=%d elem=%d]: ref=%.6f new=%.6f\n", it, E, y_ref[E], w_new);
                nbad++;
            }
        }
    }
    if (nbad == 0) printf("OK: all %d*512 elements match (maxdiff=%.3g)\n", NTEST, maxdiff);
    else           printf("FAIL: %d/%d elements mismatch (maxdiff=%.3g)\n", nbad, NTEST*512, maxdiff);
    if (nbad) return 1;

    // phase 2: multi-row buffer with the exact get_rows scalar offset logic
    // (transcribed from k_get_rows_q6_K_llmscaler / q6k_llmscaler_dequant_elem)
    const int R = 7;                 // rows
    const int T = 13;                // tiles per row (odd, to catch stride bugs)
    const int K = T * 512;           // elements per row
    const size_t ntile = T, nrows = R;
    const size_t ql_sz = nrows * ntile * QK_K;
    const size_t qh_sz = nrows * ntile * (QK_K/2);
    const size_t sc_sz = nrows * ntile * (QK_K/8);
    const size_t d_sz  = nrows * ntile * 2 * sizeof(float);
    std::vector<uint8_t> buf(ql_sz + qh_sz + sc_sz + d_sz);
    uint8_t * g_ql = buf.data();
    uint8_t * g_qh = g_ql + ql_sz;
    int8_t  * g_sc = (int8_t *) (g_qh + qh_sz);
    float   * g_d  = (float *) (g_sc + sc_sz);

    std::vector<block_q6_K> blocks(nrows * 2 * ntile);
    for (auto & b : blocks) {
        for (int j = 0; j < 128; ++j) b.ql[j] = (uint8_t) u8(rng);
        for (int j = 0; j < 64;  ++j) b.qh[j] = (uint8_t) u8(rng);
        for (int g = 0; g < 16;  ++g) b.scales[g] = (int8_t) i8(rng);
        b.d = 0.01f + 0.001f * (u8(rng) % 997);
    }
    for (int r = 0; r < R; ++r) {
        for (int t = 0; t < T; ++t) {
            const block_q6_K * blks = &blocks[2 * (r * T + t)];
            repack_tile(blks,
                        g_ql + (size_t)(r * T + t) * QK_K,
                        g_qh + (size_t)(r * T + t) * (QK_K/2),
                        g_sc + (size_t)(r * T + t) * (QK_K/8),
                        g_d  + (size_t)(r * T + t) * 2);
        }
    }

    // exact transcription of the get_rows helper (base gate)
    auto dequant_elem_getrows = [&](int r, int e) {
        const int     E = e % 512;
        const size_t  t = (size_t) r * ntile + (size_t) (e / 512);
        const uint8_t qb  = g_ql[t * (QK_K) + E/2];
        const int     qlv = (E & 1) ? (qb >> 4) : (qb & 0x0F);
        const uint8_t qhb = g_qh[t * (QK_K/2) + (E % 128)];
        const int     qhv = (qhb >> (2 * (E / 128))) & 3;
        const int     sb  = E / 16;
        const float   s   = (float) g_sc[t * (QK_K/8) + sb] * g_d[t * 2 + sb / 16];
        return ((float) (qlv | (qhv << 4)) - 32.0f) * s;
    };

    int   nbad2 = 0;
    float maxdiff2 = 0.0f;
    int tilebad[R][T] = {};
    for (int r = 0; r < R; ++r) {
        for (int e = 0; e < K; ++e) {
            const float yref = [&] {
                static float y[256];
                dequant_ref(blocks[2 * (r * T + e / 512) + (e % 512) / 256], y);
                return y[e % 256];
            }();
            const float w = dequant_elem_getrows(r, e);
            const float diff = fabsf(yref - w);
            if (diff > maxdiff2) maxdiff2 = diff;
            if (diff > 1e-3f) {
                if (nbad2 < 30) printf("MISMATCH2[row=%d tile=%d E=%d sb=%d]: ref=%.6f new=%.6f\n",
                                        r, e / 512, e % 512, (e % 512) / 16, yref, w);
                tilebad[r][e / 512]++;
                nbad2++;
            }
        }
    }
    printf("tile fail counts (row,tile: n):\n");
    for (int r = 0; r < R; ++r) {
        for (int t = 0; t < T; ++t) {
            if (tilebad[r][t]) printf("  r%d t%d: %d\n", r, t, tilebad[r][t]);
        }
    }
    if (nbad2 == 0) printf("OK2: all %d*%d get_rows elements match (maxdiff=%.3g)\n", R, K, maxdiff2);
    else            printf("FAIL2: %d/%d get_rows elements mismatch (maxdiff=%.3g)\n", nbad2, R*K, maxdiff2);
    return nbad2 ? 1 : 0;
}
