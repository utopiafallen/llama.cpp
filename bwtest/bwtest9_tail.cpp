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
