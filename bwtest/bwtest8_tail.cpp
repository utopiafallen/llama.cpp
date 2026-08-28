
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

template <ggml_type T, size_t TILE>
static void run_layout(sycl::queue & q, uint8_t * w, float * y, float * dst,
                       int NROWS, int NCOLS, int NP, int interleaved, const char * name) {
    const int    BPR = NCOLS / 256;
    const size_t NB  = (size_t) NROWS * BPR;
    const size_t WGS = ((size_t) NROWS + 1) / 2;

    auto t0 = std::chrono::steady_clock::now();
    for (int p = 0; p < NP; ++p) {
        q.submit([&](sycl::handler & h) {
            sycl::local_accessor<float, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(WGS * GGML_SYCL_DMMV_ESIMD_WG_SIZE),
                                  sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
                [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    dequantize_mul_mat_vec_reorder_esimd<T, /*ADD_RES=*/false>(
                        w, y, nullptr, dst, NCOLS, NROWS, lmem, it, interleaved);
                });
        });
    }
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gb = (double) NB * (double) TILE * (double) NP / 1e9;
    printf("  %-10s %6.2f GB in %9.1f ms -> %8.1f GB/s\n", name, gb, ms, gb * 1000.0 / ms);
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    sycl::device dev(sycl::gpu_selector_v);
    sycl::queue q(dev);
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int NROWS = 5120, NCOLS = 17408, NP = 40, QUANT = 0;
    if (argc >= 5) {
        NROWS = atoi(argv[1]);
        NCOLS = atoi(argv[2]);
        NP    = atoi(argv[3]);
        QUANT = atoi(argv[4]);
    }
    printf("probe: nrows=%d ncols=%d (bpr=%d) passes=%d quant=%s\n",
           NROWS, NCOLS, NCOLS / 256, NP,
           QUANT == 0 ? "Q6_K" : QUANT == 1 ? "Q5_K" : "Q8_0");

    const int    BPR = NCOLS / 256;
    const size_t NB  = (size_t) NROWS * BPR;

    size_t tile = 0;
    if (QUANT == 0)      tile = QK_K/2 + QK_K/4 + QK_K/16 + 2;   // 210
    else if (QUANT == 1) tile = QK_K/2 + QK_K/8 + K_SCALE_SIZE + 4; // 176
    else                 tile = QK_K + (QK_K/QK8_0) * 2;          // 272
    printf("tile: %zu bytes/block, weights: %.2f GB\n", tile, (double) NB * tile / 1e9);

    auto w   = sycl::malloc_device<uint8_t>(NB * tile, q);
    auto y   = sycl::malloc_device<float>(NCOLS, q);
    auto dst = sycl::malloc_device<float>(NROWS, q);

    fill_bytes(q, w, NB * tile, 0x165667b1u);
    q.submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(NCOLS),
                       [=](sycl::id<1> i) {
                           y[i[0]] = ((float) (hash32((uint32_t) i[0] + 0x1234abcd) & 0xFFFF) / 65536.0f) - 0.5f;
                       });
    }).wait();

    if (QUANT == 0) {
        run_layout<GGML_TYPE_Q6_K, 210>(q, w, y, dst, NROWS, NCOLS, NP, 0, "SoA");
        run_layout<GGML_TYPE_Q6_K, 210>(q, w, y, dst, NROWS, NCOLS, NP, 1, "interleaved");
    } else if (QUANT == 1) {
        run_layout<GGML_TYPE_Q5_K, 176>(q, w, y, dst, NROWS, NCOLS, NP, 0, "SoA");
        run_layout<GGML_TYPE_Q5_K, 176>(q, w, y, dst, NROWS, NCOLS, NP, 1, "interleaved");
    } else {
        run_layout<GGML_TYPE_Q8_0, 272>(q, w, y, dst, NROWS, NCOLS, NP, 0, "SoA");
        run_layout<GGML_TYPE_Q8_0, 272>(q, w, y, dst, NROWS, NCOLS, NP, 1, "interleaved");
    }

    sycl::free(dst, q);
    sycl::free(y, q);
    sycl::free(w, q);
    return 0;
}
