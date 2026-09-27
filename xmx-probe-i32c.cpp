// xmx-probe-i32c.cpp - i32 acc as COPY operand, loaded from global (no mad)
// values m*16+n (0..127), exact in f32/i32. Shape [8][16].
// K9a: i32 acc load(global ldm16) -> store i32(global ldm16)   baseline
// K9b: i32 acc load -> COPY -> f32 acc -> store f32(ldm32)   the requested test
// K9c: f32 acc load -> COPY -> i32 acc -> store i32(ldm16)   reverse direction
// K9d: f32 acc load -> COPY f32->f32 -> store f32            control (= K8)
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <string>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;
using namespace sycl;
using sub_group = sycl::sub_group;

using mptr_g_i32 = sycl::multi_ptr<int32_t, sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
using mptr_g_i8  = sycl::multi_ptr<int8_t,   sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
using mptr_g_f   = sycl::multi_ptr<float,   sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
static mptr_g_i32 xmp_g_i32(int32_t * p) { return mptr_g_i32(p); }
static mptr_g_i8   xmp_g_i8(int8_t * p)    { return mptr_g_i8(p); }
using mptr_g_h = sycl::multi_ptr<sycl::half, sycl::access::address_space::global_space, sycl::access::decorated::legacy>;
static mptr_g_h xmp_g_h(sycl::half * p) { return mptr_g_h(p); }
static mptr_g_f   xmp_g_f(float * p)     { return mptr_g_f(p); }

int main() {
    queue q(gpu_selector{});
    const std::string devname = q.get_device().get_info<sycl::info::device::name>();
    printf("device: %s\n", devname.c_str());

    const int M8 = 8, N16 = 16;
    int8_t  * A8_d  = malloc_device<int8_t>(8 * 32, q);
    int8_t  * B8_d  = malloc_device<int8_t>(32 * 16, q);
    int32_t * C_e   = malloc_device<int32_t>(16 * 32, q);
    std::vector<int8_t> hA(8 * 32), hB(32 * 16);
    for (int m = 0; m < M8; m++) for (int k = 0; k < 32; k++) hA[m * 32 + k] = (int8_t)((m % 7) - 3);
    for (int n = 0; n < N16; n++) for (int k = 0; k < 32; k++) hB[k * 16 + n] = (int8_t)((n % 5) - 2);
    int32_t * G_i32 = malloc_device<int32_t>(M8 * 16, q);
    float   * G_f   = malloc_device<float>(M8 * 16, q);
    int32_t * C_a   = malloc_device<int32_t>(16 * 32, q);
    float   * C_b   = malloc_device<float>(16 * 32, q);
    int32_t * C_c   = malloc_device<int32_t>(16 * 32, q);
    float   * C_d   = malloc_device<float>(16 * 32, q);
    {
        std::vector<int32_t> gi(M8 * 16);
        std::vector<float>   gf(M8 * 16);
        for (int m = 0; m < M8; m++) for (int n = 0; n < N16; n++) {
            gi[m * N16 + n] = m * N16 + n;
            gf[m * N16 + n] = (float)(m * N16 + n);
        }
        q.memcpy(G_i32, gi.data(), gi.size() * 4).wait();
        q.memcpy(G_f, gf.data(), gf.size() * 4).wait();
        for (auto * b : { (void *) C_a, (void *) C_b, (void *) C_c, (void *) C_d, (void *) C_e })
            q.fill(b, 0xFF, b ? 16 * 32 * 4 : 0).wait();
        q.memcpy(A8_d, hA.data(), hA.size()).wait();
        q.memcpy(B8_d, hB.data(), hB.size()).wait();
    }

    // K9a: i32 acc load + i32 store (baseline)
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> X;
            mx::joint_matrix_load(sg, X, xmp_g_i32(G_i32), 16, mx::layout::row_major);
            mx::joint_matrix_store(sg, X, xmp_g_i32(C_a), 16, mx::layout::row_major);
        });
    });
    {
        q.wait_and_throw();
        std::vector<int32_t> hC(16 * 32, -1);
        q.memcpy(hC.data(), C_a, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < M8; m++) for (int n = 0; n < N16; n++)
            if (hC[m * 16 + n] != m * N16 + n) bad++;
        printf("K9a i32acc load+store: bad=%d/128 %s\n", bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            for (int m = 0; m < M8; m++) {
                printf("row%d dev:", m);
                for (int n = 0; n < N16; n++) printf(" %5d", hC[m * 16 + n]);
                printf("\nrow%d ref:", m);
                for (int n = 0; n < N16; n++) printf(" %5d", m * N16 + n);
                printf("\n");
            }
        }
    }

    // K9b: i32 acc load -> copy -> f32 acc -> f32 store (requested test)
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> X;
            mx::joint_matrix<sub_group, float,   mx::use::accumulator, 8, 16> Y;
            mx::joint_matrix_load(sg, X, xmp_g_i32(G_i32), 16, mx::layout::row_major);
            mx::joint_matrix_copy(sg, X, Y);
            mx::joint_matrix_store(sg, Y, xmp_g_f(C_b), 32, mx::layout::row_major);
        });
    });
    {
        q.wait_and_throw();
        std::vector<float> hC(16 * 32, -1.0f);
        q.memcpy(hC.data(), C_b, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < M8; m++) for (int n = 0; n < N16; n++)
            if (hC[m * 32 + n] != (float)(m * N16 + n)) bad++;
        printf("K9b i32acc load->copy->f32 store: bad=%d/128 %s\n", bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            printf("row0 dev:");
            for (int n = 0; n < N16; n++) printf(" %7.1f", hC[n]);
            printf("\nrow0 ref:");
            for (int n = 0; n < N16; n++) printf(" %7.0f", (float) n);
            printf("\nrow1 dev:");
            for (int n = 0; n < N16; n++) printf(" %7.1f", hC[32 + n]);
            printf("\nrow1 ref:");
            for (int n = 0; n < N16; n++) printf(" %7.0f", (float)(16 + n));
            printf("\n");
        }
    }

    // K9c: f32 acc load -> copy -> i32 acc -> i32 store (reverse)
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            mx::joint_matrix<sub_group, float,   mx::use::accumulator, 8, 16> X;
            mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> Y;
            mx::joint_matrix_load(sg, X, xmp_g_f(G_f), 16, mx::layout::row_major);
            mx::joint_matrix_copy(sg, X, Y);
            mx::joint_matrix_store(sg, Y, xmp_g_i32(C_c), 16, mx::layout::row_major);
        });
    });
    {
        q.wait_and_throw();
        std::vector<int32_t> hC(16 * 32, -1);
        q.memcpy(hC.data(), C_c, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < M8; m++) for (int n = 0; n < N16; n++)
            if (hC[m * 16 + n] != m * N16 + n) bad++;
        printf("K9c f32acc load->copy->i32 store: bad=%d/128 %s\n", bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            printf("row0 dev:");
            for (int n = 0; n < N16; n++) printf(" %5d", hC[n]);
            printf("\nrow0 ref:");
            for (int n = 0; n < N16; n++) printf(" %5d", n);
            printf("\n");
        }
    }

    // K9d: f32 acc load -> copy f32->f32 -> f32 store (control)
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            mx::joint_matrix<sub_group, float, mx::use::accumulator, 8, 16> X;
            mx::joint_matrix<sub_group, float, mx::use::accumulator, 8, 16> Y;
            mx::joint_matrix_load(sg, X, xmp_g_f(G_f), 16, mx::layout::row_major);
            mx::joint_matrix_copy(sg, X, Y);
            mx::joint_matrix_store(sg, Y, xmp_g_f(C_d), 32, mx::layout::row_major);
        });
    });
    {
        q.wait_and_throw();
        std::vector<float> hC(16 * 32, -1.0f);
        q.memcpy(hC.data(), C_d, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < M8; m++) for (int n = 0; n < N16; n++)
            if (hC[m * 32 + n] != (float)(m * N16 + n)) bad++;
        printf("K9d f32acc load->copy->f32 store: bad=%d/128 %s\n", bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            printf("row0 dev:");
            for (int n = 0; n < N16; n++) printf(" %7.1f", hC[n]);
            printf("\nrow0 ref:");
            for (int n = 0; n < N16; n++) printf(" %7.0f", (float) n);
            printf("\n");
        }
    }


    // K9e: i32 acc: mad (K1's A/B) -> i32 store ldm16 (no copy at all)
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            mx::joint_matrix<sub_group, int8_t,  mx::use::a, 8, 32, mx::layout::row_major> A8;
            mx::joint_matrix<sub_group, int8_t,  mx::use::b, 32, 16> B8;
            mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> C8;
            mx::joint_matrix_fill(sg, C8, 0);
            mx::joint_matrix_load(sg, A8, xmp_g_i8(A8_d), 32);
            mx::joint_matrix_load(sg, B8, xmp_g_i8(B8_d), 16);
            mx::joint_matrix_mad(sg, C8, A8, B8, C8);
            mx::joint_matrix_store(sg, C8, xmp_g_i32(C_e), 16, mx::layout::row_major);
        });
    });
    {
        q.wait_and_throw();
        std::vector<int32_t> hC(16 * 32, -1);
        q.memcpy(hC.data(), C_e, hC.size() * 4).wait();
        int bad = 0;
        (void)0;
        for (int m = 0; m < M8; m++) for (int n = 0; n < N16; n++) {
            int s2 = 0;
            for (int k = 0; k < 32; k++) s2 += (int) hA[m * 32 + k] * (int) hB[k * 16 + n];
            if (hC[m * 16 + n] != s2) bad++;
        }
        printf("K9e mad->i32acc->i32store: bad=%d/128 %s\n", bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            printf("row0 dev:");
            for (int n = 0; n < N16; n++) printf(" %6d", hC[n]);
            printf("\nrow0 ref:");
            for (int n = 0; n < N16; n++) {
                int s2 = 0;
                for (int k = 0; k < 32; k++) s2 += (int) hA[k] * (int) hB[k * 16 + n];
                printf(" %6d", s2);
            }
            printf("\n");
        }
    }


    // K9f: full int8 mad chain, identity A: A[m][k]=(k==m), B[k][n]=((k%5)-2)+n -> C[m][n]=B[m][n]
    //      store f32 (ldm32, exonerated) so the readback is confound-free
    {
        int8_t * Af_d = malloc_device<int8_t>(8 * 32, q);
        int8_t * Bf_d = malloc_device<int8_t>(32 * 16, q);
        float  * Cf_d = malloc_device<float>(16 * 32, q);
        std::vector<int8_t> hAf(8 * 32), hBf(32 * 16);
        for (int m = 0; m < M8; m++) for (int k = 0; k < 32; k++) hAf[m * 32 + k] = (k == m) ? 1 : 0;
        for (int k = 0; k < 32; k++) for (int n = 0; n < N16; n++) hBf[k * 16 + n] = (int8_t)(((k % 5) - 2) + n);
        q.memcpy(Af_d, hAf.data(), hAf.size()).wait();
        q.memcpy(Bf_d, hBf.data(), hBf.size()).wait();
        q.fill(Cf_d, -1.0f, 16 * 32 * 4).wait();
        q.submit([&](handler & h) {
            h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
                [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                mx::joint_matrix<sub_group, int8_t,  mx::use::a, 8, 32, mx::layout::row_major> A8;
                mx::joint_matrix<sub_group, int8_t,  mx::use::b, 32, 16> B8;
                mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> C8;
                mx::joint_matrix<sub_group, float,   mx::use::accumulator, 8, 16> C8f;
                mx::joint_matrix_fill(sg, C8, 0);
                mx::joint_matrix_load(sg, A8, xmp_g_i8(Af_d), 32);
                mx::joint_matrix_load(sg, B8, xmp_g_i8(Bf_d), 16);
                mx::joint_matrix_mad(sg, C8, A8, B8, C8);
                mx::joint_matrix_copy(sg, C8, C8f);
                mx::joint_matrix_store(sg, C8f, xmp_g_f(Cf_d), 32, mx::layout::row_major);
            });
        });
        q.wait_and_throw();
        std::vector<float> hCf(16 * 32, -777.0f);
        q.memcpy(hCf.data(), Cf_d, hCf.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < M8; m++) for (int n = 0; n < N16; n++)
            if (hCf[m * 32 + n] != (float) hBf[m * 16 + n]) bad++;
        printf("K9f identity-A mad chain (f32 readback): bad=%d/128 %s\n", bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            for (int m = 0; m < M8; m++) {
                printf("row%d dev:", m);
                for (int n = 0; n < N16; n++) printf(" %6.0f", hCf[m * 32 + n]);
                printf("\nrow%d ref:", m);
                for (int n = 0; n < N16; n++) printf(" %6d", (int) hBf[m * 16 + n]);
                printf("\n");
            }
        }
        free(Af_d, q); free(Bf_d, q); free(Cf_d, q);
    }


    // K10: f16 A/B load + mad (production f16 shape: M=16, K=16, N=16, f32 acc)
    //      A identity (A[m][k]=m==k), B[k][n] = k*16+n (0..255, exact in f16)
    //      -> C[m][n] must equal m*16+n. Isolates the f16 B load on 2026.1.1.
    {
        sycl::half * A16_d = malloc_device<sycl::half>(16 * 16, q);
        sycl::half * B16_d = malloc_device<sycl::half>(16 * 16, q);
        float   * C10_d   = malloc_device<float>(16 * 32, q);
        std::vector<sycl::half> hAh(16 * 16), hBh(16 * 16);
        for (int m = 0; m < 16; m++) for (int k = 0; k < 16; k++) hAh[m * 16 + k] = sycl::half((k == m) ? 1.0f : 0.0f);
        for (int k = 0; k < 16; k++) for (int n = 0; n < 16; n++) hBh[k * 16 + n] = sycl::half((float)(k * 16 + n));
        q.memcpy(A16_d, hAh.data(), hAh.size() * 2).wait();
        q.memcpy(B16_d, hBh.data(), hBh.size() * 2).wait();
        q.fill(C10_d, -1.0f, 16 * 32 * 4).wait();
        q.submit([&](handler & h) {
            h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
                [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                mx::joint_matrix<sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> A8;
                mx::joint_matrix<sub_group, sycl::half, mx::use::b, 16, 16> B8;
                mx::joint_matrix<sub_group, float, mx::use::accumulator, 16, 16> C8;
                mx::joint_matrix_fill(sg, C8, 0.0f);
                mx::joint_matrix_load(sg, A8, xmp_g_h(A16_d), 16);
                mx::joint_matrix_load(sg, B8, xmp_g_h(B16_d), 16);
                mx::joint_matrix_mad(sg, C8, A8, B8, C8);
                mx::joint_matrix_store(sg, C8, xmp_g_f(C10_d), 32, mx::layout::row_major);
            });
        });
        q.wait_and_throw();
        std::vector<float> hC10(16 * 32, -777.0f);
        q.memcpy(hC10.data(), C10_d, hC10.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < 16; m++) for (int n = 0; n < 16; n++)
            if (hC10[m * 32 + n] != (float)(m * 16 + n)) bad++;
        printf("K10 f16 A/B mad (identity): bad=%d/256 %s\n", bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            for (int m = 0; m < 4; m++) {
                printf("row%d dev:", m);
                for (int n = 0; n < 16; n++) printf(" %6.0f", hC10[m * 32 + n]);
                printf("\nrow%d ref:", m);
                for (int n = 0; n < 16; n++) printf(" %6d", m * 16 + n);
                printf("\n");
            }
        }
        free(A16_d, q); free(B16_d, q); free(C10_d, q);
    }


    // K11: int8 B, PRODUCTION SoA geometry: B[k][n] at element (k + n*S), S=64, ldm=S, row_major
    //      identity A -> C[m][n] = B[m][n] = (m*7 + n*3) % 64 - 32
    {
        const int S = 64;
        int8_t  * B11_d = malloc_device<int8_t>(31 + 15 * S + S, q);
        int8_t  * A11_d = malloc_device<int8_t>(8 * 32, q);
        float   * C11_d = malloc_device<float>(16 * 32, q);
        std::vector<int8_t> hB(31 + 15 * S + S, 0), hA(8 * 32, 0);
        for (int k = 0; k < 32; k++) for (int n = 0; n < 16; n++) hB[k + n * S] = (int8_t)(((k * 7 + n * 3) % 64) - 32);
        for (int m = 0; m < 8; m++) for (int k = 0; k < 32; k++) hA[m * 32 + k] = (k == m) ? 1 : 0;
        q.memcpy(B11_d, hB.data(), hB.size()).wait();
        q.memcpy(A11_d, hA.data(), hA.size()).wait();
        q.fill(C11_d, -1.0f, 16 * 32 * 4).wait();
        q.submit([&](handler & h) {
            h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
                [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                mx::joint_matrix<sub_group, int8_t,  mx::use::a, 8, 32, mx::layout::row_major> A8;
                mx::joint_matrix<sub_group, int8_t,  mx::use::b, 32, 16> B8;
                mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> C8;
                mx::joint_matrix<sub_group, float,   mx::use::accumulator, 8, 16> C8f;
                mx::joint_matrix_fill(sg, C8, 0);
                mx::joint_matrix_load(sg, A8, xmp_g_i8(A11_d), 32);
                mx::joint_matrix_load(sg, B8, xmp_g_i8(B11_d), S);
                mx::joint_matrix_mad(sg, C8, A8, B8, C8);
                mx::joint_matrix_copy(sg, C8, C8f);
                mx::joint_matrix_store(sg, C8f, xmp_g_f(C11_d), 32, mx::layout::row_major);
            });
        });
        q.wait_and_throw();
        std::vector<float> hC(16 * 32, -777.0f);
        q.memcpy(hC.data(), C11_d, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < 8; m++) for (int n = 0; n < 16; n++)
            if (hC[m * 32 + n] != (float)(((m * 7 + n * 3) % 64) - 32)) bad++;
        printf("K11 int8 B SoA-geom (ldm=%d, row): bad=%d/128 %s\n", S, bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            for (int m = 0; m < 4; m++) {
                printf("row%d dev:", m);
                for (int n = 0; n < 16; n++) printf(" %6d", hC[m * 32 + n]);
                printf("\nrow%d ref:", m);
                for (int n = 0; n < 16; n++) printf(" %6d", ((m * 7 + n * 3) % 64) - 32);
                printf("\n");
            }
        }
        free(B11_d, q); free(A11_d, q); free(C11_d, q);
    }

    // K12: f16 B, B_v geometry: B[k][n] at element (k + n*S), S=32, ldm=S, row_major [16][16]
    {
        const int S = 32;
        sycl::half * B12_d = malloc_device<sycl::half>(15 + 15 * S + S, q);
        sycl::half * A12_d = malloc_device<sycl::half>(16 * 16, q);
        float   * C12_d   = malloc_device<float>(16 * 32, q);
        std::vector<sycl::half> hB(15 + 15 * S + S), hA(16 * 16);
        for (int k = 0; k < 16; k++) for (int n = 0; n < 16; n++) hB[k + n * S] = sycl::half((float)(k + n * S));
        for (int m = 0; m < 16; m++) for (int k = 0; k < 16; k++) hA[m * 16 + k] = sycl::half((k == m) ? 1.0f : 0.0f);
        q.memcpy(B12_d, hB.data(), hB.size() * 2).wait();
        q.memcpy(A12_d, hA.data(), hA.size() * 2).wait();
        q.fill(C12_d, -1.0f, 16 * 32 * 4).wait();
        q.submit([&](handler & h) {
            h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
                [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                mx::joint_matrix<sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> A8;
                mx::joint_matrix<sub_group, sycl::half, mx::use::b, 16, 16> B8;
                mx::joint_matrix<sub_group, float, mx::use::accumulator, 16, 16> C8;
                mx::joint_matrix_fill(sg, C8, 0.0f);
                mx::joint_matrix_load(sg, A8, xmp_g_h(A12_d), 16);
                mx::joint_matrix_load(sg, B8, xmp_g_h(B12_d), S);
                mx::joint_matrix_mad(sg, C8, A8, B8, C8);
                mx::joint_matrix_store(sg, C8, xmp_g_f(C12_d), 32, mx::layout::row_major);
            });
        });
        q.wait_and_throw();
        std::vector<float> hC(16 * 32, -777.0f);
        q.memcpy(hC.data(), C12_d, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < 16; m++) for (int n = 0; n < 16; n++)
            if (hC[m * 32 + n] != (float)(m + n * S)) bad++;
        printf("K12 f16 B_v-geom (ldm=%d, row): bad=%d/256 %s\n", S, bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            for (int m = 0; m < 4; m++) {
                printf("row%d dev:", m);
                for (int n = 0; n < 16; n++) printf(" %6.0f", hC[m * 32 + n]);
                printf("\nrow%d ref:", m);
                for (int n = 0; n < 16; n++) printf(" %6d", m + n * S);
                printf("\n");
            }
        }
        free(B12_d, q); free(A12_d, q); free(C12_d, q);
    }

    // K13: f16 B, B_k geometry: same element map (k + n*S) but col_major tag, ldm=S
    {
        const int S = 32;
        sycl::half * B13_d = malloc_device<sycl::half>(15 + 15 * S + S, q);
        sycl::half * A13_d = malloc_device<sycl::half>(16 * 16, q);
        float   * C13_d   = malloc_device<float>(16 * 32, q);
        std::vector<sycl::half> hB(15 + 15 * S + S), hA(16 * 16);
        for (int k = 0; k < 16; k++) for (int n = 0; n < 16; n++) hB[k + n * S] = sycl::half((float)(k + n * S));
        for (int m = 0; m < 16; m++) for (int k = 0; k < 16; k++) hA[m * 16 + k] = sycl::half((k == m) ? 1.0f : 0.0f);
        q.memcpy(B13_d, hB.data(), hB.size() * 2).wait();
        q.memcpy(A13_d, hA.data(), hA.size() * 2).wait();
        q.fill(C13_d, -1.0f, 16 * 32 * 4).wait();
        q.submit([&](handler & h) {
            h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
                [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                mx::joint_matrix<sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> A8;
                mx::joint_matrix<sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::col_major> B8;
                mx::joint_matrix<sub_group, float, mx::use::accumulator, 16, 16> C8;
                mx::joint_matrix_fill(sg, C8, 0.0f);
                mx::joint_matrix_load(sg, A8, xmp_g_h(A13_d), 16);
                mx::joint_matrix_load(sg, B8, xmp_g_h(B13_d), S);
                mx::joint_matrix_mad(sg, C8, A8, B8, C8);
                mx::joint_matrix_store(sg, C8, xmp_g_f(C13_d), 32, mx::layout::row_major);
            });
        });
        q.wait_and_throw();
        std::vector<float> hC(16 * 32, -777.0f);
        q.memcpy(hC.data(), C13_d, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < 16; m++) for (int n = 0; n < 16; n++)
            if (hC[m * 32 + n] != (float)(m + n * S)) bad++;
        printf("K13 f16 B_k-geom (ldm=%d, col): bad=%d/256 %s\n", S, bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            for (int m = 0; m < 4; m++) {
                printf("row%d dev:", m);
                for (int n = 0; n < 16; n++) printf(" %6.0f", hC[m * 32 + n]);
                printf("\nrow%d ref:", m);
                for (int n = 0; n < 16; n++) printf(" %6d", m + n * S);
                printf("\n");
            }
        }
        free(B13_d, q); free(A13_d, q); free(C13_d, q);
    }


    // K14: int8 B [32][16] col_major, n-stride geom (k + n*S, S=64), identity A
    //      C[m][n] = B[m][n] = (m*7 + n*3) % 64 - 32 (m<8)
    {
        const int S = 64;
        int8_t   * B14_d = malloc_device<int8_t>(31 + 15 * S + S, q);
        int8_t   * A14_d = malloc_device<int8_t>(8 * 32, q);
        int32_t  * C14_d = malloc_device<int32_t>(16 * 32, q);
        std::vector<int8_t> hB(31 + 15 * S + S, 0), hA(8 * 32, 0);
        for (int k = 0; k < 32; k++) for (int n = 0; n < 16; n++) hB[k + n * S] = (int8_t)(((k * 7 + n * 3) % 64) - 32);
        for (int m = 0; m < 8; m++) for (int k = 0; k < 32; k++) hA[m * 32 + k] = (k == m) ? 1 : 0;
        q.memcpy(B14_d, hB.data(), hB.size()).wait();
        q.memcpy(A14_d, hA.data(), hA.size()).wait();
        q.fill(C14_d, -1.0f, 16 * 32 * 4).wait();
        q.submit([&](handler & h) {
            h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
                [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                mx::joint_matrix<sub_group, int8_t,  mx::use::a, 8, 32, mx::layout::row_major> A8;
                mx::joint_matrix<sub_group, int8_t,  mx::use::b, 32, 16, mx::layout::col_major> B8;
                mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> C8;
                mx::joint_matrix_fill(sg, C8, 0);
                mx::joint_matrix_load(sg, A8, xmp_g_i8(A14_d), 32);
                mx::joint_matrix_load(sg, B8, xmp_g_i8(B14_d), S);
                mx::joint_matrix_mad(sg, C8, A8, B8, C8);
                mx::joint_matrix_store(sg, C8, xmp_g_i32(C14_d), 32, mx::layout::row_major);
            });
        });
        q.wait_and_throw();
        std::vector<int32_t> hC(16 * 32, -777);
        q.memcpy(hC.data(), C14_d, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < 8; m++) for (int n = 0; n < 16; n++)
            if (hC[m * 32 + n] != ((m * 7 + n * 3) % 64) - 32) bad++;
        printf("K14 int8 B col n-str (ldm=%d): bad=%d/128 %s\n", S, bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            for (int m = 0; m < 4; m++) {
                printf("row%d dev:", m);
                for (int n = 0; n < 16; n++) printf(" %6.0f", hC[m * 32 + n]);
                printf("\nrow%d ref:", m);
                for (int n = 0; n < 16; n++) printf(" %6d", ((m * 7 + n * 3) % 64) - 32);
                printf("\n");
            }
        }
        free(B14_d, q); free(A14_d, q); free(C14_d, q);
    }

    // K15: same as K14 but A = constant 1 (full K=32 span) + COPY i32->f32 before store
    //      C[m][n] = sum_{k=0..31} (k*7 + n*3) % 64 - 32
    {
        const int S = 64;
        int8_t  * B15_d = malloc_device<int8_t>(31 + 15 * S + S, q);
        int8_t  * A15_d = malloc_device<int8_t>(8 * 32, q);
        float   * C15_d = malloc_device<float>(16 * 32, q);
        std::vector<int8_t> hB(31 + 15 * S + S, 0), hA(8 * 32, 1);
        for (int k = 0; k < 32; k++) for (int n = 0; n < 16; n++) hB[k + n * S] = (int8_t)(((k * 7 + n * 3) % 64) - 32);
        q.memcpy(B15_d, hB.data(), hB.size()).wait();
        q.memcpy(A15_d, hA.data(), hA.size()).wait();
        q.fill(C15_d, -1.0f, 16 * 32 * 4).wait();
        q.submit([&](handler & h) {
            h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
                [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                mx::joint_matrix<sub_group, int8_t,  mx::use::a, 8, 32, mx::layout::row_major> A8;
                mx::joint_matrix<sub_group, int8_t,  mx::use::b, 32, 16, mx::layout::col_major> B8;
                mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> C8;
                mx::joint_matrix<sub_group, float,   mx::use::accumulator, 8, 16> C8f;
                mx::joint_matrix_fill(sg, C8, 0);
                mx::joint_matrix_load(sg, A8, xmp_g_i8(A15_d), 32);
                mx::joint_matrix_load(sg, B8, xmp_g_i8(B15_d), S);
                mx::joint_matrix_mad(sg, C8, A8, B8, C8);
                mx::joint_matrix_copy(sg, C8, C8f);
                mx::joint_matrix_store(sg, C8f, xmp_g_f(C15_d), 32, mx::layout::row_major);
            });
        });
        q.wait_and_throw();
        std::vector<float> hC(16 * 32, -777.0f);
        q.memcpy(hC.data(), C15_d, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < 8; m++) for (int n = 0; n < 16; n++) {
            int ref = 0;
            for (int k = 0; k < 32; k++) ref += ((k * 7 + n * 3) % 64) - 32;
            if (hC[m * 32 + n] != (float)ref) bad++;
        }
        printf("K15 int8 B col n-str + copy: bad=%d/128 %s\n", bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            int ref0 = 0;
            for (int k = 0; k < 32; k++) ref0 += ((k * 7 + 0 * 3) % 64) - 32;
            printf("row0 dev:");
            for (int n = 0; n < 16; n++) printf(" %6.0f", hC[n]);
            printf("\nrow0 ref n0=%d\n", ref0);
        }
        free(B15_d, q); free(A15_d, q); free(C15_d, q);
    }


    // K16: f16 B [16][16] col_major, K-STRIDE geom (B[k][n] at k*S + n, S=32), ldm=S
    //      = production B_v geometry with col_major tag. Identity A -> C[m][n] = B[m][n] = m*S + n
    {
        const int S = 32;
        sycl::half * B16_d = malloc_device<sycl::half>(15 * S + 16, q);
        sycl::half * A16_d = malloc_device<sycl::half>(16 * 16, q);
        float   * C16_d   = malloc_device<float>(16 * 32, q);
        std::vector<sycl::half> hB(15 * S + 16), hA(16 * 16);
        for (int k = 0; k < 16; k++) for (int n = 0; n < 16; n++) hB[k * S + n] = sycl::half((float)(k * S + n));
        for (int m = 0; m < 16; m++) for (int k = 0; k < 16; k++) hA[m * 16 + k] = sycl::half((k == m) ? 1.0f : 0.0f);
        q.memcpy(B16_d, hB.data(), hB.size() * 2).wait();
        q.memcpy(A16_d, hA.data(), hA.size() * 2).wait();
        q.fill(C16_d, -1.0f, 16 * 32 * 4).wait();
        q.submit([&](handler & h) {
            h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
                [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                mx::joint_matrix<sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> A8;
                mx::joint_matrix<sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::col_major> B8;
                mx::joint_matrix<sub_group, float, mx::use::accumulator, 16, 16> C8;
                mx::joint_matrix_fill(sg, C8, 0.0f);
                mx::joint_matrix_load(sg, A8, xmp_g_h(A16_d), 16);
                mx::joint_matrix_load(sg, B8, xmp_g_h(B16_d), S);
                mx::joint_matrix_mad(sg, C8, A8, B8, C8);
                mx::joint_matrix_store(sg, C8, xmp_g_f(C16_d), 32, mx::layout::row_major);
            });
        });
        q.wait_and_throw();
        std::vector<float> hC(16 * 32, -777.0f);
        q.memcpy(hC.data(), C16_d, hC.size() * 4).wait();
        int bad = 0;
        for (int m = 0; m < 16; m++) for (int n = 0; n < 16; n++)
            if (hC[m * 32 + n] != (float)(m * S + n)) bad++;
        printf("K16 f16 B col k-str (ldm=%d): bad=%d/256 %s\n", S, bad, bad == 0 ? "PASS" : "FAIL");
        if (bad) {
            for (int m = 0; m < 4; m++) {
                printf("row%d dev:", m);
                for (int n = 0; n < 16; n++) printf(" %6.0f", hC[m * 32 + n]);
                printf("\nrow%d ref:", m);
                for (int n = 0; n < 16; n++) printf(" %6d", m * S + n);
                printf("\n");
            }
        }
        free(B16_d, q); free(A16_d, q); free(C16_d, q);
    }

    free(G_i32, q); free(G_f, q); free(A8_d, q); free(B8_d, q); free(C_e, q);
    free(C_a, q); free(C_b, q); free(C_c, q); free(C_d, q);
    q.wait();
    printf("done\n");
    return 0;
}
