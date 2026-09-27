// xmx-probe-bldm.cpp - B-operand delivery-mapping probe (tag x ldm x data-layout)
//
// Purpose: instead of ASSUMING which index ldm strides, DECODE what the load
// actually delivered. Every source slot (i,j) carries a UNIQUE value
//   v(i,j) = (i*3 + 5*j) % 127 - 63   (int8)  /  i*16+j  (f16, 0..255)
// so the readback C[m][n] = value at slot (a,b) tells us exactly which source
// slot landed in tile slot (m,n):
//   C[m][n] == v(m,n)  -> IDENTITY  (tile slot (m,n) got source (m,n))
//   C[m][n] == v(n,m)  -> TRANSPOSE
//   otherwise          -> MISMATCH (dump rows)
//
// A = identity (A[m][k] = m==k) so C[m][n] = B[m][n] readback directly.
// Shapes are the ONLY claimed XMX shapes (spec appendix, bmg_g21/g31):
//   int8: M<=8, N=16, K=32, C=sint32   (A [8][32], B [32][16])
//   f16 : M=16, N=16, K=16, C=fp32     (A [16][16], B [16][16])
// All arms satisfy the spec stride restrictions (stride*sizeof(T) % 8 == 0,
// base 4B aligned, stride*sizeof(T) < 2^24).
//
// Arms (DATAPOS = where value v(i,j) is placed in the buffer;
//   "i*16+j" = compact [K][N] (i8b/production-LDS layout),
//   "n*S+i"  = SoA layout (n/pos outer, i/dim inner)):
//  int8 B [32][16]:
//   T1 row ldm=16    data i*16+j   (i8b config, old-FE PASS)
//   T2 row ldm=32    data n*32+i
//   T3 row ldm=64    data n*64+i   (K11 config)
//   T4 row ldm=1088  data n*1088+i (= PRODUCTION int8 v2 SoA load EXACTLY)
//   T5 col ldm=32    data n*32+i   (K14 config)
//   T6 col ldm=16    data i*16+j   (i8b col config, old-FE "wrong")
//  f16 B [16][16]:
//   T7 row ldm=1024  data k*1024+n (= PRODUCTION B_v PV load EXACTLY:
//                      B[k=pos][n=dim] = V[(pos_c+k)*1024 + dim_dc+n])
//   T8 row ldm=16    data i*16+j   (K10 config)
//   T9 col ldm=1024  data n*1024+i (= PRODUCTION B_k QK load EXACTLY:
//                      B[k=dim][n=pos] = K[pos*1024 + dim], SoA f16)

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace sycl;
namespace mx = sycl::ext::oneapi::experimental::matrix;

template <typename T>
T * malloc_device(size_t n, queue & q) { return (T *) sycl::malloc_device(n * sizeof(T), q); }
void free_dev(void * p, queue & q) { sycl::free(p, q); }

using mptr_g_i8 = multi_ptr<int8_t, access::address_space::global_space, access::decorated::legacy>;
using mptr_g_i32 = multi_ptr<int32_t, access::address_space::global_space, access::decorated::legacy>;
using mptr_g_h = multi_ptr<sycl::half, access::address_space::global_space, access::decorated::legacy>;
using mptr_g_f = multi_ptr<float, access::address_space::global_space, access::decorated::legacy>;
static mptr_g_i8 xmp_g_i8(int8_t * p) { return mptr_g_i8(p); }
static mptr_g_i32 xmp_g_i32(int32_t * p) { return mptr_g_i32(p); }
static mptr_g_h xmp_g_h(sycl::half * p) { return mptr_g_h(p); }
static mptr_g_f xmp_g_f(float * p) { return mptr_g_f(p); }

using sub_group = sycl::sub_group;

static const char * classify_unused(int m, int n, int val, int id, int tr) {
    if (val == id) return "IDENTITY";
    if (val == tr) return "TRANSPOSE";
    return "MISMATCH";
}

// ---------------- int8 arms ----------------
// i = K index (0..31, pairs with A row m); j = N index (0..15, output col).
// DATAPOS: 0 = i*16+j (compact), 1 = n*S+i (SoA-style, n outer, S=LDM)
template <mx::layout LAY, int LDM, int DATAPOS, int CONSTA>
void t8(queue & q, const char * name) {
    const int S = LDM;
    const size_t nbuf = DATAPOS == 0 ? (size_t)32 * 16 : (size_t)(15 * S + 32);
    int8_t * B_d = malloc_device<int8_t>(nbuf, q);
    int8_t * A_d = malloc_device<int8_t>(8 * 32, q);
    int32_t * C_d = malloc_device<int32_t>(8 * 16, q);
    std::vector<int8_t> hB(nbuf, 0), hA(8 * 32, 0);
    for (int i = 0; i < 32; i++) for (int j = 0; j < 16; j++) {
        int v = ((i * 3 + 5 * j) % 127) - 63;
        size_t pos = DATAPOS == 0 ? (size_t)i * 16 + j : (size_t)j * S + i;
        hB[pos] = (int8_t) v;
    }
    for (int m = 0; m < 8; m++) for (int k = 0; k < 32; k++)
        hA[m * 32 + k] = CONSTA ? ((k == m * 4) ? 1 : 0) : ((k == m) ? 1 : 0);
    q.memcpy(B_d, hB.data(), hB.size()).wait();
    q.memcpy(A_d, hA.data(), hA.size()).wait();
    q.fill(C_d, -777, 8 * 16 * 4).wait();
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            mx::joint_matrix<sub_group, int8_t, mx::use::a, 8, 32, mx::layout::row_major> A8;
            mx::joint_matrix<sub_group, int8_t, mx::use::b, 32, 16, LAY> B8;
            mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> C8;
            mx::joint_matrix_fill(sg, C8, 0);
            mx::joint_matrix_load(sg, A8, xmp_g_i8(A_d), 32);
            mx::joint_matrix_load(sg, B8, xmp_g_i8(B_d), LDM);
            mx::joint_matrix_mad(sg, C8, A8, B8, C8);
            mx::joint_matrix_store(sg, C8, xmp_g_i32(C_d), 16, mx::layout::row_major);
        });
    });
    q.wait_and_throw();
    std::vector<int32_t> hC(8 * 16, 0x777);
    q.memcpy(hC.data(), C_d, hC.size() * 4).wait();
    int nid = 0, ntr = 0, nom = 0;
    for (int m = 0; m < 8; m++) for (int j = 0; j < 16; j++) {
        int v = hC[m * 16 + j];
        if (v == ((m * 3 + 5 * j) % 127) - 63) nid++;
        else if (v == ((j * 3 + 5 * m) % 127) - 63) ntr++;
        else nom++;
    }
    printf("%s: id=%d tr=%d mis=%d/128 -> %s\n", name, nid, ntr, nom,
           nom == 0 && nid == 128 ? "DELIVERS-IDENTITY" :
           nom == 0 && ntr == 128 ? "DELIVERS-TRANSPOSE" : "SEE-ROWS");
    if (nom || (ntr && nid)) {
        for (int m = 0; m < 3; m++) {
            printf("  row%d dev:", m);
            for (int j = 0; j < 16; j++) printf(" %5d", hC[m * 16 + j]);
            printf("\n  row%d v(m,j):", m);
            for (int j = 0; j < 16; j++) printf(" %5d", ((m * 3 + 5 * j) % 127) - 63);
            printf("\n  row%d v(j,m):", m);
            for (int j = 0; j < 16; j++) printf(" %5d", ((j * 3 + 5 * m) % 127) - 63);
            printf("\n");
        }
    }
    free_dev(B_d, q); free_dev(A_d, q); free_dev(C_d, q);
}

// ---------------- f16 arms ----------------
// i = K index (0..15), j = N index (0..15). value = i*16+j (0..255, exact f16).
template <mx::layout LAY, int LDM, int DATAPOS>
void t16(queue & q, const char * name) {
    const int S = LDM;
    const size_t nbuf = DATAPOS == 0 ? (size_t)16 * 16
                     : DATAPOS == 1 ? (size_t)(15 * S + 16)
                                    : (size_t)(15 * S + 16);
    sycl::half * B_d = malloc_device<sycl::half>(nbuf, q);
    sycl::half * A_d = malloc_device<sycl::half>(16 * 16, q);
    float * C_d = malloc_device<float>(16 * 16, q);
    std::vector<sycl::half> hB(nbuf, sycl::half(0.0f)), hA(16 * 16, sycl::half(0.0f));
    for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) {
        int v = i * 16 + j;
        size_t pos = DATAPOS == 0 ? (size_t)i * 16 + j
                     : DATAPOS == 1 ? (size_t)j * S + i
                                    : (size_t)i * S + j;
        hB[pos] = sycl::half((float) v);
    }
    for (int m = 0; m < 16; m++) for (int k = 0; k < 16; k++) hA[m * 16 + k] = sycl::half((k == m) ? 1.0f : 0.0f);
    q.memcpy(B_d, hB.data(), hB.size() * 2).wait();
    q.memcpy(A_d, hA.data(), hA.size() * 2).wait();
    q.fill(C_d, -777.0f, 16 * 16 * 4).wait();
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            mx::joint_matrix<sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> A8;
            mx::joint_matrix<sub_group, sycl::half, mx::use::b, 16, 16, LAY> B8;
            mx::joint_matrix<sub_group, float, mx::use::accumulator, 16, 16> C8;
            mx::joint_matrix_fill(sg, C8, 0.0f);
            mx::joint_matrix_load(sg, A8, xmp_g_h(A_d), 16);
            mx::joint_matrix_load(sg, B8, xmp_g_h(B_d), LDM);
            mx::joint_matrix_mad(sg, C8, A8, B8, C8);
            mx::joint_matrix_store(sg, C8, xmp_g_f(C_d), 16, mx::layout::row_major);
        });
    });
    q.wait_and_throw();
    std::vector<float> hC(16 * 16, -777.0f);
    q.memcpy(hC.data(), C_d, hC.size() * 4).wait();
    int nid = 0, ntr = 0, nom = 0;
    for (int m = 0; m < 16; m++) for (int j = 0; j < 16; j++) {
        float v = hC[m * 16 + j];
        if (v == (float)(m * 16 + j)) nid++;
        else if (v == (float)(j * 16 + m)) ntr++;
        else nom++;
    }
    printf("%s: id=%d tr=%d mis=%d/256 -> %s\n", name, nid, ntr, nom,
           nom == 0 && nid == 256 ? "DELIVERS-IDENTITY" :
           nom == 0 && ntr == 256 ? "DELIVERS-TRANSPOSE" : "SEE-ROWS");
    if (nom || (ntr && nid)) {
        for (int m = 0; m < 3; m++) {
            printf("  row%d dev:", m);
            for (int j = 0; j < 16; j++) printf(" %6.0f", hC[m * 16 + j]);
            printf("\n  row%d v(m,j):", m);
            for (int j = 0; j < 16; j++) printf(" %6d", m * 16 + j);
            printf("\n  row%d v(j,m):", m);
            for (int j = 0; j < 16; j++) printf(" %6d", j * 16 + m);
            printf("\n");
        }
    }
    free_dev(B_d, q); free_dev(A_d, q); free_dev(C_d, q);
}



// T15: K9f's exact pipeline with UNIQUE B values: int8 identity A + row ldm=16 B
//      -> f32 acc [8][16] -> f32 store ldm=32. Decodes whether the 4n pattern in
//      K9f came from the B load (periodic values) or the f32 store (ldm=32).
void t15(queue & q) {
    int8_t * B_d = malloc_device<int8_t>(32 * 16, q);
    int8_t * A_d = malloc_device<int8_t>(8 * 32, q);
    float * C_d = malloc_device<float>(8 * 32, q);
    std::vector<int8_t> hB(32 * 16), hA(8 * 32);
    for (int i = 0; i < 32; i++) for (int j = 0; j < 16; j++)
        hB[i * 16 + j] = (int8_t)((((i * 3 + 5 * j) % 127) - 63));
    for (int m = 0; m < 8; m++) for (int k = 0; k < 32; k++) hA[m * 32 + k] = (k == m) ? 1 : 0;
    q.memcpy(B_d, hB.data(), hB.size()).wait();
    q.memcpy(A_d, hA.data(), hA.size()).wait();
    q.fill(C_d, -777.0f, 8 * 32 * 4).wait();
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            mx::joint_matrix<sub_group, int8_t, mx::use::a, 8, 32, mx::layout::row_major> A8;
            mx::joint_matrix<sub_group, int8_t, mx::use::b, 32, 16, mx::layout::row_major> B8;
            mx::joint_matrix<sub_group, float, mx::use::accumulator, 8, 16> C8;
            mx::joint_matrix_fill(sg, C8, 0.0f);
            mx::joint_matrix_load(sg, A8, xmp_g_i8(A_d), 32);
            mx::joint_matrix_load(sg, B8, xmp_g_i8(B_d), 16);
            mx::joint_matrix_mad(sg, C8, A8, B8, C8);
            mx::joint_matrix_store(sg, C8, xmp_g_f(C_d), 32, mx::layout::row_major);
        });
    });
    q.wait_and_throw();
    std::vector<float> hC(8 * 32, -777.0f);
    q.memcpy(hC.data(), C_d, hC.size() * 4).wait();
    int nid = 0, ntr = 0, nom = 0;
    for (int m = 0; m < 8; m++) for (int j = 0; j < 16; j++) {
        float v = hC[m * 32 + j];
        float id = (float)(((m * 3 + 5 * j) % 127) - 63);
        float tr = (float)(((j * 3 + 5 * m) % 127) - 63);
        if (v == id) nid++;
        else if (v == tr) ntr++;
        else nom++;
    }
    printf("T15 i8->f32acc store ldm=32 (K9f pipeline, uniq B): id=%d tr=%d mis=%d/128 -> %s\n",
           nid, ntr, nom, nom == 0 && nid == 128 ? "DELIVERS-IDENTITY" :
           nom == 0 && ntr == 128 ? "DELIVERS-TRANSPOSE" : "SEE-ROWS");
    for (int m = 0; m < 2; m++) {
        printf("  row%d dev:", m);
        for (int j = 0; j < 16; j++) printf(" %7.0f", hC[m * 32 + j]);
        printf("\n");
    }
    free_dev(B_d, q); free_dev(A_d, q); free_dev(C_d, q);
}

// T16: K9e EXACT reproduction (constant A row a_m=(m%7)-3, B[k][n]=((k%5)-2)+n)
//      with CPU reference of the TRUE product - settles whether K9e's pattern is real.
void t16k(queue & q) {
    int8_t * B_d = malloc_device<int8_t>(32 * 16, q);
    int8_t * A_d = malloc_device<int8_t>(8 * 32, q);
    int32_t * C_d = malloc_device<int32_t>(8 * 16, q);
    std::vector<int8_t> hB(32 * 16), hA(8 * 32);
    for (int k = 0; k < 32; k++) for (int n = 0; n < 16; n++) hB[k * 16 + n] = (int8_t)(((k % 5) - 2) + n);
    for (int m = 0; m < 8; m++) for (int k = 0; k < 32; k++) hA[m * 32 + k] = (int8_t)((m % 7) - 3);
    q.memcpy(B_d, hB.data(), hB.size()).wait();
    q.memcpy(A_d, hA.data(), hA.size()).wait();
    q.fill(C_d, -777, 8 * 16 * 4).wait();
    q.submit([&](handler & h) {
        h.parallel_for(nd_range<1>(range<1>(16), range<1>(16)),
            [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            mx::joint_matrix<sub_group, int8_t, mx::use::a, 8, 32, mx::layout::row_major> A8;
            mx::joint_matrix<sub_group, int8_t, mx::use::b, 32, 16, mx::layout::row_major> B8;
            mx::joint_matrix<sub_group, int32_t, mx::use::accumulator, 8, 16> C8;
            mx::joint_matrix_fill(sg, C8, 0);
            mx::joint_matrix_load(sg, A8, xmp_g_i8(A_d), 32);
            mx::joint_matrix_load(sg, B8, xmp_g_i8(B_d), 16);
            mx::joint_matrix_mad(sg, C8, A8, B8, C8);
            mx::joint_matrix_store(sg, C8, xmp_g_i32(C_d), 16, mx::layout::row_major);
        });
    });
    q.wait_and_throw();
    std::vector<int32_t> hC(8 * 16, -777);
    q.memcpy(hC.data(), C_d, hC.size() * 4).wait();
    int bad = 0;
    for (int m = 0; m < 8; m++) {
        int ref0 = 0;
        for (int k = 0; k < 32; k++) ref0 += hA[m * 32 + k] * hB[k * 16 + 0];
        for (int n = 0; n < 16; n++) {
            int ref = 0;
            for (int k = 0; k < 32; k++) ref += hA[m * 32 + k] * hB[k * 16 + n];
            if (hC[m * 16 + n] != ref) bad++;
        }
        if (m < 3) {
            int ref = 0;
            for (int k = 0; k < 32; k++) ref += hA[m * 32 + k] * hB[k * 16 + 0];
            (void) ref0;
        }
    }
    printf("T16 K9e-exact (const A, period-5 B): bad=%d/128 vs CPU ref\n", bad);
    for (int m = 0; m < 2; m++) {
        printf("  row%d dev:", m);
        for (int n = 0; n < 16; n++) printf(" %5d", hC[m * 16 + n]);
        printf("\n  row%d ref:", m);
        for (int n = 0; n < 16; n++) {
            int ref = 0;
            for (int k = 0; k < 32; k++) ref += hA[m * 32 + k] * hB[k * 16 + n];
            printf(" %5d", ref);
        }
        printf("\n");
    }
    free_dev(B_d, q); free_dev(A_d, q); free_dev(C_d, q);
}

int main() {
    try {
        queue q(gpu_selector{});
        const std::string devname = q.get_device().get_info<info::device::name>();
        printf("device: %s\n", devname.c_str());

        t8 <mx::layout::row_major, 16, 0, 0>(q, "T1 i8 row ldm=16   data i*16+j  (i8b cfg)");
        t8 <mx::layout::row_major, 32, 1, 0>(q, "T2 i8 row ldm=32   data n*32+i  ");
        t8 <mx::layout::row_major, 64, 1, 0>(q, "T3 i8 row ldm=64   data n*64+i  (K11 cfg)");
        t8 <mx::layout::row_major, 1088, 1, 0>(q, "T4 i8 row ldm=1088 data n*1088+i (PROD int8 v2)");
        t8 <mx::layout::col_major, 32, 1, 0>(q, "T5 i8 col ldm=32   data n*32+i  (K14 cfg)");
        t8 <mx::layout::col_major, 16, 0, 0>(q, "T6 i8 col ldm=16   data i*16+j  (i8b col)");

        t8 <mx::layout::row_major, 16, 0, 1>(q, "T10 i8 row ldm=16  A onehot k=4m (K9e A-cfg)");

        t16<mx::layout::row_major, 1024, 2>(q, "T7 f16 row ldm=1024 data k*1024+n (PROD B_v)");
        t16<mx::layout::row_major, 16, 0>(q, "T8 f16 row ldm=16   data i*16+j  (K10 cfg)");
        t16<mx::layout::col_major, 1024, 1>(q, "T9 f16 col ldm=1024 data n*1024+i (PROD B_k)");

        // t15 removed: int8 A/B -> f32 acc mad is rejected at JIT (unclaimed combo)
        t16k(q);
    } catch (const exception & e) {
        printf("SYCL ERROR: %s\n", e.what());
        return 1;
    }
    printf("done\n");
    return 0;
}
