#include <vector>
#include <type_traits>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "ggml-quants.h"
#include <cmath>
#include <cstring>

#define QK_I2_S 128
#define QK_I2 128

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#include <immintrin.h>
// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#elif defined(__loongarch_asx)
// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {

    __m256i tmp1 = __lasx_xvpermi_q(a, a, 0x11);
    __m256i tmp2 = __lasx_xvpermi_q(a, a, 0x00);

    __m128i  tmp1_128 = lasx_extracti128_lo(tmp1);
    __m128i  tmp2_128 = lasx_extracti128_lo(tmp2);

    __m128i sum128 = __lsx_vadd_w(tmp1_128, tmp2_128);

    __m128i ev = __lsx_vpickev_w(sum128, sum128);
    __m128i od = __lsx_vpickod_w(sum128, sum128);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    int sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    return  sum64_1 + sum64_2;
}
#endif
//
// size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
//     // 2 bits per weight
//
//     size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);
//
//     int n = nrow * n_per_row;
//
//     // f32 -> q8
//     double max = 0;
//     for (int i = 0; i < n; ++i) {
//         max = fmax(max, (double)fabs((double)src[i]));
//     }
//     double i2_scale = max;
//
//     uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
//     for (int i=0; i<n; i++) {
//         if (fabs((double)(src[i])) < 1e-6) {
//             q8[i] = 1;
//             continue;
//         }
//         q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
//     }
//
//     memset(dst, 0, n * sizeof(uint8_t) / 4);
//
//     // q8 -> 0, 1, 2
//     //       |  |  |
//     //      -1, 0, 1
//
//     uint8_t* i2_weight = (uint8_t*)dst;
//     for (int i = 0; i < n / QK_I2; i++) {
//         for (int j = 0; j < QK_I2; j++) {
//             int group_idx = j / 32;
//             int group_pos = j % 32;
//             uint8_t temp = (q8[i * QK_I2 + j] << (6 - 2 * group_idx));
//             i2_weight[i * 32 + group_pos] |= temp;
//         }
//     }
//
//     float* scale_ptr = (float*)((char*)i2_weight + n / 4);
//     scale_ptr[0] = i2_scale;
//
//     free(q8);
//
//     // 32B for alignment
//     return nrow * row_size / 4 + 32;
// }

#define ZEROISH_THRESHOLD 1e-6f

// We define codes:
//   3 (binary 11) => negative
//   0 (binary 00) => zero-ish
//   1 (binary 01) => positive
//
// Each float => 2 bits, stored sequentially in memory.

// new quantization?

// size_t quantize_i2_s(const float * src, void * dst,
//                      int64_t nrow, int64_t n_per_row,
//                      const float * quant_weights)
// {
//     // Unused parameter in this variant
//     (void) quant_weights;
//
//     // 1) Calculate row size (like the original code)
//     size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);
//
//     // 2) Total number of floats
//     int64_t n = nrow * n_per_row;
//
//     // 3) Find max absolute value to store as scale
//     double max_val = 0;
//     for (int64_t i = 0; i < n; i++) {
//         double x = fabs((double) src[i]);
//         if (x > max_val) {
//             max_val = x;
//         }
//     }
//     double i2_scale = max_val;
//
//     // 4) Build a temporary array q8 of "codes" in {0,1,3}
//     uint8_t * q8 = (uint8_t *) malloc(n * sizeof(uint8_t));
//     if (!q8) {
//         // Handle allocation failure if needed
//         return 0;
//     }
//
//     for (int64_t i = 0; i < n; i++) {
//         double val = (double) src[i];
//         if (fabs(val) < 1e-6) {
//             // near-zero
//             q8[i] = 0;
//         } else if (val > 0.0) {
//             // positive
//             q8[i] = 1;
//         } else {
//             // negative
//             q8[i] = 3;
//         }
//     }
//
//     // 5) The original code zeroed out n/4 bytes for the 2-bit data:
//     //    Because each float is 2 bits => 2*n bits => 2*n/8 = n/4 bytes.
//     //    (This does NOT round up, so if n is not multiple of 4, some bits are lost.)
//     int64_t bytes_for_bits = n / 4;
//
//     // Clear that memory region
//     memset(dst, 0, bytes_for_bits);
//
//     // 6) Pack the codes into 2 bits each (linearly, no grouping)
//     uint8_t * out_data = (uint8_t *) dst;
//     for (int64_t i = 0; i < n; i++) {
//         // code in {0,1,3}
//         uint8_t code = q8[i];
//
//         // bit index = 2*i
//         // byte index = (2*i)/8
//         // bit offset = (2*i)%8
//         int64_t bit_index  = 2 * i;
//         int64_t byte_index = bit_index / 8;
//         int64_t bit_offset = bit_index % 8;
//
//         // Place the 2-bit code
//         out_data[byte_index] |= (code << bit_offset);
//     }
//
//     // 7) Write the scale factor at offset n/4 from the start of dst
//     float * scale_ptr = (float *) ((uint8_t *) dst + bytes_for_bits);
//     *scale_ptr = (float) i2_scale;
//
//     free(q8);
//
//     // 8) The original code returns: nrow * row_size / 4 + 32
//     //    for alignment. We'll do the same.
//     return nrow * row_size / 4 + 32;
// }
//

// size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
//     // 2 bits per weight
//
//     size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);
//
//     int n = nrow * n_per_row;
//
//     // f32 -> q8
//     double max = 0;
//     for (int i = 0; i < n; ++i) {
//         max = fmax(max, (double)fabs((double)src[i]));
//     }
//     double i2_scale = max;
//
//     uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
//     for (int i=0; i<n; i++) {
//         if (fabs((double)(src[i])) < 1e-6) {
//             q8[i] = 0;
//             continue;
//         }
//         q8[i] = (double)src[i] * i2_scale > 0 ? 1 : 3;
//     }
//
//     memset(dst, 0, n * sizeof(uint8_t) / 4);
//
//     // q8 -> 3, 0, 1
//     //       |  |  |
//     //      -1, 0, 1
//
//     uint8_t* i2_weight = (uint8_t*)dst;
//     for (int i = 0; i < n / QK_I2; i++) {
//         for (int j = 0; j < QK_I2; j++) {
//             int group_idx = j / 32;
//             int group_pos = j % 32;
//             uint8_t temp = (q8[i * QK_I2 + j] << (6 - 2 * group_idx));
//             i2_weight[i * 32 + group_pos] |= temp;
//         }
//     }
//
//     float* scale_ptr = (float*)((char*)i2_weight + n / 4);
//     scale_ptr[0] = i2_scale;
//
//     free(q8);
//
//     // 32B for alignment
//     return nrow * row_size / 4 + 32;
// }

// size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
//     (void) quant_weights; // not used in this implementation
//
//     // Total number of elements to quantize.
//     int64_t n = nrow * n_per_row;
//
//     // 1. Compute the scale factor as the maximum absolute value over src.
//     double max_val = 0.0;
//     for (int64_t i = 0; i < n; i++) {
//         double abs_val = fabs(src[i]);
//         if (abs_val > max_val) {
//             max_val = abs_val;
//         }
//     }
//     double i2_scale = max_val;
//
//     // 2. Create a temporary array to hold the quantized 2-bit codes.
//     // Mapping: if fabs(src[i]) < 1e-6 then code = 0; else if src[i] > 0 then code = 1; else code = 3.
//     uint8_t * q8 = (uint8_t *) malloc(n * sizeof(uint8_t));
//     for (int64_t i = 0; i < n; i++) {
//         float v = src[i];
//         if (fabs(v) < 1e-6f) {
//             q8[i] = 0;
//         } else {
//             q8[i] = (v > 0.0f) ? 1 : 3;
//         }
//     }
//
//     // 3. Pack the 2-bit codes into dst in a linear layout.
//     // Each block of QK_I2_S (128) elements occupies 128 * 2 bits = 256 bits = 32 bytes.
//     uint8_t * i2_weight = (uint8_t *) dst;
//     int64_t nblocks = n / QK_I2_S;  // number of full blocks
//     int64_t remainder = n % QK_I2_S;  // remaining elements (if any)
//
//     // Process each full block.
//     for (int64_t block = 0; block < nblocks; block++) {
//         int64_t base = block * QK_I2_S;   // index offset in q8 for this block
//         uint8_t * out_block = i2_weight + block * 64;  // each block occupies 32 bytes
//         memset(out_block, 0, 64);         // clear the block
//
//         // Pack QK_I2_S values into 32 bytes in sequential order.
//         for (int j = 0; j < QK_I2_S; j++) {
//             // Each byte holds 4 2-bit values.
//             int byte_idx = j / 4;         // which byte in the block
//             int shift = 2 * (j % 4);        // bit offset within that byte: 0,2,4,6
//             uint8_t code = q8[base + j] & 0x03;  // ensure only 2 bits are used
//             out_block[byte_idx] |= (code << shift);
//         }
//     }
//
//     // Process any leftover elements as one additional (partial) block.
//     if (remainder > 0) {
//         int64_t base = nblocks * QK_I2_S;
//         uint8_t * out_block = i2_weight + nblocks * 64;
//         memset(out_block, 0, 64);
//         for (int j = 0; j < remainder; j++) {
//             int byte_idx = j / 4;
//             int shift = 2 * (j % 4);
//             uint8_t code = q8[base + j] & 0x03;
//             out_block[byte_idx] |= (code << shift);
//         }
//         nblocks++;  // count the partial block as a full block for storage
//     }
//
//     // 4. Store the scale factor immediately after the quantized data.
//     // The scale is stored as a 4-byte float.
//     float * scale_ptr = (float *) (i2_weight + nblocks * 64);
//     scale_ptr[0] = (float) i2_scale;
//
//     // 5. Calculate and return the total number of bytes used.
//     size_t total_bytes = nblocks * 64 + sizeof(float);
//
//     free(q8);
//     return total_bytes;
// }


size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    // 2 bits per weight

    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int n = nrow * n_per_row;

    // f32 -> q8
    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 0;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 1 : 3;
    }

    memset(dst, 0, n * sizeof(uint8_t) / 4);

    // q8 -> -1, 0, 1
    //       |  |  |
    //      11, 0, 1

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i = 0; i < n / QK_I2; i++) {
        for (int j = 0; j < QK_I2; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2 + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;

    free(q8);

    // 32B for alignment
    return nrow * row_size / 4 + 32;
}

// void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
//     const uint8_t *    x = (uint8_t *)vx;
//     const int8_t  *    y = (int8_t *)vy;
//
//     const int nb = n / QK_I2_S;
//     const int group32_num = nb / 32;
//     const int la_num = nb % 32;
//     const int groupla_num = nb % 32 != 0 ? 1 : 0;
//
// #if defined(__AVX2__)
//
//     __m256i mask = _mm256_set1_epi8(0x03);
//     __m256i accu = _mm256_setzero_si256();
//
//     for (int i=0; i < group32_num; i++){
//         __m256i accu32 = _mm256_setzero_si256();
//         for (int j=0; j < 32; j++) {
//         // 128 index
//         __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + i * 32 * 32 + j * 32));
//         __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
//         __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
//         __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);
//
//         // each 32 index
//         xq8_3 = _mm256_and_si256(xq8_3, mask);
//         xq8_2 = _mm256_and_si256(xq8_2, mask);
//         xq8_1 = _mm256_and_si256(xq8_1, mask);
//         xq8_0 = _mm256_and_si256(xq8_0, mask);
//
//         // each 32 index
//         __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 0));
//         __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 32));
//         __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 64));
//         __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 96));
//
//         // 128 index accumulation add
//         // split into 32 accumulation block
//         // each block each 128 index accumulated 4index
//         // each index maximum 256
//         // each block maximum 4 * 256
//         // each block accumulation maximum 127 * 256
//         // each 32 group index (128 index in one group) needs cast to int32
//         xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
//         xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
//         xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
//         xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);
//
//         accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
//         accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
//         }
//         accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
//     }
//
//     for (int i = 0; i < groupla_num; i++){
//         __m256i accula = _mm256_setzero_si256();
//         for (int j = 0; j < la_num; j++) {
//         // 128 index
//         __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + group32_num * 32 * 32 + j * 32));
//         __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
//         __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
//         __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);
//
//         // each 32 index
//         xq8_3 = _mm256_and_si256(xq8_3, mask);
//         xq8_2 = _mm256_and_si256(xq8_2, mask);
//         xq8_1 = _mm256_and_si256(xq8_1, mask);
//         xq8_0 = _mm256_and_si256(xq8_0, mask);
//
//         // each 32 index
//         __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 0));
//         __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 32));
//         __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 64));
//         __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 96));
//
//         // 128 index accumulation add
//         // split into 32 accumulation block
//         // each block each 128 index accumulated 4index
//         // each index maximum 256
//         // each block maximum 4 * 256
//         // each block accumulation maximum 127 * 256
//         // each 32 group index (128 index in one group) needs cast to int32
//         xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
//         xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
//         xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
//         xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);
//
//         accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
//         accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
//         }
//         accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
//     }
//     int sumi = hsum_i32_8(accu);
//     *s = (float)sumi;
//
// #elif defined(__ARM_NEON)
//
//     int32x4_t accu_0 = vdupq_n_s32(0);
//     int32x4_t accu_1 = vdupq_n_s32(0);
//     int32x4_t accu_2 = vdupq_n_s32(0);
//     int32x4_t accu_3 = vdupq_n_s32(0);
//     const uint8x16_t mask = vdupq_n_u8(3);
//
//     for (int i=0; i < group32_num; i++) {
//
// #if defined(__ARM_FEATURE_DOTPROD)
//
// #else
//         int16x8_t accu32_0 = vdupq_n_s16(0);
//         int16x8_t accu32_1 = vdupq_n_s16(0);
//         int16x8_t accu32_2 = vdupq_n_s16(0);
//         int16x8_t accu32_3 = vdupq_n_s16(0);
// #endif
//
//         for (int j=0; j < 32; j++) {
//             uint8x16_t xq8_6 = vld1q_u8(x + i * 32 * 32 + j * 32);
//             uint8x16_t xq8_7 = vld1q_u8(x + i * 32 * 32 + j * 32 + 16);
//             uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
//             uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
//             uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
//             uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
//             uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
//             uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);
//
//             int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
//             int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
//             int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
//             int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
//             int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
//             int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
//             int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
//             int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));
//
//             const int8x16_t yq8_0 = vld1q_s8(y + i * 128 * 32 + j * 128 + 0);
//             const int8x16_t yq8_1 = vld1q_s8(y + i * 128 * 32 + j * 128 + 16);
//             const int8x16_t yq8_2 = vld1q_s8(y + i * 128 * 32 + j * 128 + 32);
//             const int8x16_t yq8_3 = vld1q_s8(y + i * 128 * 32 + j * 128 + 48);
//             const int8x16_t yq8_4 = vld1q_s8(y + i * 128 * 32 + j * 128 + 64);
//             const int8x16_t yq8_5 = vld1q_s8(y + i * 128 * 32 + j * 128 + 80);
//             const int8x16_t yq8_6 = vld1q_s8(y + i * 128 * 32 + j * 128 + 96);
//             const int8x16_t yq8_7 = vld1q_s8(y + i * 128 * 32 + j * 128 + 112);
//
// #if defined(__ARM_FEATURE_DOTPROD)
//             accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
//             accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
//             accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
//             accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
//             accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
//             accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
//             accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
//             accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
// #else
//             accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
//             accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
//             accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
//             accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
//             accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
//             accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
//             accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
//             accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
//             accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
//             accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
//             accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
//             accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
//             accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
//             accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
//             accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
//             accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
// #endif
//         }
//
// #if defined(__ARM_FEATURE_DOTPROD)
//
// #else
//         accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accu32_0)));
//         accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accu32_0));
//         accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accu32_1)));
//         accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accu32_1));
//         accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accu32_2)));
//         accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accu32_2));
//         accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accu32_3)));
//         accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accu32_3));
// #endif
//     }
//
//     for (int i = 0; i < groupla_num; i++){
// #if defined(__ARM_FEATURE_DOTPROD)
//
// #else
//         int16x8_t accula_0 = vdupq_n_s16(0);
//         int16x8_t accula_1 = vdupq_n_s16(0);
//         int16x8_t accula_2 = vdupq_n_s16(0);
//         int16x8_t accula_3 = vdupq_n_s16(0);
// #endif
//         for (int j = 0; j < la_num; j++) {
//             uint8x16_t xq8_6 = vld1q_u8(x + group32_num * 32 * 32 + j * 32);
//             uint8x16_t xq8_7 = vld1q_u8(x + group32_num * 32 * 32 + j * 32 + 16);
//             uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
//             uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
//             uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
//             uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
//             uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
//             uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);
//
//             int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
//             int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
//             int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
//             int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
//             int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
//             int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
//             int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
//             int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));
//
//             const int8x16_t yq8_0 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 0);
//             const int8x16_t yq8_1 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 16);
//             const int8x16_t yq8_2 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 32);
//             const int8x16_t yq8_3 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 48);
//             const int8x16_t yq8_4 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 64);
//             const int8x16_t yq8_5 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 80);
//             const int8x16_t yq8_6 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 96);
//             const int8x16_t yq8_7 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 112);
//
// #if defined(__ARM_FEATURE_DOTPROD)
//             accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
//             accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
//             accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
//             accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
//             accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
//             accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
//             accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
//             accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
// #else
//             accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
//             accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
//             accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
//             accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
//             accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
//             accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
//             accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
//             accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
//             accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
//             accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
//             accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
//             accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
//             accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
//             accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
//             accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
//             accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
// #endif
//         }
// #if defined(__ARM_FEATURE_DOTPROD)
//
// #else
//         accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accula_0)));
//         accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accula_0));
//         accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accula_1)));
//         accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accula_1));
//         accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accula_2)));
//         accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accula_2));
//         accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accula_3)));
//         accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accula_3));
// #endif
//     }
//     accu_0 = vaddq_s32(accu_0, accu_1);
//     accu_2 = vaddq_s32(accu_2, accu_3);
//     accu_0 = vaddq_s32(accu_0, accu_2);
//     int sumi = vaddlvq_s32(accu_0);
//     *s = (float)sumi;
//
// #endif
// }


#define QK_I2_S 128  // typical from the original code

static inline uint8_t top_bits_of_byte(uint8_t byte)    { return (byte >> 6) & 0x03; }
static inline uint8_t upper_mid_bits(uint8_t byte)      { return (byte >> 4) & 0x03; }
static inline uint8_t lower_mid_bits(uint8_t byte)      { return (byte >> 2) & 0x03; }
static inline uint8_t bottom_bits_of_byte(uint8_t byte) { return (byte >> 0) & 0x03; }

/// Scalar version that processes one 128-element block of i2 in `x_block`
/// against one 128-byte block of i8 in `y_block`.
/// Each block of x has 32 bytes, each byte containing 4 x_i2 values.
/// The block of y has 128 bytes, split into 4 segments of 32 bytes each.
/// Return the dot-product sum for this block.
// static long long dot_block_i2_i8(const uint8_t * x_block, const int8_t * y_block) {
//     long long sum = 0;
//
//     // x_block has 32 bytes => each byte yields 4 “2-bit” values => total 128 i2.
//     // y_block has 128 bytes => split into 4 segments of 32 each.
//     //   segment 0 => offsets [0..31]
//     //   segment 1 => offsets [32..63]
//     //   segment 2 => offsets [64..95]
//     //   segment 3 => offsets [96..127]
//
//     for (int b = 0; b < 32; b++) {
//         uint8_t xb = x_block[b];  // read 1 byte => bits [7..0]
//
//         // Extract 4 slices of 2 bits each
//         int x0 = top_bits_of_byte   (xb);  // bits [6..7]
//         int x1 = upper_mid_bits     (xb);  // bits [4..5]
//         int x2 = lower_mid_bits     (xb);  // bits [2..3]
//         int x3 = bottom_bits_of_byte(xb);  // bits [0..1]
//
//         // Multiply each slice by correct chunk of y:
//         // top bits  =>  y_block[  0..31 ]
//         // next bits =>  y_block[ 32..63 ]
//         // next bits =>  y_block[ 64..95 ]
//         // bottom    =>  y_block[ 96..127]
//         sum += (long long) x0 * y_block[    b ]; // offset b in first 32
//         sum += (long long) x1 * y_block[32 + b ];
//         sum += (long long) x2 * y_block[64 + b ];
//         sum += (long long) x3 * y_block[96 + b];
//     }
//     return sum;
// }
//
// void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs,
//                           const void * vx, size_t bx,
//                           const void * vy, size_t by,
//                           int nrc)
// {
//     // If your real code needs to use bx, by, etc. for offsets, do so here:
//     const uint8_t * x = (const uint8_t *) vx;
//     const int8_t  * y = (const int8_t  *) vy;
//
//     // Each block is 128 “i2” values => 32 bytes in x, 128 bytes in y
//     int nb = n / QK_I2_S;  // how many 128-element blocks
//     // The leftover (if n not multiple of 128) is typically ignored or separate?
//     // Original code does "const int nb = n / QK_I2_S;"
//     // ignoring any remainder. We'll match that.
//
//     int group32_num = nb / 32;   // how many full groups
//     int la_num      = nb % 32;   // leftover sub-blocks
//     int groupla_num = (la_num != 0) ? 1 : 0;
//
//     long long sum = 0;  // 64-bit accumulation
//
//     // 1. Full groups
//     for (int i = 0; i < group32_num; i++) {
//         for (int j = 0; j < 32; j++) {
//             // Offsets just like the AVX2 code
//             int x_off = (i*32 + j)*32;    // i*(32*32) + j*32
//             int y_off = (i*32 + j)*128;   // i*(128*32) + j*128
//             sum += dot_block_i2_i8(&x[x_off], &y[y_off]);
//         }
//     }
//
//     // 2. Leftover group if any
//     //   The original code does: "for (int i = 0; i < groupla_num; i++) { ... }"
//     //   but that i will be only 0 or 1, because groupla_num is either 0 or 1.
//     //   Inside, we do j < la_num (the leftover # of sub-blocks).
//     for (int i = 0; i < groupla_num; i++) {
//         for (int j = 0; j < la_num; j++) {
//             int x_off = ((group32_num + i)*32 + j)*32;
//             int y_off = ((group32_num + i)*32 + j)*128;
//             sum += dot_block_i2_i8(&x[x_off], &y[y_off]);
//         }
//     }
//
//     // store final
//     *s = (float) sum;
// }

/// Scalar version that processes one 128-element block of i2 in `x_block`
/// against one 128-byte block of i8 in `y_block`.
/// Each block of x has 32 bytes, each byte containing 4 x_i2 values.
/// The block of y has 128 bytes, split into 4 segments of 32 bytes each.
/// Return the dot-product sum for this block.


// int decode_2bit(int code) {
//     // e.g. 00->-1, 01->0, 10->+1, 11->0, etc.
//     switch (code) {
//         case 0: return -1;
//         case 1: return  0;
//         case 2: return +1;
//         default: return 0;
//     }
// }

int decode_2bit(int code) {
    // e.g. 00->-1, 01->0, 10->+1, 11->0, etc.
    switch (code) {
        case 0: return 0;
        case 1: return  1;
        case 3: return -1;
        default: return 0;
    }
}

// static long long dot_block_i2_i8(const uint8_t * x_block, const int8_t * y_block) {
//     long long sum = 0;
//
//     // x_block has 32 bytes => each byte yields 4 “2-bit” values => total 128 i2.
//     // y_block has 128 bytes => split into 4 segments of 32 each.
//     //   segment 0 => offsets [0..31]
//     //   segment 1 => offsets [32..63]
//     //   segment 2 => offsets [64..95]
//     //   segment 3 => offsets [96..127]
//
//     for (int b = 0; b < 32; b++) {
//         uint8_t xb = x_block[b];  // read 1 byte => bits [7..0]
//
//         // Extract 4 slices of 2 bits each
//         int x0 = decode_2bit(top_bits_of_byte   (xb));  // bits [6..7]
//         int x1 = decode_2bit(upper_mid_bits     (xb));  // bits [4..5]
//         int x2 = decode_2bit(lower_mid_bits     (xb));  // bits [2..3]
//         int x3 = decode_2bit(bottom_bits_of_byte(xb));  // bits [0..1]
//
//         // Multiply each slice by correct chunk of y:
//         // top bits  =>  y_block[  0..31 ]
//         // next bits =>  y_block[ 32..63 ]
//         // next bits =>  y_block[ 64..95 ]
//         // bottom    =>  y_block[ 96..127]
//         sum += (long long) x0 * y_block[    b ]; // offset b in first 32
//         sum += (long long) x1 * y_block[32 + b ];
//         sum += (long long) x2 * y_block[64 + b ];
//         sum += (long long) x3 * y_block[96 + b];
//     }
//     return sum;
// }

static long long dot_block_i2_i8(const uint8_t * x_block, const int8_t * y_block) {
    long long sum = 0;
    // Iterate over 128 elements
    for (int i = 0; i < 256; i++) {
        int byte_idx = i / 4;       // which byte holds the i-th 2-bit value
        int shift = 2 * (i % 4);      // position in the byte
        // Extract 2 bits:
        uint8_t code = (x_block[byte_idx] >> shift) & 0x03;
        int x_val = decode_2bit(code);  // same decoding: 0->0, 1->+1, 3->-1
        sum += (long long) x_val * (long long) y_block[i];
    }
    return sum;
}

// // Helper: Unpack 8 bytes of 2-bit-packed values into 32 int8 values.
// // (Each 2-bit value is stored in the lower bits of a byte.)
// int dot_block_i2_i8_avx2_vectorized_unpack(const uint8_t *x_block, const int8_t *y_block) {
//     __m256i accum = _mm256_setzero_si256();
//
//     // Shuffle masks for extracting a specific byte from a 32–bit word in a 128–bit lane.
//     // Each mask will extract the kth byte (0,1,2, or 3) from each 32–bit word.
//     const __m128i shuf_mask0 = _mm_setr_epi8(0,  4,  8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
//     const __m128i shuf_mask1 = _mm_setr_epi8(1,  5,  9, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
//     const __m128i shuf_mask2 = _mm_setr_epi8(2,  6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
//     const __m128i shuf_mask3 = _mm_setr_epi8(3,  7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
//
//     // Process 16 packed x bytes at a time.
//     // 16 x bytes → 16*4 = 64 decoded values, and 64 y bytes (16 int32_t’s) correspond.
//     for (int i = 0; i < 4; i++) {
//         // --- Decode 16 x bytes ---
//         __m128i x_bytes = _mm_loadu_si128((const __m128i*)(x_block + i * 16));
//         __m256i x_vals  = _mm256_cvtepu8_epi16(x_bytes);
//         const __m256i mask3_16 = _mm256_set1_epi16(0x0003);
//         // Extract the four 2–bit fields from each x byte:
//         __m256i field0 = _mm256_and_si256(x_vals, mask3_16);                      // bits [1:0]
//         __m256i field1 = _mm256_and_si256(_mm256_srli_epi16(x_vals, 2), mask3_16);   // bits [3:2]
//         __m256i field2 = _mm256_and_si256(_mm256_srli_epi16(x_vals, 4), mask3_16);   // bits [5:4]
//         __m256i field3 = _mm256_and_si256(_mm256_srli_epi16(x_vals, 6), mask3_16);   // bits [7:6]
//
//         // --- Load and de–interleave 64 y bytes ---
//         // For these 16 x bytes, there are 16 groups of 4 y bytes = 64 bytes.
//         int y_offset = i * 16 * 4;
//         // Load 64 bytes using two 256–bit loads (32 bytes each)
//         __m256i y256a = _mm256_loadu_si256((const __m256i*)(y_block + y_offset));      // first 32 bytes (8 int32_t’s)
//         __m256i y256b = _mm256_loadu_si256((const __m256i*)(y_block + y_offset + 32));   // next 32 bytes (8 int32_t’s)
//
//         // Split each 256–bit register into two 128–bit lanes.
//         __m128i lane0 = _mm256_castsi256_si128(y256a);             // int32_t indices 0..3
//         __m128i lane1 = _mm256_extracti128_si256(y256a, 1);          // int32_t indices 4..7
//         __m128i lane2 = _mm256_castsi256_si128(y256b);             // int32_t indices 8..11
//         __m128i lane3 = _mm256_extracti128_si256(y256b, 1);          // int32_t indices 12..15
//
//         // Helper macro: de–interleave one field from the 4 lanes.
//         #define DEINTERLEAVE_FIELD(mask) ( \
//             _mm_unpacklo_epi64( \
//                 _mm_unpacklo_epi64(_mm_shuffle_epi8(lane0, mask), _mm_shuffle_epi8(lane1, mask)), \
//                 _mm_unpacklo_epi64(_mm_shuffle_epi8(lane2, mask), _mm_shuffle_epi8(lane3, mask)) \
//             ))
//
//         __m128i y0_128 = DEINTERLEAVE_FIELD(shuf_mask0);  // Field 0: first byte from each 32–bit word
//         __m128i y1_128 = DEINTERLEAVE_FIELD(shuf_mask1);  // Field 1: second byte
//         __m128i y2_128 = DEINTERLEAVE_FIELD(shuf_mask2);  // Field 2: third byte
//         __m128i y3_128 = DEINTERLEAVE_FIELD(shuf_mask3);  // Field 3: fourth byte
//         #undef DEINTERLEAVE_FIELD
//
//         // Extend the 16 decoded y bytes (per field) to 16–bit values.
//         __m256i y0_vec = _mm256_cvtepi8_epi16(y0_128);
//         __m256i y1_vec = _mm256_cvtepi8_epi16(y1_128);
//         __m256i y2_vec = _mm256_cvtepi8_epi16(y2_128);
//         __m256i y3_vec = _mm256_cvtepi8_epi16(y3_128);
//
//         // --- Compute contributions ---
//         // For each field, add y when field equals 01 and subtract y when field equals 11.
//         __m256i ones_16   = _mm256_set1_epi16(1);
//         __m256i threes_16 = _mm256_set1_epi16(3);
//         __m256i pos_mask0 = _mm256_cmpeq_epi16(field0, ones_16);
//         __m256i neg_mask0 = _mm256_cmpeq_epi16(field0, threes_16);
//         __m256i pos_mask1 = _mm256_cmpeq_epi16(field1, ones_16);
//         __m256i neg_mask1 = _mm256_cmpeq_epi16(field1, threes_16);
//         __m256i pos_mask2 = _mm256_cmpeq_epi16(field2, ones_16);
//         __m256i neg_mask2 = _mm256_cmpeq_epi16(field2, threes_16);
//         __m256i pos_mask3 = _mm256_cmpeq_epi16(field3, ones_16);
//         __m256i neg_mask3 = _mm256_cmpeq_epi16(field3, threes_16);
//
//         // For negative entries, compute -y via XOR–inversion: -y = ~y + 1.
//         __m256i minus_one_16 = _mm256_set1_epi16(0xFFFF);
//         __m256i one_16       = _mm256_set1_epi16(1);
//         __m256i neg_y0 = _mm256_add_epi16(_mm256_xor_si256(y0_vec, minus_one_16), one_16);
//         __m256i neg_y1 = _mm256_add_epi16(_mm256_xor_si256(y1_vec, minus_one_16), one_16);
//         __m256i neg_y2 = _mm256_add_epi16(_mm256_xor_si256(y2_vec, minus_one_16), one_16);
//         __m256i neg_y3 = _mm256_add_epi16(_mm256_xor_si256(y3_vec, minus_one_16), one_16);
//
//         // Select contributions: y if field == 01, and -y if field == 11.
//         __m256i contrib0 = _mm256_add_epi16(_mm256_and_si256(y0_vec, pos_mask0),
//                                             _mm256_and_si256(neg_y0, neg_mask0));
//         __m256i contrib1 = _mm256_add_epi16(_mm256_and_si256(y1_vec, pos_mask1),
//                                             _mm256_and_si256(neg_y1, neg_mask1));
//         __m256i contrib2 = _mm256_add_epi16(_mm256_and_si256(y2_vec, pos_mask2),
//                                             _mm256_and_si256(neg_y2, neg_mask2));
//         __m256i contrib3 = _mm256_add_epi16(_mm256_and_si256(y3_vec, pos_mask3),
//                                             _mm256_and_si256(neg_y3, neg_mask3));
//
//         __m256i sum_fields = _mm256_add_epi16(_mm256_add_epi16(contrib0, contrib1),
//                                               _mm256_add_epi16(contrib2, contrib3));
//
//         // Widen to 32–bit and add into the overall accumulator.
//         __m256i sum32_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum_fields));
//         __m256i sum32_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_fields, 1));
//         __m256i partial  = _mm256_add_epi32(sum32_lo, sum32_hi);
//         accum = _mm256_add_epi32(accum, partial);
//     }
//
//     // Horizontal sum of the 8 32–bit lanes in 'accum'.
//     __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(accum),
//                                    _mm256_extracti128_si256(accum, 1));
//     sum128 = _mm_hadd_epi32(sum128, sum128);
//     sum128 = _mm_hadd_epi32(sum128, sum128);
//     return _mm_cvtsi128_si32(sum128);
// }


// Make sure to include BMI2 intrinsics (e.g. -mbmi2 compiler flag may be needed)
#if !defined(__BMI2__)
#error "BMI2 support is required for _pext_u32."
#endif

// Dot product of 256 int8_t y values with a 2‑bit–packed x vector.
// x_block contains 64 bytes (256 2‑bit values) where each 2‑bit field is encoded as:
//    00 → 0, 01 → +1, 11 → –1    (note: code 10 is impossible).
// y_block contains 256 int8_t values.
// This implementation processes x in two halves of 32 bytes each (each half yields 128 decoded values)
// and processes y in chunks of 32 bytes (the optimal width for AVX2 registers).
int dot_block_i2_i8_avx2_bmi2(const uint8_t *x_block, const int8_t *y_block) {
    int total_sum = 0;

    // There are 64 x-packed bytes total. Process 32 bytes at a time.
    for (int half = 0; half < 2; half++) {
        // Pointer to one half (32 bytes) of x-packed data.
        // These 32 bytes hold 32*4 = 128 decoded x values.
        const uint32_t *x_words = (const uint32_t *)(x_block + half * 32);
        // Temporary buffer for 128 decoded values.
        // We will store each decoded value as an int8_t: 0, 1, or -1.
        alignas(32) uint8_t decoded[128];
        int dec_index = 0;
        // Each 32-byte block can be viewed as 8 32‑bit words.
        // Each 32‑bit word contains 4 bytes, i.e. 4*8 = 32 bits,
        // and since each 2‑bit field encodes one value, each word holds 32/2 = 16 decoded values.
        for (int i = 0; i < 8; i++) {
            uint32_t word = x_words[i];
            // Use _pext_u32 to extract the absolute bits and sign bits.
            // The mask 0x55555555 (binary 0101…0101) extracts bits at positions 0,2,4,…,30.
            uint32_t abs_bits = _pext_u32(word, 0x55555555);   // 16 bits (one per decoded value)
            // The mask 0xAAAAAAAA (binary 1010…1010) extracts bits at positions 1,3,5,…,31.
            uint32_t sign_bits = _pext_u32(word, 0xAAAAAAAA);  // 16 bits
            // For each of the 16 decoded values in this word:
            for (int j = 0; j < 16; j++) {
                // If the absolute bit is 0, then the decoded value is 0.
                // Otherwise, if absolute bit is 1 then check the corresponding sign bit:
                //   if sign bit is 0 → +1, if sign bit is 1 → -1.
                uint8_t abs_bit = (abs_bits >> j) & 1;
                if (abs_bit == 0) {
                    decoded[dec_index++] = 0;
                } else {
                    uint8_t sign_bit = (sign_bits >> j) & 1;
                    decoded[dec_index++] = sign_bit ? (uint8_t)(-1) : 1;
                }
            }
        }
        // Now, 'decoded' holds 128 decoded int8_t values for this half.
        // The corresponding 128 y values are in y_block, starting at offset half*128.
        int y_offset = half * 128;
        int half_sum = 0;
        // Process y (and corresponding decoded x) 32 values at a time.
        for (int i = 0; i < 128; i += 32) {
            // Load 32 y values (int8_t) into a 256-bit register.
            __m256i y_vec = _mm256_loadu_si256((const __m256i *)(y_block + y_offset + i));
            // Load 32 decoded x values.
            __m256i x_dec = _mm256_loadu_si256((const __m256i *)(decoded + i));
            // Since we are doing 8-bit multiplications and accumulating,
            // widen the 8-bit values to 16-bit to avoid overflow.
            __m256i y_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(y_vec));
            __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x_dec));
            __m256i prod_lo = _mm256_mullo_epi16(x_lo, y_lo);
            __m256i y_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(y_vec, 1));
            __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x_dec, 1));
            __m256i prod_hi = _mm256_mullo_epi16(x_hi, y_hi);
            __m256i prod16 = _mm256_add_epi16(prod_lo, prod_hi);
            // Now widen to 32-bit and sum all 16-bit values.
            __m256i sum32_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod16));
            __m256i sum32_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod16, 1));
            __m256i sum32 = _mm256_add_epi32(sum32_lo, sum32_hi);
            // Horizontally add the 8 32-bit integers in sum32.
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum32),
                                           _mm256_extracti128_si256(sum32, 1));
            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum128 = _mm_hadd_epi32(sum128, sum128);
            half_sum += _mm_cvtsi128_si32(sum128);
        }
        total_sum += half_sum;
    }
    return total_sum;
}

static const uint64_t expand_byte_lookup[256] = {
    // 0x00–0x0F:
    0x0000000000000000ULL, 0x00000000000000FFULL, 0x000000000000FF00ULL, 0x000000000000FFFFULL,
    0x0000000000FF0000ULL, 0x0000000000FF00FFULL, 0x0000000000FFFF00ULL, 0x0000000000FFFFFFFFULL,
    0x00000000FF000000ULL, 0x00000000FF0000FFULL, 0x00000000FF00FF00ULL, 0x00000000FF00FFFFULL,
    0x00000000FFFF0000ULL, 0x00000000FFFF00FFULL, 0x00000000FFFFFF00ULL, 0x00000000FFFFFFFFULL,

    // 0x10–0x1F:
    0x000000FF00000000ULL, 0x000000FF000000FFULL, 0x000000FF0000FF00ULL, 0x000000FF0000FFFFULL,
    0x000000FF00FF0000ULL, 0x000000FF00FF00FFULL, 0x000000FF00FFFF00ULL, 0x000000FF00FFFFFFULL,
    0x000000FFFF000000ULL, 0x000000FFFF0000FFULL, 0x000000FFFF00FF00ULL, 0x000000FFFF00FFFFULL,
    0x000000FFFFFFFF00ULL, 0x000000FFFFFFFFFFULL, 0x000000FFFFFF0000ULL, 0x000000FFFFFF00FFULL,

    // 0x20–0x2F:
    0x0000FF0000000000ULL, 0x0000FF00000000FFULL, 0x0000FF000000FF00ULL, 0x0000FF000000FFFFULL,
    0x0000FF0000FF0000ULL, 0x0000FF0000FF00FFULL, 0x0000FF0000FFFF00ULL, 0x0000FF0000FFFFFFULL,
    0x0000FF00FF000000ULL, 0x0000FF00FF0000FFULL, 0x0000FF00FF00FF00ULL, 0x0000FF00FF00FFFFULL,
    0x0000FF00FFFF0000ULL, 0x0000FF00FFFF00FFULL, 0x0000FF00FFFFFF00ULL, 0x0000FF00FFFFFFFFULL,

    // 0x30–0x3F:
    0x0000FFFF00000000ULL, 0x0000FFFF000000FFULL, 0x0000FFFF0000FF00ULL, 0x0000FFFF0000FFFFULL,
    0x0000FFFF00FF0000ULL, 0x0000FFFF00FF00FFULL, 0x0000FFFF00FFFF00ULL, 0x0000FFFF00FFFFFFULL,
    0x0000FFFFFFFF0000ULL, 0x0000FFFFFFFF00FFULL, 0x0000FFFFFFFFFF00ULL, 0x0000FFFFFFFFFFFFULL,
    0x0000FFFFFF000000ULL, 0x0000FFFFFF0000FFULL, 0x0000FFFFFF00FF00ULL, 0x0000FFFFFF00FFFFULL,

    // 0x40–0x4F:
    0x00FF000000000000ULL, 0x00FF0000000000FFULL, 0x00FF000000000FF00ULL, 0x00FF000000000FFFFULL,
    0x00FF00000000FF0000ULL, 0x00FF00000000FF00FFULL, 0x00FF00000000FFFF00ULL, 0x00FF00000000FFFFFFULL,
    0x00FF0000FF000000ULL, 0x00FF0000FF0000FFULL, 0x00FF0000FF00FF00ULL, 0x00FF0000FF00FFFFULL,
    0x00FF0000FFFF0000ULL, 0x00FF0000FFFF00FFULL, 0x00FF0000FFFFFF00ULL, 0x00FF0000FFFFFFFFULL,

    // 0x50–0x5F:
    0x00FF00FF00000000ULL, 0x00FF00FF000000FFULL, 0x00FF00FF0000FF00ULL, 0x00FF00FF0000FFFFULL,
    0x00FF00FFFF000000ULL, 0x00FF00FFFF0000FFULL, 0x00FF00FFFF00FF00ULL, 0x00FF00FFFF00FFFFULL,
    0x00FF00FFFFFF0000ULL, 0x00FF00FFFFFF00FFULL, 0x00FF00FFFFFFFF00ULL, 0x00FF00FFFFFFFFFFULL,
    0x00FFFF0000000000ULL, 0x00FFFF00000000FFULL, 0x00FFFF000000FF00ULL, 0x00FFFF000000FFFFULL,

    // 0x60–0x6F:
    0x00FFFF0000FF0000ULL, 0x00FFFF0000FF00FFULL, 0x00FFFF0000FFFF00ULL, 0x00FFFF0000FFFFFFULL,
    0x00FFFF00FF000000ULL, 0x00FFFF00FF0000FFULL, 0x00FFFF00FF00FF00ULL, 0x00FFFF00FF00FFFFULL,
    0x00FFFF00FFFF0000ULL, 0x00FFFF00FFFF00FFULL, 0x00FFFF00FFFFFF00ULL, 0x00FFFF00FFFFFFFFULL,
    0x00FFFFFF00000000ULL, 0x00FFFFFF000000FFULL, 0x00FFFFFF0000FF00ULL, 0x00FFFFFF0000FFFFULL,

    // 0x70–0x7F:
    0x00FFFFFF00FF0000ULL, 0x00FFFFFF00FF00FFULL, 0x00FFFFFF00FFFF00ULL, 0x00FFFFFF00FFFFFFULL,
    0x00FFFFFFFF000000ULL, 0x00FFFFFFFF000000FFULL, 0x00FFFFFFFF0000FF00ULL, 0x00FFFFFFFF0000FFFFULL,
    0x00FFFFFFFF00FF0000ULL, 0x00FFFFFFFF00FF00FFULL, 0x00FFFFFFFF00FFFF00ULL, 0x00FFFFFFFF00FFFFFFULL,
    0x00FFFFFFFFFF000000ULL, 0x00FFFFFFFFFF0000FFULL, 0x00FFFFFFFFFF00FF00ULL, 0x00FFFFFFFFFF00FFFFULL,

    // 0x80–0x8F:
    0xFF00000000000000ULL, 0xFF000000000000FFULL, 0xFF0000000000FF00ULL, 0xFF0000000000FFFFULL,
    0xFF00000000FF0000ULL, 0xFF00000000FF00FFULL, 0xFF00000000FFFF00ULL, 0xFF00000000FFFFFFULL,
    0xFF000000FF000000ULL, 0xFF000000FF0000FFULL, 0xFF000000FF00FF00ULL, 0xFF000000FF00FFFFULL,
    0xFF000000FFFF0000ULL, 0xFF000000FFFF00FFULL, 0xFF000000FFFFFF00ULL, 0xFF000000FFFFFFFFULL,

    // 0x90–0x9F:
    0xFF0000FF00000000ULL, 0xFF0000FF000000FFULL, 0xFF0000FF0000FF00ULL, 0xFF0000FF0000FFFFULL,
    0xFF0000FF00FF0000ULL, 0xFF0000FF00FF00FFULL, 0xFF0000FF00FFFF00ULL, 0xFF0000FF00FFFFFFULL,
    0xFF0000FFFF000000ULL, 0xFF0000FFFF0000FFULL, 0xFF0000FFFF00FF00ULL, 0xFF0000FFFF00FFFFULL,
    0xFF0000FFFFFFFF00ULL, 0xFF0000FFFFFFFFFFULL, 0xFF0000FFFFFF0000ULL, 0xFF0000FFFFFF00FFULL,

    // 0xA0–0xAF:
    0xFF00FF0000000000ULL, 0xFF00FF00000000FFULL, 0xFF00FF000000FF00ULL, 0xFF00FF000000FFFFULL,
    0xFF00FF0000FF0000ULL, 0xFF00FF0000FF00FFULL, 0xFF00FF0000FFFF00ULL, 0xFF00FF0000FFFFFFULL,
    0xFF00FFFF00000000ULL, 0xFF00FFFF000000FFULL, 0xFF00FFFF0000FF00ULL, 0xFF00FFFF0000FFFFULL,
    0xFF00FFFFFFFF0000ULL, 0xFF00FFFFFFFF00FFULL, 0xFF00FFFFFFFFFF00ULL, 0xFF00FFFFFFFFFFFFULL,

    // 0xB0–0xBF (corrected):
    0xFF00FFFF00000000ULL, 0xFF00FFFF000000FFULL, 0xFF00FFFF0000FF00ULL, 0xFF00FFFF0000FFFFULL,
    0xFF00FFFF00FF0000ULL, 0xFF00FFFF00FF00FFULL, 0xFF00FFFF00FFFF00ULL, 0xFF00FFFF00FFFFFFULL,
    0xFF00FFFFFF000000ULL, 0xFF00FFFFFF0000FFULL, 0xFF00FFFFFF000FF00ULL, 0xFF00FFFFFF00FFFFULL,
    0xFF00FFFFFFFF0000ULL, 0xFF00FFFFFFFF00FFULL, 0xFF00FFFFFFFFFF00ULL, 0xFF00FFFFFFFFFFFFULL,

    // 0xC0–0xCF:
    0xFFFF000000000000ULL, 0xFFFF0000000000FFULL, 0xFFFF000000000FF00ULL, 0xFFFF000000000FFFFULL,
    0xFFFF000000FF0000ULL, 0xFFFF000000FF00FFULL, 0xFFFF000000FFFF00ULL, 0xFFFF000000FFFFFFULL,
    0xFFFF0000FF000000ULL, 0xFFFF0000FF0000FFULL, 0xFFFF0000FF00FF00ULL, 0xFFFF0000FF00FFFFULL,
    0xFFFF0000FFFF0000ULL, 0xFFFF0000FFFF00FFULL, 0xFFFF0000FFFFFF00ULL, 0xFFFF0000FFFFFFFFULL,

    // 0xD0–0xDF:
    0xFFFF00FF00000000ULL, 0xFFFF00FF000000FFULL, 0xFFFF00FF0000FF00ULL, 0xFFFF00FF0000FFFFULL,
    0xFFFF00FF00FF0000ULL, 0xFFFF00FF00FF00FFULL, 0xFFFF00FF00FFFF00ULL, 0xFFFF00FF00FFFFFFULL,
    0xFFFF00FFFF000000ULL, 0xFFFF00FFFF0000FFULL, 0xFFFF00FFFF00FF00ULL, 0xFFFF00FFFF00FFFFULL,
    0xFFFF00FFFFFFFF00ULL, 0xFFFF00FFFFFFFFFFULL, 0xFFFF00FFFFFF0000ULL, 0xFFFF00FFFFFF00FFULL,

    // 0xE0–0xEF:
    0xFFFFFFFF00000000ULL, 0xFFFFFFFF000000FFULL, 0xFFFFFFFF0000FF00ULL, 0xFFFFFFFF0000FFFFULL,
    0xFFFFFFFF00FF0000ULL, 0xFFFFFFFF00FF00FFULL, 0xFFFFFFFF00FFFF00ULL, 0xFFFFFFFF00FFFFFFULL,
    0xFFFFFFFFFF000000ULL, 0xFFFFFFFFFF0000FFULL, 0xFFFFFFFFFF00FF00ULL, 0xFFFFFFFFFF00FFFFULL,
    0xFFFFFFFFFFFFFF00ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,

    // 0xF0–0xFF:
    0xFFFFFFFF00000000ULL, 0xFFFFFFFF000000FFULL, 0xFFFFFFFF0000FF00ULL, 0xFFFFFFFF0000FFFFULL,
    0xFFFFFFFF00FF0000ULL, 0xFFFFFFFF00FF00FFULL, 0xFFFFFFFF00FFFF00ULL, 0xFFFFFFFF00FFFFFFULL,
    0xFFFFFFFFFF000000ULL, 0xFFFFFFFFFF0000FFULL, 0xFFFFFFFFFF00FF00ULL, 0xFFFFFFFFFF00FFFFULL,
    0xFFFFFFFFFFFFFF00ULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};


static inline uint64_t expandByte(uint8_t m) {
    return expand_byte_lookup[m];
}


//
// static inline uint64_t expandByte(uint8_t m) {
//     // _pdep_u64 deposits the bits of m (from LSB upward) into the bit–positions where dep_mask has 1's.
//     // Here dep_mask = 0x0101010101010101ULL has a 1 in the LSB of every byte.
//     // Then multiplying by 0xFF replicates that bit to fill the entire byte.
//     return _pdep_u64((uint64_t)m, 0x0101010101010101ULL) * 0xFFULL;
// }
//
// // Fast expansion of a 32-bit mask X into a 256-bit vector (32 bytes).
// // The 32-bit mask X is split into 4 bytes; each byte is expanded to 8 bytes using expandByte().
// // The resulting 256-bit vector has its first 8 bytes corresponding to bits 0–7 of X,
// // next 8 bytes to bits 8–15, etc.
// __m256i expandMask32To256(uint32_t X) {
//     uint64_t part0 = expandByte((uint8_t)(X       & 0xFF));
//     uint64_t part1 = expandByte((uint8_t)((X >> 8)  & 0xFF));
//     uint64_t part2 = expandByte((uint8_t)((X >> 16) & 0xFF));
//     uint64_t part3 = expandByte((uint8_t)((X >> 24) & 0xFF));
//     return _mm256_setr_epi64x(part0, part1, part2, part3);
// }

// Now, instead of shifting X to extract each byte,
// we simply read its bytes from memory.

// Expands a 32-bit mask into a 256-bit vector by expanding each of its 4 bytes.
//
// __m256i expandMask32To256(uint32_t X) {
//     const uint8_t *p = reinterpret_cast<const uint8_t*>(&X);
//     uint64_t part0 = expandByte(p[0]);
//     uint64_t part1 = expandByte(p[1]);
//     uint64_t part2 = expandByte(p[2]);
//     uint64_t part3 = expandByte(p[3]);
//     return _mm256_setr_epi64x(part0, part1, part2, part3);
// }
// int horizontal_sum_int8_avx2(__m256i v) {
//     // Step 1: Pairwise sum adjacent int8 values into int16 values.
//     const __m256i ones8 = _mm256_set1_epi8(1);
//     __m256i sum16 = _mm256_maddubs_epi16(ones8, v);  // Now each int16 element is (v[2*i] + v[2*i+1])
//
//     // Step 2: Sum adjacent int16 values into int32 values.
//     const __m256i ones16 = _mm256_set1_epi16(1);
//     __m256i sum32 = _mm256_madd_epi16(ones16, sum16);  // Now each int32 element is the sum of 4 original int8 values.
//
//     // Step 3: Horizontally add the eight int32 values.
//     __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum32),
//                                    _mm256_extracti128_si256(sum32, 1));
//     sum128 = _mm_hadd_epi32(sum128, sum128);
//     sum128 = _mm_hadd_epi32(sum128, sum128);
//     return _mm_cvtsi128_si32(sum128);
// }


// Extract odd-indexed bits (positions 1, 3, …, 63) from a 64-bit word.


// static inline __m256i expandByte_AVX2(uint8_t b) {
//     __m256i mask = _mm256_set1_epi8(b);
//     __m256i bit_mask = _mm256_setr_epi8(
//         0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
//         0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
//         0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80,
//         0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, (char)0x80
//     );
//     return _mm256_cmpeq_epi8(_mm256_and_si256(mask, bit_mask), bit_mask);
// }



__m256i expandMask32To256(uint32_t X) {
    // Step 1: Load the 4 mask bytes into the low 32 bits of a 128-bit register.
    __m128i low = _mm_cvtsi32_si128(X); // holds our 4 bytes in the lowest 32 bits

    // Expand the 4 8-bit values into 4 64-bit lanes.
    // Note: _mm256_cvtepu8_epi64 is available on newer toolchains.
    __m256i mask4 = _mm256_cvtepu8_epi64(low);
    // Now each 64-bit lane has its mask byte in the lowest byte (other bytes are zero).

    // Step 2: Broadcast each 64-bit lane's lowest byte to every byte in that lane.
    // The control vector replicates the low byte of each 64-bit lane.
    const __m256i broadcastControl = _mm256_setr_epi8(
         0,  0,  0,  0,  0,  0,  0,  0,   // first 64-bit lane (mask byte from offset 0)
         8,  8,  8,  8,  8,  8,  8,  8,   // second lane (mask byte from offset 8)
        16, 16, 16, 16, 16, 16, 16, 16,   // third lane (mask byte from offset 16)
        24, 24, 24, 24, 24, 24, 24, 24    // fourth lane (mask byte from offset 24)
    );
    __m256i broadcastMask = _mm256_shuffle_epi8(mask4, broadcastControl);

    // Step 3: Prepare a constant vector with bit-weights in every 64-bit lane.
    // Each lane: {1, 2, 4, 8, 16, 32, 64, 128}
    const __m256i bitWeights = _mm256_setr_epi8(
         1,   2,   4,    8,  16,  32,  64, 128,
         1,   2,   4,    8,  16,  32,  64, 128,
         1,   2,   4,    8,  16,  32,  64, 128,
         1,   2,   4,    8,  16,  32,  64, 128
    );

    // Step 4: Isolate each bit by ANDing the broadcasted mask with the bitWeights.
    __m256i bits = _mm256_and_si256(broadcastMask, bitWeights);

    // Step 5: Compare the result with bitWeights.
    // If a particular bit was set in the original mask, then (mask & bitWeight) equals bitWeight.
    // The compare returns 0xFF where equal and 0x00 otherwise.
    __m256i expanded = _mm256_cmpeq_epi8(bits, bitWeights);

    return expanded;
}


// Horizontally sum 32 int8 values stored in a 256-bit vector.
// The offset trick avoids issues with signed values in _mm256_sad_epu8.
// int horizontal_sum_int8_avx2(__m256i v) {
//     __m256i offset = _mm256_set1_epi8(128);
//     __m256i v_offset = _mm256_add_epi8(v, offset);
//     __m256i sum256 = _mm256_sad_epu8(v_offset, _mm256_setzero_si256());
//     __m128i sum128_lo = _mm256_castsi256_si128(sum256);
//     __m128i sum128_hi = _mm256_extracti128_si256(sum256, 1);
//     __m128i sum128 = _mm_add_epi64(sum128_lo, sum128_hi);
//     uint64_t partialSum = _mm_cvtsi128_si64(sum128) +
//                             _mm_cvtsi128_si64(_mm_unpackhi_epi64(sum128, sum128));
//     int64_t total = static_cast<int64_t>(partialSum) - 4096; // 128*32 = 4096
//     return static_cast<int>(total);
// }
//
//
//
//
// static inline uint32_t extractOddBits(uint64_t word) {
//     return (uint32_t)_pext_u64(word, 0xAAAAAAAAAAAAAAAAULL);
// }
//
// // Extract even-indexed bits (positions 0, 2, …, 62) from a 64-bit word.
// static inline uint32_t extractEvenBits(uint64_t word) {
//     return (uint32_t)_pext_u64(word, 0x5555555555555555ULL);
// }
//
static inline void extractOddAndEvenBits(uint64_t word,
                                          uint32_t* odd,
                                          uint32_t* even) {
    *even = (uint32_t)_pext_u64(word, 0x5555555555555555ULL);
    *odd  = (uint32_t)_pext_u64(word, 0xAAAAAAAAAAAAAAAAULL);
}
//
// // Process one block of 8 x-bytes (encoding 32 decoded i2 values)
// // and 32 int8 y values. Returns the dot product for this block.
// int dot_block_16x32_avx2(const uint8_t *x_ptr, const int8_t *y_ptr) {
//     // Optionally prefetch upcoming y data
//     _mm_prefetch(reinterpret_cast<const char*>(y_ptr + 64), _MM_HINT_T0);
//
//     // Load 64 y values (2 x 32 bytes)
//     __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y_ptr));
//     __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y_ptr + 32));
//
//     // Load two 64-bit words that contain the packed i2 multipliers.
//     uint64_t word0 = *(reinterpret_cast<const uint64_t*>(x_ptr));
//     uint64_t word1 = *(reinterpret_cast<const uint64_t*>(x_ptr + 8));
//
//     // Extract odd (negative) and even (absolute) bits for each word.
//     uint32_t neg0, abs0, neg1, abs1;
//     extractOddAndEvenBits(word0, &neg0, &abs0);
//     extractOddAndEvenBits(word1, &neg1, &abs1);
//
//     // Instead of XOR, compute positive bits as: pos = abs AND (NOT neg)
//     uint32_t pos0 = abs0 & ~neg0;
//     uint32_t pos1 = abs1 & ~neg1;
//
//     // Expand the 32-bit masks to 256-bit vectors (each byte becomes a lane).
//     __m256i expanded_sign0 = expandMask32To256(neg0);
//     __m256i expanded_pos0  = expandMask32To256(pos0);
//     __m256i expanded_sign1 = expandMask32To256(neg1);
//     __m256i expanded_pos1  = expandMask32To256(pos1);
//
//     // Build the multiplier mask: multiplier = (positive mask) - (negative mask)
//     __m256i multiplier0 = _mm256_sub_epi8(expanded_pos0, expanded_sign0);
//     __m256i multiplier1 = _mm256_sub_epi8(expanded_pos1, expanded_sign1);
//
//     // Apply the multipliers:
//     // _mm256_sign_epi8(y, multiplier) returns y if multiplier is positive,
//     // -y if multiplier is negative, or 0 if multiplier is 0.
//     __m256i result0 = _mm256_sign_epi8(y0, multiplier0);
//     __m256i result1 = _mm256_sign_epi8(y1, multiplier1);
//
//     // Sum the results.
//     return horizontal_sum_int8_avx2(result0) + horizontal_sum_int8_avx2(result1);
// }
// Horizontal sum of 32 int8 values stored in a __m256i.
static inline int horizontal_sum_int8_avx2(__m256i v) {
    __m256i offset = _mm256_set1_epi8((char)128);
    __m256i v_offset = _mm256_add_epi8(v, offset);
    __m256i sum256 = _mm256_sad_epu8(v_offset, _mm256_setzero_si256());
    __m128i sum128_lo = _mm256_castsi256_si128(sum256);
    __m128i sum128_hi = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi64(sum128_lo, sum128_hi);
    uint64_t partialSum = _mm_cvtsi128_si64(sum128) +
                          _mm_cvtsi128_si64(_mm_unpackhi_epi64(sum128, sum128));
    int64_t total = static_cast<int64_t>(partialSum) - 4096; // subtract offset: 128 * 32
    return static_cast<int>(total);
}

// This 128-bit constant lookup table maps a 2-bit code (0..3) to the multiplier.
// According to our convention:
//   0 (00) → 0,
//   1 (01) → +1,
//   2 (10) → (should not occur; we map it to 0),
//   3 (11) → -1.
static inline __m128i getI2LUT128() {
    return _mm_setr_epi8( 0,  1,  0, -1,
                           0,  1,  0, -1,
                           0,  1,  0, -1,
                           0,  1,  0, -1 );
}

// Expand a 32-bit packed i2 word (4 bytes) into a 128-bit vector of 16 multipliers.
// Each original byte packs 4 i2 values (2 bits each). We want to “deposit” each byte
// into its own 32-bit lane by replicating it four times, then mask out all but the two low bits,
// and finally map each 2-bit code to its multiplier.
static inline __m128i expandI2_32To128(uint32_t X) {
    // Step 1: Load the 4 bytes (32 bits) into the low 32 bits of a 128-bit register.
    __m128i low = _mm_cvtsi32_si128(X);  // low now contains our 4 bytes

    // Step 2: Unpack the 4 bytes to 4 32-bit integers.
    __m128i bytes = _mm_cvtepu8_epi32(low);  // Each 32-bit element holds one original byte.

    // Step 3: Broadcast each 32-bit element’s low byte to all four bytes in that element.
    // For element0, we want: {0,0,0,0}; for element1: {4,4,4,4}; for element2: {8,8,8,8}; for element3: {12,12,12,12}.
    const __m128i control = _mm_setr_epi8(0, 0, 0, 0,
                                           4, 4, 4, 4,
                                           8, 8, 8, 8,
                                          12,12,12,12);
    __m128i expanded = _mm_shuffle_epi8(bytes, control);
    // Now each 32-bit element contains four copies of its original byte.

    // Step 4: Mask each byte to keep only the low two bits.
    expanded = _mm_and_si128(expanded, _mm_set1_epi8(3));

    // Step 5: Use the lookup table to map each 2-bit code to the multiplier.
    __m128i lut = getI2LUT128();
    __m128i multiplier = _mm_shuffle_epi8(lut, expanded);
    return multiplier;
}

// The dot-product function using the new unpack approach.
// - x_ptr points to 16 bytes of packed i2 data (encoding 64 multipliers).
// - y_ptr points to 64 int8 values.
// We decode x_ptr into 64 multipliers and then compute the dot product.
int dot_block_16x32_avx2(const uint8_t *x_ptr, const int8_t *y_ptr) {
    _mm_prefetch(reinterpret_cast<const char*>(y_ptr + 64), _MM_HINT_T0);

    // Load 64 y values in two 256-bit registers.
    __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y_ptr));
    __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y_ptr + 32));

    // Load four 32-bit words (16 bytes total) from x_ptr.
    uint32_t x0 = *(const uint32_t*)(x_ptr + 0);
    uint32_t x1 = *(const uint32_t*)(x_ptr + 4);
    uint32_t x2 = *(const uint32_t*)(x_ptr + 8);
    uint32_t x3 = *(const uint32_t*)(x_ptr + 12);

    // Expand each 32-bit word into a 128-bit vector of 16 multipliers.
    __m128i m0 = expandI2_32To128(x0);
    __m128i m1 = expandI2_32To128(x1);
    __m128i m2 = expandI2_32To128(x2);
    __m128i m3 = expandI2_32To128(x3);

    // Combine m0 and m1 into one 256-bit vector (first 32 multipliers).
    __m256i multiplier0 = _mm256_inserti128_si256(_mm256_castsi128_si256(m0), m1, 1);
    // Combine m2 and m3 into another 256-bit vector (second 32 multipliers).
    __m256i multiplier1 = _mm256_inserti128_si256(_mm256_castsi128_si256(m2), m3, 1);

    // Apply the multipliers to y:
    // _mm256_sign_epi8(y, multiplier) returns:
    //    y if multiplier is +1,
    //   -y if multiplier is -1,
    //    0 if multiplier is 0.
    __m256i result0 = _mm256_sign_epi8(y0, multiplier0);
    __m256i result1 = _mm256_sign_epi8(y1, multiplier1);

    // Sum and return the dot product.
    return horizontal_sum_int8_avx2(result0) + horizontal_sum_int8_avx2(result1);
}

// Process 8 sub-blocks (each 8 x-bytes and 32 y values) unrolled.
int dot_block_i2_i8_avx2_full(const uint8_t *x_block, const int8_t *y_block) {
    int total = 0;
    total += dot_block_16x32_avx2(x_block + 0*16, y_block + 0*64);
    total += dot_block_16x32_avx2(x_block + 1*16, y_block + 1*64);
    total += dot_block_16x32_avx2(x_block + 2*16, y_block + 2*64);
    total += dot_block_16x32_avx2(x_block + 3*16, y_block + 3*64);
    return total;
}




// int dot_block_i2_i8_avx2_full(const uint8_t *x_block, const int8_t *y_block) {
//     // Unroll four sub-blocks; declare accumulators as int.
//     int total = 0;
//
//     for (int sub = 0; sub < 4; sub++) {
//         // Pointers to current sub-block.
//         const uint8_t *x_ptr = x_block + sub * 16;
//         const int8_t  *y_ptr = y_block  + sub * 64;
//
//         // Load y values.
//         __m256i y0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y_ptr));
//         __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y_ptr + 32));
//
//         // Load and extract bits.
//         uint64_t word0 = *(const uint64_t*)x_ptr;
//         uint64_t word1 = *(const uint64_t*)(x_ptr + 8);
//
//         uint32_t neg0, abs0, neg1, abs1;
//         extractOddAndEvenBits(word0, &neg0, &abs0);
//         extractOddAndEvenBits(word1, &neg1, &abs1);
//         uint32_t pos0 = abs0 ^ neg0;
//         uint32_t pos1 = abs1 ^ neg1;
//
//         __m256i expanded_sign0 = expandMask32To256(neg0);
//         __m256i expanded_pos0  = expandMask32To256(pos0);
//         __m256i expanded_sign1 = expandMask32To256(neg1);
//         __m256i expanded_pos1  = expandMask32To256(pos1);
//
//         __m256i dot0 = _mm256_sub_epi8(_mm256_and_si256(y0, expanded_pos0),
//                                        _mm256_and_si256(y0, expanded_sign0));
//         __m256i dot1 = _mm256_sub_epi8(_mm256_and_si256(y1, expanded_pos1),
//                                        _mm256_and_si256(y1, expanded_sign1));
//
//         total += horizontal_sum_int8_avx2(dot0) + horizontal_sum_int8_avx2(dot1);
//     }
//     return total;
// }quantize_row_i8_s

// Wrapper function remains essentially the same.

// void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs,
//                            const void * vx, size_t bx,
//                            const void * vy, size_t by,
//                            int nrc)
// {
//     const uint8_t * x = reinterpret_cast<const uint8_t *>(vx);
//     const int8_t  * y = reinterpret_cast<const int8_t *>(vy);
//     int nb = n / 256; // Each block decodes 256 values.
//     long long sum = 0;
//     for (int i = 0; i < nb; i++) {
//         sum += dot_block_i2_i8_avx2_full(x + i * 64, y + i * 256);
//     }
//     *s = static_cast<float>(sum);
// }



void ggml_vec_dot_i2_i8_s(
    int n,
    float * s,
    size_t bs,
    const void * vx,
    size_t bx,
    const void * vy,
    size_t by,
    int nrc)
{
    const uint8_t * x = (const uint8_t *) vx;
    const int8_t  * y = (const int8_t  *) vy;

    // How many groups of QK_I2_S, same as before:
    const int nb = n / QK_I2_S;
    const int group32_num  = nb / 32;
    const int la_num       = nb % 32;
    const int groupla_num  = la_num ? 1 : 0;

#if defined(__AVX2__)

    // We'll accumulate final result in 32-bit registers
    __m256i accu = _mm256_setzero_si256();

    // A mask to keep only the *top* 2 bits in each byte, i.e. 0xC0 = b11000000
    const __m256i top2_mask = _mm256_set1_epi8((char)0xC0);

    for (int i = 0; i < group32_num; i++) {
        // partial sum in 32 bits
        __m256i accu32 = _mm256_setzero_si256();

        // loop over 32 "blocks" (similar to old code)
        for (int j = 0; j < 32; j++) {
            // load 32 bytes from x
            __m256i x_bytes = _mm256_loadu_si256((const __m256i *)(x + i*32*32 + j*32));

            // For 2-bit data in the top bits, you might have multiple sub-chunks
            // each sub-chunk could represent one of the 4 weights inside that byte.
            // This example just shows the conceptual approach:

            __m256i x0 = x_bytes;                         // group 0: bits [7:6]
            __m256i x1 = _mm256_slli_epi16(x_bytes, 2);       // group 1: bits [5:4] → [7:6]
            __m256i x2 = _mm256_slli_epi16(x_bytes, 4);       // group 2: bits [3:2] → [7:6]
            __m256i x3 = _mm256_slli_epi16(x_bytes, 6);
            // keep only the top bits
            x0 = _mm256_and_si256(x0, top2_mask);
            x1 = _mm256_and_si256(x1, top2_mask);
            x2 = _mm256_and_si256(x2, top2_mask);
            x3 = _mm256_and_si256(x3, top2_mask);

            // load y in 4 sub-chunks (32 bytes each) - same as the old maddubs approach
            __m256i y0 = _mm256_loadu_si256((const __m256i*)(y + i*128*32 + j*128 + 0));
            __m256i y1 = _mm256_loadu_si256((const __m256i*)(y + i*128*32 + j*128 + 32));
            __m256i y2 = _mm256_loadu_si256((const __m256i*)(y + i*128*32 + j*128 + 64));
            __m256i y3 = _mm256_loadu_si256((const __m256i*)(y + i*128*32 + j*128 + 96));


            // "Multiply" by -1, 0, or +1 via sign bits in xN.
            // PSIGNB: if x < 0 => -y, if x = 0 => 0, if x>0 => y
            __m256i r0 = _mm256_sign_epi8(y0, x0);
            __m256i r1 = _mm256_sign_epi8(y1, x1);
            __m256i r2 = _mm256_sign_epi8(y2, x2);
            __m256i r3 = _mm256_sign_epi8(y3, x3);

            // Now we have 32 int8s in each rN, but we want to sum them in 16 or 32 bits.
            // We'll sign-extend from int8 -> int16 in two steps (low half, high half).
            __m256i r0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r0));
            __m256i r0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r0, 1));

            __m256i r1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r1));
            __m256i r1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r1, 1));

            __m256i r2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r2));
            __m256i r2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r2, 1));

            __m256i r3_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r3));
            __m256i r3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r3, 1));

            // sum each pair of 16-bit vectors
            __m256i sum0 = _mm256_add_epi16(r0_lo, r0_hi);
            __m256i sum1 = _mm256_add_epi16(r1_lo, r1_hi);
            __m256i sum2 = _mm256_add_epi16(r2_lo, r2_hi);
            __m256i sum3 = _mm256_add_epi16(r3_lo, r3_hi);

            __m256i sumA = _mm256_add_epi16(sum0, sum1);
            __m256i sumB = _mm256_add_epi16(sum2, sum3);
            __m256i sumAll = _mm256_add_epi16(sumA, sumB);

            // Finally, we can "multiply" by 1 in 16 bits to get 32 bits
            // or use `_mm256_madd_epi16(sumAll, ones16)`
            __m256i ones16 = _mm256_set1_epi16(1);
            // each 16-bit lane of sumAll -> add into 32-bit lane:
            __m256i sum32 = _mm256_madd_epi16(sumAll, ones16);

            // accumulate into accu32
            accu32 = _mm256_add_epi32(accu32, sum32);
        }
        // add partial sum to global accumulator
        accu = _mm256_add_epi32(accu, accu32);
    }

        // handle leftover 'la_num' if needed
    if (groupla_num) {
        __m256i accula = _mm256_setzero_si256();
        // Pointer offset for leftover blocks:
        const uint8_t * x_left = x + group32_num * 32 * 32;
        const int8_t  * y_left = y + group32_num * 128 * 32;
        for (int j = 0; j < la_num; j++) {
            __m256i x_bytes = _mm256_loadu_si256((const __m256i *)(x_left + j * 32));
            __m256i x0 = _mm256_and_si256(x_bytes, top2_mask);
            __m256i x1 = _mm256_and_si256(_mm256_slli_epi16(x_bytes, 2), top2_mask);
            __m256i x2 = _mm256_and_si256(_mm256_slli_epi16(x_bytes, 4), top2_mask);
            __m256i x3 = _mm256_and_si256(_mm256_slli_epi16(x_bytes, 6), top2_mask);

            __m256i y0 = _mm256_loadu_si256((const __m256i *)(y_left + j * 128 +  0));
            __m256i y1 = _mm256_loadu_si256((const __m256i *)(y_left + j * 128 + 32));
            __m256i y2 = _mm256_loadu_si256((const __m256i *)(y_left + j * 128 + 64));
            __m256i y3 = _mm256_loadu_si256((const __m256i *)(y_left + j * 128 + 96));

            __m256i r0 = _mm256_sign_epi8(y0, x0);
            __m256i r1 = _mm256_sign_epi8(y1, x1);
            __m256i r2 = _mm256_sign_epi8(y2, x2);
            __m256i r3 = _mm256_sign_epi8(y3, x3);

            __m256i r0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r0));
            __m256i r0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r0, 1));
            __m256i r1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r1));
            __m256i r1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r1, 1));
            __m256i r2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r2));
            __m256i r2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r2, 1));
            __m256i r3_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(r3));
            __m256i r3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(r3, 1));

            __m256i sum0 = _mm256_add_epi16(r0_lo, r0_hi);
            __m256i sum1 = _mm256_add_epi16(r1_lo, r1_hi);
            __m256i sum2 = _mm256_add_epi16(r2_lo, r2_hi);
            __m256i sum3 = _mm256_add_epi16(r3_lo, r3_hi);
            __m256i sumA = _mm256_add_epi16(sum0, sum1);
            __m256i sumB = _mm256_add_epi16(sum2, sum3);
            __m256i sumAll = _mm256_add_epi16(sumA, sumB);

            __m256i ones16 = _mm256_set1_epi16(1);
            __m256i sum32 = _mm256_madd_epi16(sumAll, ones16);

            accula = _mm256_add_epi32(accula, sum32);
        }
        accu = _mm256_add_epi32(accu, accula);
    }

    // final horizontal sum
    int sumi = hsum_i32_8(accu);
    *s = (float) sumi;
#endif

}


// ggml_vec_dot_i2_i8_s


// void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs,
//                            const void * vx, size_t bx,
//                            const void * vy, size_t by,
//                            int nrc)
// {
//     (void) bs;  (void) bx;  (void) by;  (void) nrc;
//
//     // 1) Cast pointers
//     const uint8_t * x = (const uint8_t *) vx;   // 2-bit packed array
//     const int8_t  * y = (const int8_t  *) vy;   // int8 array
//
//     // 2) We do NOT multiply by the scale factor (return non-scaled)
//     //    -> The user explicitly said "no scale in the dot product"
//     //       so we skip retrieving it from x[n/4].
//
//     // 3) Accumulate in 64-bit to avoid overflow
//     int64_t sum = 0;
//
//     // 4) For each element i in [0..n-1], decode 2 bits from x
//     //    and map to { -1, 0, +1 }.
//     for (int i = 0; i < n; i++) {
//         int code = (x[i/4] >> (2 * (i % 4))) & 0x03;
//
//         // According to our quantize function:
//         //   3 => -1
//         //   1 => +1
//         //   0 =>  0
//         // We do NOT expect "2" from the new mapping, but let's map it to 0 if it appears.
//         int xi;
//         if      (code == 3) xi = -1;
//         else if (code == 1) xi = +1;
//         else                xi =  0;
//
//         // Multiply by y[i] and accumulate
//         sum += (int64_t) xi * (int64_t) y[i];
//     }
//
//     // 5) Store final sum as float in *s (no scale factor applied)
//     *s = (float) sum;
// }


void ggml_vec_dot_i2_i8_s2(int n, float * s, size_t bs,
                          const void * vx, size_t bx,
                          const void * vy, size_t by,
                          int nrc) {
    // Cast input pointers to the appropriate types
    const uint8_t * x = (const uint8_t *) vx;
    const int8_t  * y = (const int8_t  *) vy;

    // We'll accumulate in 64-bit to avoid overflow for large n
    int64_t sum = 0;

    // Each byte of x contains four 2-bit values:
    //   x[i/4] >> (2*(i%4)) & 0x3
    // extracts the i-th 2-bit value (0..3).
    for (int i = 0; i < n; i++) {
        // Extract the i-th 2-bit value
        int xi = (x[i / 4] >> (2 * (i % 4))) & 0x03;

        // Multiply by y[i] (signed int8)
        sum += xi * y[i];
    }

    // Store the result as a float
    *s = (float) sum;
}