#include <iostream>

#include <QApplication>
#include <QImage>
#include <QRgb>

#include "popcnt.h"
#include <x86intrin.h>

#include <bitset>

using namespace std;

const int GRAY_THRESHOLD = 20;

int __idx(int r, int c) { return (r * 32 + c) / 128; }

int __offset(int r, int c) { return (r * 32 + c) % 128; }

inline __m128i mm_bitshift_left(__m128i x, unsigned count)
{
    __m128i carry = _mm_bslli_si128(x, 8); // old compilers only have the
                                           // confusingly named _mm_slli_si128
                                           // synonym
    if (count >= 64)
        // the non-carry part is all zero, so return early
        return _mm_slli_epi64(carry, count - 64);
    // else
    // After bslli shifted left by 64b
    carry = _mm_srli_epi64(carry, 64 - count);

    x = _mm_slli_epi64(x, count);
    return _mm_or_si128(x, carry);
}

// Prints the binary representation of a m128 in 32 bit groups
void __printbin32(__m128i var)
{
    uint32_t *val = (uint32_t *)&var;
    std::cout << std::bitset<32>(val[0]) << std::endl;
    std::cout << std::bitset<32>(val[1]) << std::endl;
    std::cout << std::bitset<32>(val[2]) << std::endl;
    std::cout << std::bitset<32>(val[3]) << std::endl;
}

void __print2bin32(__m128i var1, __m128i var2)
{
    uint32_t *val1 = (uint32_t *)&var1;
    uint32_t *val2 = (uint32_t *)&var2;
    std::cout << std::bitset<32>(val1[0]) << " " << std::bitset<32>(val2[0])
              << std::endl;
    std::cout << std::bitset<32>(val1[1]) << " " << std::bitset<32>(val2[1])
              << std::endl;
    std::cout << std::bitset<32>(val1[2]) << " " << std::bitset<32>(val2[2])
              << std::endl;
    std::cout << std::bitset<32>(val1[3]) << " " << std::bitset<32>(val2[3])
              << std::endl;
}

void printmem(__m128i (*vec)[8], int N)
{
    for (int i = 0; i < N; i++) {
        std::cout << "Element " << i << ": " << std::endl;
        for (int j = 0; j < 8; j++)
            __printbin32(vec[i][j]);
    }
}

void printmem2(__m128i *vec1, __m128i *vec2)
{
    for (int j = 0; j < 8; j++)
        __print2bin32(vec1[j], vec2[j]);
}

int argmin(int *v, int N, int idx = 0)
{
    if (N == 1) {
        return idx;
    } else if (N == 2) {
        return (v[0] < v[1]) ? idx : idx + 1;
    } else {
        int a = argmin(v, (N / 2), idx);
        int b = argmin(&(v[N / 2]), N - (N / 2), idx + (N / 2));
        return (v[a - idx] < v[b - idx]) ? a : b;
    }
}

int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    __m128i memory[5][8];

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 8; j++) {
            memory[i][j] ^= memory[i][j];
        }
    }

    for (int i = 0; i < 5; i++) {
        string filename = (":/training" + std::to_string(i) + ".png").c_str();
        QImage img(filename.c_str());
        img = img.scaled(32, 32);

        __m128i bit;

        // Thresholding the image and memorizing it
        for (int r = 0; r < img.height(); r++) {
            uchar *scan = img.scanLine(r);
            int depth = 4;
            if (r % 4 == 0) {
                bit = _mm_setr_epi32(1, 0, 0, 0);
            }
            for (int c = 0; c < img.width(); c++) {
                QRgb *rgbpixel = reinterpret_cast<QRgb *>(scan + c * depth);
                int gray = qGray(*rgbpixel);
                if (gray > GRAY_THRESHOLD) {
                    int idx = __idx(r, c);
                    memory[i][idx] = _mm_xor_si128(memory[i][idx], bit);
                }
                bit = mm_bitshift_left(bit, 1);
            }
        }
    }

    std::cout << "Training set memorized. Dumping memory..." << std::endl;
    printmem(memory, 5);

    for (int i = 0; i < 20; i++) {
        string filename = (":/patterns" + std::to_string(i) + ".png").c_str();
        QImage img(filename.c_str());
        img = img.scaled(32, 32);

        __m128i pattern[8];
        __m128i bit;

        // Thresholding the image and storing it to pattern
        for (int r = 0; r < img.height(); r++) {
            uchar *scan = img.scanLine(r);
            int depth = 4;
            if (r % 4 == 0) {
                pattern[__idx(r, 0)] = _mm_setr_epi32(0, 0, 0, 0);
                bit = _mm_setr_epi32(1, 0, 0, 0);
            }
            for (int c = 0; c < img.width(); c++) {
                QRgb *rgbpixel = reinterpret_cast<QRgb *>(scan + c * depth);
                int gray = qGray(*rgbpixel);
                if (gray > GRAY_THRESHOLD) {
                    int idx = __idx(r, c);
                    pattern[idx] = _mm_xor_si128(pattern[idx], bit);
                }
                bit = mm_bitshift_left(bit, 1);
            }
        }

        int energies[5];

        for (int j = 0; j < 5; j++) {
            energies[j] = 0;
            for (int k = 0; k < 8; k++) {
                __m128i xord = _mm_xor_si128(pattern[k], memory[j][k]);
                energies[j] += popcnt128(xord);
            }
        }

        int match = argmin(energies, 5);

        std::cout << "Pattern " << i << " energies (";
        for (int j = 0; j < 5; j++) {
            std::cout << energies[j];
            if (j != 4)
                std::cout << ", ";
        }
        std::cout << "), min: " << match << std::endl << std::endl;
        std::cout << "            Pattern             |             Match"
                  << std::endl;
        printmem2(pattern, memory[match]);
    }

    return 0;
}
