#ifndef SIMBA_GLOBALS
#define SIMBA_GLOBALS
#include <cstdint>

#define LARGE_UINT uint64_t

const uint32_t maxNumOfVars = 10;
const uint32_t maxNumOfSamples = 100;
const uint32_t maxFormulaSize = 32;
const uint32_t maxOperators = 8;
const char *opStr[maxOperators] = {"(~)", "(&)", "(|)", "(^)",
                                   "(Neg)", "(+)", "(-)", "(*)"};

__constant__ uint32_t d_inputs[maxNumOfVars * maxNumOfSamples];
__constant__ uint32_t d_outputs[maxNumOfSamples];
__device__ __constant__ char d_opChar[maxOperators] = {'~', '&', '|', '^',
                                                       '#', // Unary minus
                                                       '+', '-', '*'};
__device__ __constant__ LARGE_UINT d_numForm[maxFormulaSize + 1][9];
__device__ __constant__ LARGE_UINT d_numFormUncum[maxFormulaSize + 1];

__constant__ LARGE_UINT d_numFormP[maxFormulaSize - 1][maxFormulaSize - 1];

static_assert(maxFormulaSize <= 32);
__device__ __forceinline__ void setFormulaChar(
    uint64_t &f0, uint64_t &f1,
    uint64_t &f2, uint64_t &f3,
    int pos, char value)
{
    uint64_t mask = 0xffULL << ((pos & 7) * 8);
    uint64_t v = uint64_t(uint8_t(value)) << ((pos & 7) * 8);

    switch (pos >> 3)
    {
    case 0:
        f0 = (f0 & ~mask) | v;
        break;
    case 1:
        f1 = (f1 & ~mask) | v;
        break;
    case 2:
        f2 = (f2 & ~mask) | v;
        break;
    default:
        f3 = (f3 & ~mask) | v;
        break;
    }
}

__device__ __forceinline__ char getFormulaChar(
    uint64_t f0, uint64_t f1,
    uint64_t f2, uint64_t f3,
    int pos)
{
    uint64_t x;

    switch (pos >> 3)
    {
    case 0:
        x = f0;
        break;
    case 1:
        x = f1;
        break;
    case 2:
        x = f2;
        break;
    default:
        x = f3;
        break;
    }

    return char((x >> ((pos & 7) * 8)) & 0xff);
}
#endif
