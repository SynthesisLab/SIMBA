#ifndef SIMBA_GLOBALS
#define SIMBA_GLOBALS
#include <cstdint>

const int maxNumOfVars = 10;
const int maxNumOfSamples = 100;
const int maxFormulaSize = 32;
const int maxOperators = 8;
const char *opStr[maxOperators] = {"(~)", "(&)", "(|)", "(^)",
                                   "(Neg)", "(+)", "(-)", "(*)"};

__constant__ uint32_t d_inputs[maxNumOfVars * maxNumOfSamples];
__constant__ uint32_t d_outputs[maxNumOfSamples];
__device__ __constant__ char d_opChar[maxOperators] = {'~', '&', '|', '^',
                                                       '#', // Unary minus
                                                       '+', '-', '*'};
__constant__ uint64_t d_numForm[maxFormulaSize + 1][9];
__constant__ uint64_t d_numFormT[maxFormulaSize + 1];

#endif