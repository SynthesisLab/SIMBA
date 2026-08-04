#ifndef SIMBA_GLOBALS
#define SIMBA_GLOBALS
#include "simba_limits.h"
#include <cstdint>

__constant__ uint32_t d_inputs[max_n_variables * max_n_examples];
__constant__ uint32_t d_outputs[max_n_examples];
__constant__ uint32_t d_constants[max_n_constants];
// __device__ __constant__ char d_opChar[8] = {'~', '&', '|', '^',
//                                             '#', // Unary minus
//                                             '+', '-', '*'};
// Descriptor for one active op, used by nocache.cu's runtime-generic
// bijective numbering (numberToFormula / compute_n_formulas).
struct ActiveOp {
  char unique_char;
  bool is_binary;
};
__constant__ ActiveOp d_activeOps[max_n_ops];
__constant__ uint64_t d_numForm[max_formula_size + 1][max_n_ops + 1];

inline __device__ bool is_solution(uint32_t *cs, const int n_examples) {
  bool ok = true;
  for (int i = 0; i < n_examples; ++i) {
    ok = ok && (cs[i] == d_outputs[i]);
  }

  return ok;
}

#endif