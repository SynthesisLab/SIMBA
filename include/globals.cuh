#ifndef SIMBA_GLOBALS
#define SIMBA_GLOBALS
#include <cstdint>

const int max_n_variables = 10;
const int max_n_examples = 100;
const int max_formula_size = 32;

__constant__ uint32_t d_inputs[max_n_variables * max_n_examples];
__constant__ uint32_t d_outputs[max_n_examples];
// __device__ __constant__ char d_opChar[8] = {'~', '&', '|', '^',
//                                             '#', // Unary minus
//                                             '+', '-', '*'};
__constant__ uint64_t d_numForm[max_formula_size + 1][9];

__device__ bool is_solution(uint32_t *cs, const int n_examples) {
  bool ok = true;
  for (int i = 0; i < n_examples; ++i) {
    ok = ok && (cs[i] == d_outputs[i]);
  }

  return ok;
}

#endif