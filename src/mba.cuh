#ifndef SIMBA_MBA
#define SIMBA_MBA
#include "globals.cuh"

// Test whether the given LTLf formula is a solution.
__device__ bool is_solution_rpn(char *formula, int formula_size,
                                int num_examples, int num_variables) {
  uint32_t stack[maxFormulaSize];
  int stackIdx;

  for (int i = 0; i < num_examples; ++i) {

    stackIdx = -1;

    for (int j = 0; j < formula_size; ++j) {

      char token = formula[j];

      if (token >= '0' && token <= '9') {
        // Variables
        int var = token - '0';
        stack[++stackIdx] = d_inputs[i * num_variables + var];
      } else {
        uint32_t left, right;
        if (token == '~' || token == '#') {
          // Unary operators
          left = stack[stackIdx];
          switch (token) {
          case '~':
            stack[stackIdx] = ~left;
            break;
          case '#':
            stack[stackIdx] = -left;
            break;
          }
        } else {
          // Binary operators
          right = stack[stackIdx--];
          left = stack[stackIdx];
          switch (token) {
          case '&':
            stack[stackIdx] = left & right;
            break;
          case '|':
            stack[stackIdx] = left | right;
            break;
          case '^':
            stack[stackIdx] = left ^ right;
            break;
          case '+':
            stack[stackIdx] = left + right;
            break;
          case '-':
            stack[stackIdx] = left - right;
            break;
          case '*':
            stack[stackIdx] = left * right;
            break;
          }
        }
      }
    }

    if (stack[0] != d_outputs[i])
      return false;
  }

  return true;
}

#endif