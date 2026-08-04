#ifndef SIMBA_MBA
#define SIMBA_MBA
#include "globals.cuh"

// Test whether the given LTLf formula is a solution.
__device__ bool is_solution_rpn(char *formula, int formula_size,
                                int num_examples, int num_variables) {
  uint32_t stack[max_formula_size];
  int stack_index;

  for (int i = 0; i < num_examples; ++i) {

    stack_index = -1;

    for (int j = 0; j < formula_size; ++j) {

      char token = formula[j];

      if (token >= '0' && token <= '9') {
        // Variables
        int var = token - '0';
        stack[++stack_index] = d_inputs[i * num_variables + var];
      } else if (token >= 'A' && token <= 'A' + max_n_constants - 1) {
        // Constants
        stack[++stack_index] = d_constants[token - 'A'];
      } else {
        uint32_t left, right;
        if (token == '~' || token == '#') {
          // Unary operators
          left = stack[stack_index];
          switch (token) {
          case '~':
            stack[stack_index] = ~left;
            break;
          case '#':
            stack[stack_index] = -left;
            break;
          }
        } else {
          // Binary operators
          right = stack[stack_index--];
          left = stack[stack_index];
          switch (token) {
          case '&':
            stack[stack_index] = left & right;
            break;
          case '|':
            stack[stack_index] = left | right;
            break;
          case '^':
            stack[stack_index] = left ^ right;
            break;
          case '+':
            stack[stack_index] = left + right;
            break;
          case '-':
            stack[stack_index] = left - right;
            break;
          case '*':
            stack[stack_index] = left * right;
            break;
          case '<':
            stack[stack_index] = left << right;
            break;
          case '>':
            stack[stack_index] = left >> right;
            break;
          case '/':
            stack[stack_index] = left / right;
            break;
          case '%':
            stack[stack_index] = left % right;
            break;
          case 'a':
            stack[stack_index] =
                static_cast<uint32_t>(static_cast<int32_t>(left) >> right);
            break;
          case 'q':
            stack[stack_index] = static_cast<uint32_t>(
                static_cast<int32_t>(left) / static_cast<int32_t>(right));
            break;
          case 'r':
            stack[stack_index] = static_cast<uint32_t>(
                static_cast<int32_t>(left) % static_cast<int32_t>(right));
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

__device__ uint32_t evaluateRPN(char *formula, int sample_index,
                                int num_variables) {

  uint32_t stack[max_formula_size];
  int stack_index = -1; // Index of the top of the stack.

  int idx = 0;
  char token;
  while ((token = formula[idx++]) != '\0') {
    if (token >= '0' && token <= '9') {
      int var = token - '0';
      stack[++stack_index] = d_inputs[sample_index * num_variables + var];
    } else if (token >= 'A' && token <= 'A' + max_n_constants - 1) {
      stack[++stack_index] = d_constants[token - 'A'];
    } else {
      uint32_t left, right;
      if (token == '~' || token == '#') {
        left = stack[stack_index];
        if (token == '~')
          stack[stack_index] = ~left;
        else
          stack[stack_index] = -left;
      } else {
        right = stack[stack_index--];
        left = stack[stack_index];
        switch (token) {
        case '&':
          stack[stack_index] = left & right;
          break;
        case '|':
          stack[stack_index] = left | right;
          break;
        case '^':
          stack[stack_index] = left ^ right;
          break;
        case '<':
          stack[stack_index] = left << right;
          break;
        case '>':
          stack[stack_index] = left >> right;
          break;
        case '+':
          stack[stack_index] = left + right;
          break;
        case '-':
          stack[stack_index] = left - right;
          break;
        case '*':
          stack[stack_index] = left * right;
          break;
        case '/':
          stack[stack_index] = left / right;
          break;
        case '%':
          stack[stack_index] = left % right;
          break;
        case 'a':
          stack[stack_index] =
              static_cast<uint32_t>(static_cast<int32_t>(left) >> right);
          break;
        case 'q':
          stack[stack_index] = static_cast<uint32_t>(
              static_cast<int32_t>(left) / static_cast<int32_t>(right));
          break;
        case 'r':
          stack[stack_index] = static_cast<uint32_t>(
              static_cast<int32_t>(left) % static_cast<int32_t>(right));
          break;
        }
      }
    }
  }

  return stack[0];
}

#endif