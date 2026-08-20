#ifndef SIMBA_MBA
#define SIMBA_MBA
#include "globals.cuh"

// Test whether the given LTLf formula is a solution.
#define UNARY_MASK (3 << 1)
#define VAR_MASK (1)
#define BINARY_MASK ((3 << 3) + (3 << 5) + (3 << 6) )

__device__ int is_solution_rpn(char *formula, int formula_size,
                                int num_examples, int num_variables)
{
  uint32_t stack[maxFormulaSize];
  int stack_index;
  uint32_t valids = 0;
  char last_token = formula[formula_size - 1];
  bool unary = (last_token == '~') || (last_token == '#');
  bool var_only = last_token >= '0' && last_token <= '9';
  bool binary = !unary && !var_only;
  if (unary)
  {
    valids = UNARY_MASK;
  }
  else if (binary)
  {
    valids = BINARY_MASK;
  }
  else
  {
    valids = VAR_MASK;
  }

  for (int i = 0; i < num_examples; ++i)
  {

    stack_index = -1;

    for (int j = 0; j < formula_size - 1 + var_only; ++j)
    {

      char token = formula[j];

      if (token >= '0' && token <= '9')
      {
        // Variables
        int var = token - '0';
        stack[++stack_index] = d_inputs[i * num_variables + var];
      }
      else
      {
        uint32_t left, right;
        if (token == '~' || token == '#')
        {
          // Unary operators
          left = stack[stack_index];
          switch (token)
          {
          case '~':
            stack[stack_index] = ~left;
            break;
          case '#':
            stack[stack_index] = -left;
            break;
          }
        }
        else
        {
          // Binary operators
          right = stack[stack_index--];
          left = stack[stack_index];
          switch (token)
          {
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
          }
        }
      }
    }

    // Ok so here we have everything but the last formula char
    uint32_t target = d_outputs[i];
    if (unary)
    {
      uint32_t result = stack[0];
      uint32_t mask = (-result != target) << 1;
      mask |= (~result != target);
      mask <<= 1;
      valids ^= mask & valids;
      if (valids == 0)
        return -1;
    }
    else if (binary)
    {
      uint32_t left = stack[0];
      uint32_t right = stack[1];

      uint32_t mask = ((left * right) != target) << 8;
      mask |= ((left - right) != target) << 7;
      mask |= ((left + right) != target) << 6;
      mask |= ((left ^ right) != target) << 5;
      mask |= ((left | right) != target) << 4;
      mask |= ((left & right) != target) << 3;
      valids ^= mask & valids;
      if (valids == 0)
        return -1;
    }
    else
    {
      uint32_t result = stack[0];
      if (result != target)
        return -1;
    }
    
  }

  return valids;
}

__device__ uint32_t evaluateRPN(char *formula, int sample_index,
                                int num_variables)
{

  uint32_t stack[maxFormulaSize];
  int stack_index = -1; // Index of the top of the stack.

  int idx = 0;
  char token;
  while ((token = formula[idx++]) != '\0')
  {
    if (token >= '0' && token <= '9')
    {
      int var = token - '0';
      stack[++stack_index] = d_inputs[sample_index * num_variables + var];
    }
    else
    {
      uint32_t left, right;
      if (token == '~' || token == '#')
      {
        left = stack[stack_index];
        if (token == '~')
          stack[stack_index] = ~left;
        else
          stack[stack_index] = -left;
      }
      else
      {
        right = stack[stack_index--];
        left = stack[stack_index];
        switch (token)
        {
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
        }
      }
    }
  }

  return stack[0];
}

#endif