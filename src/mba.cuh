#ifndef SIMBA_MBA
#define SIMBA_MBA
#include "globals.cuh"

// Test whether the given LTLf formula is a solution.
static constexpr uint32_t VAR_MASK = 1u << 0;
static constexpr uint32_t NOT_MASK = 1u << 1;
static constexpr uint32_t NEG_MASK = 1u << 2;
static constexpr uint32_t AND_MASK = 1u << 3;
static constexpr uint32_t OR_MASK = 1u << 4;
static constexpr uint32_t XOR_MASK = 1u << 5;
static constexpr uint32_t ADD_MASK = 1u << 6;
static constexpr uint32_t SUB_MASK = 1u << 7;
static constexpr uint32_t MUL_MASK = 1u << 8;
static constexpr uint32_t UNARY_MASK = NOT_MASK | NEG_MASK;
static constexpr uint32_t BINARY_MASK =
    AND_MASK | OR_MASK | XOR_MASK | ADD_MASK | SUB_MASK | MUL_MASK;

__device__ __forceinline__ int first_valid_operator(uint32_t valids)
{
  return __ffs(valids) - 1;
}
__device__ __forceinline__ void eval(const uint64_t f0, const uint64_t f1, const uint64_t f2, const uint64_t f3, int formula_size,
                                     int sample_index, int num_variables, uint32_t stack[maxFormulaSize])
{
  int stack_index = -1;
  for (int j = 0; j < formula_size - 1; ++j)
    {

      char token = getFormulaChar(f0, f1, f2, f3, j);

      if (token >= '0' && token <= '9')
      {
        // Variables
        uint8_t var = token - '0';
        stack[++stack_index] = d_inputs[sample_index * num_variables + var];
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
}
__device__ int is_solution_rpn_unary(const uint64_t f0, const uint64_t f1, const uint64_t f2, const uint64_t f3, int formula_size,
                                     int num_examples, int num_variables)
{
  uint32_t stack[maxFormulaSize];
  uint32_t valids = UNARY_MASK;

  for (int i = 0; i < num_examples; ++i)
  {
    eval(f0, f1, f2, f3, formula_size, i, num_variables, stack);

    // Ok so here we have everything but the last formula char
    uint32_t target = d_outputs[i];
    uint32_t result = stack[0];
    uint32_t invalids = 0;
      if ((valids & NEG_MASK) && -result != target)
        invalids |= NEG_MASK;
      if ((valids & NOT_MASK) && ~result != target)
        invalids |= NOT_MASK;
      valids &= ~invalids;
      if (valids == 0)
        return -1;

  }

  return first_valid_operator(valids);
}

__device__ int is_solution_rpn_binary(const uint64_t f0, const uint64_t f1, const uint64_t f2, const uint64_t f3, int formula_size,
                                      int num_examples, int num_variables)
{
  uint32_t stack[maxFormulaSize];
  uint32_t valids = BINARY_MASK;

  for (int i = 0; i < num_examples; ++i)
  {

    eval(f0, f1, f2, f3, formula_size, i, num_variables, stack);

    // Ok so here we have everything but the last formula char
    uint32_t target = d_outputs[i];
    uint32_t left = stack[0];
    uint32_t right = stack[1];
    uint32_t invalids = 0;
    if ((valids & MUL_MASK) && left * right != target)
      invalids |= MUL_MASK;
    if ((valids & SUB_MASK) && left - right != target)
      invalids |= SUB_MASK;
    if ((valids & ADD_MASK) && left + right != target)
      invalids |= ADD_MASK;
    if ((valids & XOR_MASK) && (left ^ right) != target)
      invalids |= XOR_MASK;
    if ((valids & OR_MASK) && (left | right) != target)
      invalids |= OR_MASK;
    if ((valids & AND_MASK) && (left & right) != target)
      invalids |= AND_MASK;
    valids &= ~invalids;
    if (valids == 0)
      return -1;

  }

  return first_valid_operator(valids);
}

__device__ int is_solution_rpn(const uint64_t f0, const uint64_t f1, const uint64_t f2, const uint64_t f3, int formula_size,
                               int num_examples, int num_variables)
{
  char last_token = getFormulaChar(f0, f1, f2, f3, formula_size - 1);
  if ( (last_token == '~') || (last_token == '#'))
    return is_solution_rpn_unary(f0, f1, f2, f3, formula_size, num_examples, num_variables);
  else if ( last_token >= '0' && last_token <= '9'){
    int var = last_token - '0';
    for (int i = 0; i < num_examples; ++i)
    {
      if(d_inputs[i * num_variables + var] != d_outputs[i])
        return -1;
    }
    return 0;
  }
  else
    return is_solution_rpn_binary(f0, f1, f2, f3, formula_size, num_examples, num_variables);
}

#endif
