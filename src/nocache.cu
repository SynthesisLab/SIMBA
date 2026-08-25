#include "globals.cuh"
#include "instance.h"
#include "mba.cuh"
#include "utils.cuh"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <format>
#include <stack>
#include <vector>
#include <cuda/cmath>
#define MEASUREMENT_MODE

__device__ int solution_found = 0;

__device__ void numberToFormula(LARGE_UINT n, int size,
                                uint64_t &f0, uint64_t &f1, uint64_t &f2, uint64_t &f3)
{

  LARGE_UINT n_stack[maxFormulaSize / 2];
  uint8_t size_stack[maxFormulaSize / 2];
  uint8_t pos = size - 1;
  int stackIdx = -1;
  
  while (true)
  {
    if(size == 1){
      setFormulaChar(f0, f1, f2, f3, pos--, '0' + n);
      if(stackIdx < 0)
        break;
      n = n_stack[stackIdx];
      size = size_stack[stackIdx--];
    } else {
      uint8_t opIdx = 0;
      opIdx += n >= d_numForm[size][0];
      opIdx += n >= d_numForm[size][1];
      opIdx += n >= d_numForm[size][2];
      opIdx += n >= d_numForm[size][3];
      opIdx += n >= d_numForm[size][4];
      opIdx += n >= d_numForm[size][5];
      opIdx += n >= d_numForm[size][6];
      if(opIdx)
        n -= d_numForm[size][opIdx - 1];

      size--;
      setFormulaChar(f0, f1, f2, f3, pos--, d_opChar[opIdx]);

      if (!(opIdx == 0 || opIdx == 4))
      {
        uint8_t fIdx = 1;

        while (n >= d_numFormP[size - 1][fIdx - 1])
        {
          n -= d_numFormP[size - 1][fIdx - 1];
          fIdx++;
        }

        LARGE_UINT rightsum = d_numFormUncum[size - fIdx];
        n_stack[++stackIdx] = n / rightsum;
        size_stack[stackIdx] = fIdx;
        n = n % rightsum;
        size = size - fIdx;
      }
    }
  }
}


__device__ __forceinline__ char operator_char_from_index(int opIdx)
{
  switch (opIdx)
  {
  case 1:
    return '~';
  case 2:
    return '#';
  case 3:
    return '&';
  case 4:
    return '|';
  case 5:
    return '^';
  case 6:
    return '+';
  case 7:
    return '-';
  case 8:
    return '*';
  default:
    return '\0';
  }
}

__global__ void processOperator(const int n_examples, const int n_variables,
                                int formula_size, LARGE_UINT offset,
                                LARGE_UINT n_threads, char *d_res)
{

  LARGE_UINT tid =
      static_cast<LARGE_UINT>(blockDim.x) * static_cast<LARGE_UINT>(blockIdx.x) +
      static_cast<LARGE_UINT>(threadIdx.x);

  if (tid < n_threads)
  {

    tid += offset;

    uint64_t f0 = 0, f1 = 0, f2 = 0, f3 = 0;
    numberToFormula(tid, formula_size, f0, f1, f2, f3);
    int opIdx =
        is_solution_rpn(f0, f1, f2, f3, formula_size, n_examples, n_variables);
    bool found = opIdx >= 0;
    if (found)
      if (atomicCAS(&solution_found, 0, 1) == 0){
        for (int i = 0; i < formula_size - 1; ++i)
          d_res[i] = getFormulaChar(f0, f1, f2, f3, i);
        if(opIdx == 0){
          d_res[formula_size - 1] = getFormulaChar(f0, f1, f2, f3, formula_size - 1);
        } else {
          d_res[formula_size - 1] = operator_char_from_index(opIdx);
        }
      }
       
  }
}

void printMatrix(LARGE_UINT *matrix, int m, int n)
{
  for (int i = 0; i < m; ++i)
  {
    for (int j = 0; j < n; ++j)
      fprintf(stderr, "%-12lu ", matrix[i * n + j]);
    fprintf(stderr, "\n");
  }
}

using u128 = unsigned __int128;

constexpr LARGE_UINT maxSignedInt128()
{
  return (static_cast<u128>(1) << 127) - 1;
}

bool checkedAdd128(LARGE_UINT a, LARGE_UINT b, LARGE_UINT &result)
{
  if (a > maxSignedInt128() - b)
    return false;
  result = a + b;
  return true;
}

bool checkedMul128(LARGE_UINT a, LARGE_UINT b, LARGE_UINT &result)
{
  if (a != 0 && b > maxSignedInt128() / a)
    return false;
  result = a * b;
  return true;
}

bool checkedAddProduct128(LARGE_UINT &sum, LARGE_UINT a, LARGE_UINT b)
{
  LARGE_UINT product;
  if (!checkedMul128(a, b, product))
    return false;
  return checkedAdd128(sum, product, sum);
}

bool fitsUint64(LARGE_UINT value)
{
  return value <= static_cast<u128>(UINT64_MAX);
}

std::tuple<std::vector<LARGE_UINT>, std::vector<LARGE_UINT>, std::vector<LARGE_UINT>, std::vector<LARGE_UINT>> compute_n_formulas(int max_size, int n_variables)
{

  // Last column is the number of formulas of the given size
  std::vector<LARGE_UINT> numForm((max_size + 1) * 9, 0);
  std::vector<LARGE_UINT> unaryForm((max_size + 1) * 9, 0);
  std::vector<LARGE_UINT> binaryForm((max_size + 1) * 9, 0);
  std::vector<LARGE_UINT> numFormCumulative((max_size + 1) * 9, 0);

  // Initialisation with variables and unary operators
  numForm[1 * 9 + 8] = n_variables;
  numForm[2 * 9 + 0] = n_variables;
  numForm[2 * 9 + 4] = n_variables;
  numForm[2 * 9 + 8] = 2 * n_variables;
  unaryForm[2 * 9 + 8] = 2 * n_variables;

  // Generation
  uint64_t numOfUnaryFormulas;
  uint64_t numOfCommutBinaryFormulas;
  uint64_t numOfNotCommutBinaryFormulas;
  bool overflow128 = false;
  bool overflow64 = false;

  for (int i = 3; i <= max_size; ++i)
  {
    numOfUnaryFormulas = numForm[(i - 1) * 9 + 8];
    numOfCommutBinaryFormulas = 0;
    numOfNotCommutBinaryFormulas = 0;
    for (int j = 1; j <= (i - 1) / 2; ++j)
    {
      if (!checkedAddProduct128(numOfCommutBinaryFormulas,
                                numForm[j * 9 + 8],
                                numForm[(i - j - 1) * 9 + 8]))
      {
        fprintf(stderr, "Error: d_numForm precomputation overflowed 128 bits at formula size %d.\n", i);
        overflow128 = true;
      }
    }
    for (int j = 1; j <= i - 2; ++j)
    {
      if (!checkedAddProduct128(numOfNotCommutBinaryFormulas,
                                numForm[j * 9 + 8],
                                numForm[(i - j - 1) * 9 + 8]))
      {
        fprintf(stderr, "Error: d_numForm precomputation overflowed 128 bits at formula size %d.\n", i);
        overflow128 = true;
      }
    }
    LARGE_UINT total = 0;
    LARGE_UINT term = 0;
    if (!checkedMul128(2, numOfUnaryFormulas, total) ||
        !checkedMul128(5, numOfCommutBinaryFormulas, term) ||
        !checkedAdd128(total, term, total) ||
        !checkedAdd128(total, numOfNotCommutBinaryFormulas, total))
    {
      fprintf(stderr, "Error: d_numForm precomputation overflowed 128 bits at formula size %d.\n", i);
      overflow128 = true;
    }

    numForm[i * 9] = numOfUnaryFormulas;
    numForm[i * 9 + 1] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 2] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 3] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 4] = numOfUnaryFormulas;
    numForm[i * 9 + 5] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 6] = numOfNotCommutBinaryFormulas;
    numForm[i * 9 + 7] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 8] = total;

    for (int j = 0; j < 9; ++j)
    {
      if (!fitsUint64(numForm[i * 9 + j]))
        overflow64 = true;
    }

    unaryForm[i * 9] = numOfUnaryFormulas;
    unaryForm[i * 9 + 1] = 0;
    unaryForm[i * 9 + 2] = 0;
    unaryForm[i * 9 + 3] = 0;
    unaryForm[i * 9 + 4] = numOfUnaryFormulas;
    unaryForm[i * 9 + 5] = 0;
    unaryForm[i * 9 + 6] = 0;
    unaryForm[i * 9 + 7] = 0;
    unaryForm[i * 9 + 8] = 2 * numOfUnaryFormulas;

    binaryForm[i * 9] = 0;
    binaryForm[i * 9 + 1] = numOfCommutBinaryFormulas;
    binaryForm[i * 9 + 2] = numOfCommutBinaryFormulas;
    binaryForm[i * 9 + 3] = numOfCommutBinaryFormulas;
    binaryForm[i * 9 + 4] = 0;
    binaryForm[i * 9 + 5] = numOfCommutBinaryFormulas;
    binaryForm[i * 9 + 6] = numOfNotCommutBinaryFormulas;
    binaryForm[i * 9 + 7] = numOfCommutBinaryFormulas;
    binaryForm[i * 9 + 8] = 5 * numOfCommutBinaryFormulas +
                                                    numOfNotCommutBinaryFormulas;
  }

  for (int i = 1; i <= max_size; ++i)
  {
    LARGE_UINT cumulative = 0;
    for (int j = 0; j < 9; ++j)
    {
      if (!checkedAdd128(cumulative, numForm[i * 9 + j], cumulative))
      {
        fprintf(stderr, "Error: d_numForm cumulative precomputation overflowed 128 bits at formula size %d.\n", i);
        overflow128 = true;
      }
      numFormCumulative[i * 9 + j] = cumulative;
    }
    //numFormCumulative[i * 9 + 8] = numForm[i * 9 + 8];
  }

  if (overflow64)
    fprintf(stderr, "Error: formula count exceeds 64 bits; kernel launch sizes may be invalid.\n");
  if (overflow128)
    fprintf(stderr, "Error: d_numForm values after the first 128-bit overflow are invalid.\n");

  printMatrix(numForm.data(), max_size + 1, 9);
  return {numForm, unaryForm, binaryForm, numFormCumulative};
}

std::string MBAToString(const char *MBAFormula)
{
  std::stack<std::string> stack;
  for (int i = 0; MBAFormula[i] != '\0'; ++i)
  {
    char token = MBAFormula[i];

    if (token >= '0' && token <= '9')
    {
      stack.push("v" + std::string(1, token));
    }
    else
    {
      if (token == '~' || token == '#')
      {
        if (!stack.empty())
        {
          std::string operand = stack.top();
          stack.pop();
          std::string op = (token == '#') ? "-" : "~";
          stack.push("(" + op + operand + ")");
        }
      }
      else
      {
        if (stack.size() < 2)
          return std::format("Error : Incorrect formula (stack size < 2). {}\n",
                             MBAFormula);
        std::string right = stack.top();
        stack.pop();
        std::string left = stack.top();
        stack.pop();
        stack.push("(" + left + " " + std::string(1, token) + " " + right +
                   ")");
      }
    }
  }

  if (stack.size() != 1)
    return std::format("Error : Incorrect formula (stack size != 2). {}\n",
                       MBAFormula);
  return stack.top();
}

class NoCache
{
public:
  static std::string solve(const MbaInstance &mba, const int max_size)
  {
    if (max_size > maxFormulaSize)
    {
      throw std::format("This version supports formulas of size at most {}",
                        maxFormulaSize);
    }

    if (mba.n_examples > maxNumOfSamples)
    {
      throw std::format("This version supports at most {} samples",
                        maxNumOfSamples);
    }

    if (mba.n_variables > maxNumOfVars)
    {
      throw std::format("This version supports at most {} variables",
                        maxNumOfVars);
    }

    // Copying number of vars, inputs and outputs into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_inputs, mba.inputs.data(),
                                 mba.inputs.size() * sizeof(uint32_t)));
    checkCuda(cudaMemcpyToSymbol(d_outputs, mba.outputs.data(),
                                 mba.outputs.size() * sizeof(uint32_t)));

    // Number of generated formulas
    uint64_t allMBAs{};

#ifndef MEASUREMENT_MODE
    fprintf(stderr,
            "Length %-2d | Vars  | CheckedMBAs: %-13lu | ToBeChecked: %-12d \n",
            1, allMBAs, mba.n_variables);
#endif

    // Number of formulas matrix
    std::vector<LARGE_UINT> numForm;
    std::vector<LARGE_UINT> unaryForm;
    std::vector<LARGE_UINT> binaryForm;
    std::vector<LARGE_UINT> numFormCumulative;
    std::tie(numForm, unaryForm, binaryForm, numFormCumulative) = compute_n_formulas(max_size, mba.n_variables);
    checkCuda(cudaMemcpyToSymbol(d_numForm, numFormCumulative.data(),
                                 (max_size + 1) * 9 * sizeof(LARGE_UINT)));
    LARGE_UINT numFormUncum[max_size + 1];
    for (size_t i = 0; i < max_size + 1; i++)
    {
      numFormUncum[i] = numForm[i * 9 + 8];
    }
    checkCuda(cudaMemcpyToSymbol(d_numFormUncum, numFormUncum,
                                 (max_size + 1) * sizeof(LARGE_UINT)));

    LARGE_UINT numFormP[max_size - 1][maxFormulaSize - 1];
    for (size_t i = 1; i < max_size; i++)
    {
      for (size_t j =  i; j < max_size; j++)
      {
        numFormP[j - 1][i - 1] = numForm[i * 9 + 8] * numForm[(j - i) * 9 + 8];
      }
      
    }
    checkCuda(cudaMemcpyToSymbol(d_numFormP, &numFormP,
                                 (maxFormulaSize - 1) * (max_size - 1) * sizeof(LARGE_UINT)));

    // Memory allocation for the potential solution
    char *d_res;
    checkCuda(cudaMalloc(&d_res, maxFormulaSize * sizeof(char)));
    checkCuda(cudaMemset(d_res, 0, maxFormulaSize * sizeof(char)));
    char res[maxFormulaSize] = {0};

    LARGE_UINT offset;
    LARGE_UINT N, U, B;
    LARGE_UINT blockSize;
    bool stop = false;
    for (int formula_size = 1; formula_size < max_size && !stop;
         ++formula_size)
    {

      offset = 0;

      for (int i = 0; i < 8; ++i)
      {

        N = numForm[formula_size * 9 + i];
        U = unaryForm[formula_size * 9 + i];
        B = binaryForm[formula_size * 9 + i];

        blockSize = cuda::ceil_div(N - U / 2 -  4 * B / 5, 1024);
#ifndef MEASUREMENT_MODE
        fprintf(
            stderr,
            "Length %-2d | %-5s | CheckedMBAs: %-13lu | ToBeChecked: %-12lu \n",
            formula_size, opStr[i], allMBAs, N);
#endif
        if (N > 0)
          processOperator<<<blockSize, 1024>>>(mba.n_examples, mba.n_variables,
                                               formula_size, offset, N, d_res);
        offset += N;
        allMBAs += N;

        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaMemcpy(res, d_res, maxFormulaSize * sizeof(char),
                             cudaMemcpyDeviceToHost));

        if (res[0] != '\0')
        {
          stop = true;
          break;
        }
      }
    }

    std::string output;

    if (res[0] != '\0')
    {
      output = MBAToString(res);
    }
    else
    {
      output = "Not found !";
    }

    checkCuda(cudaFree(d_res));
    return output;
  }
};

int main(int argc, char *argv[])
{

  // -----------------
  // Reading the input
  // -----------------

  if (argc != 3)
  {
    fprintf(stderr, "Arguments should be in the form of\n");
    fprintf(
        stderr,
        "-----------------------------------------------------------------\n");
    fprintf(stderr, "%s <input_file_address> <maxLen>\n", argv[0]);
    fprintf(
        stderr,
        "-----------------------------------------------------------------\n");
    return 0;
  }

  if (atoi(argv[2]) < 1 || atoi(argv[2]) > 50)
  {
    fprintf(stderr, "Argument maxLen = %s should be between 1 and %d", argv[2],
            maxFormulaSize);
    return 0;
  }

  std::string file_name = argv[1];
  auto mba = MbaInstance::from_json_file(file_name);
  int max_size = atoi(argv[2]);

  // ------------------------------
  // Mixed Boolean Arithmetic (MBA)
  // ------------------------------

#ifdef MEASUREMENT_MODE
  auto start = std::chrono::high_resolution_clock::now();
#endif

  std::string output = NoCache::solve(mba, max_size);
  if (output == "see_the_error")
    return 0;

#ifdef MEASUREMENT_MODE
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
          .count();
  double time_s = (double)duration / 1e6;
  fprintf(stderr, "\nRunning Time: %f s,", (double)duration * 0.000001);
#endif

  std::cout << file_name << "," << output << "," << time_s << std::endl;
  return 0;
}
