#include "globals.cuh"
#include "instance.h"
#include "mba.cuh"
#include "utils.cuh"
#include <cassert>
#include <cstdint>
#include <fmt/core.h>
#include <stack>
#include <vector>

struct StackEntry {
  uint64_t n;
  int size;
  // int shift;
};

__device__ void printFormula(int size, char formula[max_formula_size]) {
  for (int i = 0; i < size; ++i)
    printf("%c ", formula[i]);
  printf("\n");
}

template <class OpGroup>
__host__ __device__ void numberToFormula(uint64_t n, int size,
                                         char formula[max_formula_size]) {

  StackEntry stack[max_formula_size];
  int shift = 0;
  stack[0] = {n, size};
  // stack[0] = {n, size, shift};
  int stackIdx = 0;

  uint64_t opPartsum;
  uint64_t fPartsum;
  uint64_t rightsum;

  while (stackIdx >= 0) {

    StackEntry entry = stack[stackIdx--];
    n = entry.n;
    size = entry.size;
    // shift = entry.shift;

    if (size == 1)
      formula[shift] = '0' + n;

    else {
      uint64_t opPartsum = 0;
      uint64_t opIdx = 0;
      uint64_t fIdx = 0;

      // Find the next operator
      while (n >= opPartsum + d_numForm[size][opIdx])
        opPartsum += d_numForm[size][opIdx++];
      while (d_numForm[size][opIdx] == 0)
        opIdx--;
      n -= opPartsum;

      size--;
      formula[shift + size] = OpGroup::char_representation[opIdx];

      if (opIdx == 0 || opIdx == 4)
        stack[++stackIdx] = {n, size};

      else {

        fPartsum = 0;
        fIdx = 1;

        // Operator is binary : Find the sizes of the operands
        while (n >= fPartsum + d_numForm[fIdx][8] * d_numForm[size - fIdx][8]) {
          fPartsum += d_numForm[fIdx][8] * d_numForm[size - fIdx][8];
          fIdx++;
        }
        n -= fPartsum;

        rightsum = d_numForm[size - fIdx][8];
        stack[++stackIdx] = {n / rightsum, fIdx, shift};
        stack[++stackIdx] = {n % rightsum, size - fIdx, shift + fIdx};
      }
    }
  }
}

__global__ void processOperator(const int n_examples, const int n_variables,
                                int formula_size, uint64_t offset,
                                uint64_t n_threads, char *d_res) {

  uint64_t tid =
      static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(blockIdx.x) +
      static_cast<uint64_t>(threadIdx.x);

  if (tid < n_threads) {

    tid += offset;

    char formula[max_formula_size];
    numberToFormula(tid, formula_size, formula);
    bool found =
        is_solution_rpn(formula, formula_size, n_examples, n_variables);

    if (found)
      for (int i = 0; i < formula_size; ++i)
        d_res[i] = formula[i];
  }
}

void printMatrix(uint64_t *matrix, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j)
      printf("%-12lu ", matrix[i * n + j]);
    printf("\n");
  }
}

std::vector<uint64_t> compute_n_formulas(int max_size, int n_variables) {

  // Last column is the number of formulas of the given size
  std::vector<uint64_t> numForm((max_size + 1) * 9, 0);

  // Initialisation with variables and unary operators
  numForm[1 * 9 + 8] = n_variables;
  numForm[2 * 9 + 0] = n_variables;
  numForm[2 * 9 + 4] = n_variables;
  numForm[2 * 9 + 8] = 2 * n_variables;

  // Generation
  uint64_t numOfUnaryFormulas;
  uint64_t numOfCommutBinaryFormulas;
  uint64_t numOfNotCommutBinaryFormulas;

  for (int i = 3; i <= max_size; ++i) {
    numOfUnaryFormulas = numForm[(i - 1) * 9 + 8];
    numOfCommutBinaryFormulas = 0;
    numOfNotCommutBinaryFormulas = 0;
    for (int j = 1; j <= (i - 1) / 2; ++j) {
      numOfCommutBinaryFormulas +=
          numForm[j * 9 + 8] * numForm[(i - j - 1) * 9 + 8];
    }
    for (int j = 1; j <= i - 2; ++j) {
      numOfNotCommutBinaryFormulas +=
          numForm[j * 9 + 8] * numForm[(i - j - 1) * 9 + 8];
    }
    numForm[i * 9] = numOfUnaryFormulas;
    numForm[i * 9 + 1] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 2] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 3] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 4] = numOfUnaryFormulas;
    numForm[i * 9 + 5] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 6] = numOfNotCommutBinaryFormulas;
    numForm[i * 9 + 7] = numOfCommutBinaryFormulas;
    numForm[i * 9 + 8] = 2 * numOfUnaryFormulas +
                         5 * numOfCommutBinaryFormulas +
                         numOfNotCommutBinaryFormulas;
  }

  printMatrix(numForm.data(), max_size + 1, 9);
  return numForm;
}

std::string MBAToString(const char *MBAFormula) {
  std::stack<std::string> stack;
  for (int i = 0; MBAFormula[i] != '\0'; ++i) {
    char token = MBAFormula[i];

    if (token >= '0' && token <= '9') {
      stack.push("v" + std::string(1, token));
    } else {
      if (token == '~' || token == '#') {
        if (!stack.empty()) {
          std::string operand = stack.top();
          stack.pop();
          std::string op = (token == '#') ? "-" : "~";
          stack.push("(" + op + operand + ")");
        }
      } else {
        if (stack.size() < 2)
          return "Error : Incorrect formula.\n";
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
    return "Error : Incorrect Formula.\n";
  return stack.top();
}

class NoCache {
public:
  static std::string solve(const MbaInstance &mba, const int max_size) {
    if (max_size > max_formula_size) {
      throw fmt::format("This version supports formulas of size at most {}",
                        max_formula_size);
    }

    if (mba.n_examples > max_n_examples) {
      throw fmt::format("This version supports at most {} samples",
                        max_n_examples);
    }

    if (mba.n_variables > max_n_variables) {
      throw fmt::format("This version supports at most {} variables",
                        max_n_variables);
    }

    // Copying number of vars, inputs and outputs into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_inputs, mba.inputs.data(),
                                 mba.inputs.size() * sizeof(uint32_t)));
    checkCuda(cudaMemcpyToSymbol(d_outputs, mba.outputs.data(),
                                 mba.outputs.size() * sizeof(uint32_t)));

    // Number of generated formulas
    uint64_t allMBAs{};

#ifndef MEASUREMENT_MODE
    printf("Length %-2d | Vars  | CheckedMBAs: %-13lu | ToBeChecked: %-12d \n",
           1, allMBAs, mba.n_variables);
#endif

    // Number of formulas matrix
    std::vector<uint64_t> numForm =
        compute_n_formulas(max_size, mba.n_variables);
    checkCuda(cudaMemcpyToSymbol(d_numForm, numForm.data(),
                                 (max_size + 1) * 9 * sizeof(uint64_t)));

    // Memory allocation for the potential solution
    char *d_res;
    checkCuda(cudaMalloc(&d_res, max_formula_size * sizeof(char)));
    checkCuda(cudaMemset(d_res, 0, max_formula_size * sizeof(char)));
    char res[max_formula_size] = {0};

    uint64_t offset;
    uint64_t N;
    uint64_t blockSize;
    bool stop = false;
    for (int formula_size = 1; formula_size < max_size && !stop;
         ++formula_size) {

      offset = 0;

      for (int i = 0; i < 8; ++i) {

        N = numForm[formula_size * 9 + i];
        blockSize = (N + 1023) / 1024;
#ifndef MEASUREMENT_MODE
        printf(
            "Length %-2d | %-5s | CheckedMBAs: %-13lu | ToBeChecked: %-12lu \n",
            formula_size, opStr[i], allMBAs, N);
#endif
        if (N > 0)
          processOperator<<<blockSize, 1024>>>(mba.n_examples, mba.n_variables,
                                               formula_size, offset, N, d_res);
        offset += N;
        allMBAs += N;

        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaMemcpy(res, d_res, max_formula_size * sizeof(char),
                             cudaMemcpyDeviceToHost));

        if (res[0] != '\0') {
          stop = true;
          break;
        }
      }
    }

    std::string output;

    if (res[0] != '\0') {
      output = MBAToString(res);
    } else {
      output = "Not found !";
    }

    checkCuda(cudaFree(d_res));
    return output;
  }
};

int main(int argc, char *argv[]) {

  // -----------------
  // Reading the input
  // -----------------

  if (argc != 3) {
    printf("Arguments should be in the form of\n");
    printf(
        "-----------------------------------------------------------------\n");
    printf("%s <input_file_address> <maxLen>\n", argv[0]);
    printf(
        "-----------------------------------------------------------------\n");
    return 0;
  }

  if (atoi(argv[2]) < 1 || atoi(argv[2]) > 50) {
    printf("Argument maxLen = %s should be between 1 and %d", argv[2],
           max_formula_size);
    return 0;
  }

  std::string file_name = argv[1];
  auto mba = MbaInstance::from_json_file(file_name);
  int max_size = atoi(argv[2]);

  // ------------------------------
  // Mixed Boolean Arithmetic (MBA)
  // ------------------------------

#ifdef MEASUREMENT_MODE
  auto start = chrono::high_resolution_clock::now();
#endif

  std::string output = NoCache::solve(mba, max_size);
  if (output == "see_the_error")
    return 0;

#ifdef MEASUREMENT_MODE
  auto stop = chrono::high_resolution_clock::now();
#endif

#ifdef MEASUREMENT_MODE
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
          .count();
  // printf("\nNumber of All LTLs: %lu", allLTLs);
  // printf("\nCost of Final LTL: %d", LTLcost);
  printf("\nRunning Time: %f s", (double)duration * 0.000001);
#endif

  printf("\nMBA: \"%s\"\n", output.c_str());
  return 0;
}