#include "globals.cuh"
#include "instance.h"
#include "mba.cuh"
#include "mba_ops.cuh"
#include "sygus_parser.h"
#include "utils.cuh"
#include <cassert>
#include <cstdint>
#include <fmt/core.h>
#include <stack>
#include <vector>

struct StackEntry {
  uint64_t n;
  int size;
  int shift;
};

__device__ void printFormula(int size, char formula[max_formula_size]) {
  for (int i = 0; i < size; ++i)
    printf("%c ", formula[i]);
  printf("\n");
}

// Reads which ops are active from d_activeOps/d_numForm (populated at
// runtime from an OpFlags - see build_active_ops below), rather than a
// compile-time OpGroup: any combination of enabled op groups is supported
// without a template instantiation per combination.
__host__ __device__ void numberToFormula(uint64_t n, int size, int n_variables,
                                         char formula[max_formula_size]) {

  StackEntry stack[max_formula_size];
  stack[0] = {n, size, 0};
  int stackIdx = 0;

  uint64_t opPartsum;
  uint64_t fPartsum;
  uint64_t rightsum;

  while (stackIdx >= 0) {

    StackEntry entry = stack[stackIdx--];
    n = entry.n;
    size = entry.size;
    int shift = entry.shift;

    if (size == 1)
      formula[shift] = (n < static_cast<uint64_t>(n_variables))
                          ? static_cast<char>('0' + n)
                          : static_cast<char>('A' + n - n_variables);

    else {
      uint64_t opPartsum = 0;
      uint64_t opIdx = 0;

      // Find the next operator
      while (n >= opPartsum + d_numForm[size][opIdx])
        opPartsum += d_numForm[size][opIdx++];
      while (d_numForm[size][opIdx] == 0)
        opIdx--;
      n -= opPartsum;

      size--;
      formula[shift + size] = d_activeOps[opIdx].unique_char;

      if (!d_activeOps[opIdx].is_binary)
        stack[++stackIdx] = {n, size, shift};

      else {

        fPartsum = 0;
        uint64_t fIdx = 1;

        // Operator is binary : Find the sizes of the operands
        while (n >= fPartsum + d_numForm[fIdx][max_n_ops] *
                                   d_numForm[size - fIdx][max_n_ops]) {
          fPartsum +=
              d_numForm[fIdx][max_n_ops] * d_numForm[size - fIdx][max_n_ops];
          fIdx++;
        }
        n -= fPartsum;

        rightsum = d_numForm[size - fIdx][max_n_ops];
        stack[++stackIdx] = {n / rightsum, static_cast<int>(fIdx), shift};
        stack[++stackIdx] = {n % rightsum, size - static_cast<int>(fIdx),
                             shift + static_cast<int>(fIdx)};
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
    numberToFormula(tid, formula_size, n_variables, formula);
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

// Builds the runtime list of active ops (and, host-side only, whether each
// commutes - needed by compute_n_formulas but not by the device, since
// numberToFormula's split logic already derives the right operand-size
// range purely from the per-op formula-count budget computed below).
template <class Op>
void maybe_add_op(std::vector<ActiveOp> &ops, std::vector<bool> &op_commutes,
                  const OpFlags &flags) {
  if (in_group<Op>(flags)) {
    ops.push_back({Op::unique_char, Op::is_binary});
    op_commutes.push_back(Op::commutes);
  }
}

template <class... Ops>
void build_active_ops(const OpFlags &flags, std::vector<ActiveOp> &ops,
                      std::vector<bool> &op_commutes) {
  (maybe_add_op<Ops>(ops, op_commutes, flags), ...);
}

// numForm[size][op] = number of formulas of exactly `size` tokens whose root
// is active_ops[op]; numForm[size][max_n_ops] = total formulas of that size.
// Sized to match d_numForm's fixed [max_formula_size+1][max_n_ops+1] layout
// regardless of how many ops are actually active, so it can be copied to
// device constant memory directly; unused trailing op columns stay 0.
std::vector<uint64_t> compute_n_formulas(int max_size, int n_leaves,
                                         const std::vector<ActiveOp> &active_ops,
                                         const std::vector<bool> &op_commutes) {

  const int cols = max_n_ops + 1;
  const int n_ops = static_cast<int>(active_ops.size());
  std::vector<uint64_t> numForm((max_size + 1) * cols, 0);

  numForm[1 * cols + max_n_ops] = n_leaves;

  for (int size = 2; size <= max_size; ++size) {
    uint64_t total = 0;
    for (int op = 0; op < n_ops; ++op) {
      uint64_t count;
      if (!active_ops[op].is_binary) {
        count = numForm[(size - 1) * cols + max_n_ops];
      } else if (op_commutes[op]) {
        count = 0;
        for (int j = 1; j <= (size - 1) / 2; ++j)
          count += numForm[j * cols + max_n_ops] *
                   numForm[(size - 1 - j) * cols + max_n_ops];
      } else {
        count = 0;
        for (int j = 1; j <= size - 2; ++j)
          count += numForm[j * cols + max_n_ops] *
                   numForm[(size - 1 - j) * cols + max_n_ops];
      }
      numForm[size * cols + op] = count;
      total += count;
    }
    numForm[size * cols + max_n_ops] = total;
  }

  printMatrix(numForm.data(), max_size + 1, cols);
  return numForm;
}

std::string MBAToString(const char *MBAFormula,
                        const std::vector<uint32_t> &constants) {
  std::stack<std::string> stack;
  for (int i = 0; MBAFormula[i] != '\0'; ++i) {
    char token = MBAFormula[i];

    if (token >= '0' && token <= '9') {
      stack.push("v" + std::string(1, token));
    } else if (token >= 'A' && token < 'A' + static_cast<int>(constants.size())) {
      stack.push(std::to_string(constants[token - 'A']));
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
  static std::string solve(const MbaInstance &mba, const int max_size,
                           const OpFlags &flags = OpFlags{},
                           const std::vector<uint32_t> &constants = {}) {
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

    if (static_cast<int>(constants.size()) > max_n_constants) {
      throw fmt::format("This version supports at most {} constants",
                        max_n_constants);
    }

    // Copying number of vars, inputs, outputs and constants into the
    // constant memory
    checkCuda(cudaMemcpyToSymbol(d_inputs, mba.inputs.data(),
                                 mba.inputs.size() * sizeof(uint32_t)));
    checkCuda(cudaMemcpyToSymbol(d_outputs, mba.outputs.data(),
                                 mba.outputs.size() * sizeof(uint32_t)));
    if (!constants.empty())
      checkCuda(cudaMemcpyToSymbol(d_constants, constants.data(),
                                   constants.size() * sizeof(uint32_t)));

    // Build the runtime active-op list from the flags, and upload it.
    std::vector<ActiveOp> active_ops;
    std::vector<bool> op_commutes;
    build_active_ops<Not, And, Or, Xor, LShift, RShift, AShr, Neg, Add, Sub,
                     Mul, Div, Mod, SDiv, SRem>(flags, active_ops,
                                                op_commutes);
    const int n_ops = static_cast<int>(active_ops.size());
    if (n_ops > max_n_ops) {
      throw fmt::format("This version supports at most {} active ops",
                        max_n_ops);
    }
    checkCuda(cudaMemcpyToSymbol(d_activeOps, active_ops.data(),
                                 active_ops.size() * sizeof(ActiveOp)));

    // Number of generated formulas
    uint64_t allMBAs{};

#ifndef MEASUREMENT_MODE
    printf("Length %-2d | Vars  | CheckedMBAs: %-13lu | ToBeChecked: %-12d \n",
           1, allMBAs, mba.n_variables);
#endif

    // Number of formulas matrix
    const int n_leaves = mba.n_variables + static_cast<int>(constants.size());
    std::vector<uint64_t> numForm =
        compute_n_formulas(max_size, n_leaves, active_ops, op_commutes);
    checkCuda(cudaMemcpyToSymbol(d_numForm, numForm.data(),
                                 (max_size + 1) * (max_n_ops + 1) *
                                     sizeof(uint64_t)));

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

      for (int i = 0; i < n_ops; ++i) {

        N = numForm[formula_size * (max_n_ops + 1) + i];
        blockSize = (N + 1023) / 1024;
        // #ifndef MEASUREMENT_MODE
        //         printf(
        //             "Length %-2d | %-5s | CheckedMBAs: %-13lu | ToBeChecked:
        //             %-12lu \n", formula_size, opStr[i], allMBAs, N);
        // #endif
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
      output = MBAToString(res, constants);
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

  if (argc != 3 && argc != 4) {
    printf("Arguments should be in the form of\n");
    printf(
        "-----------------------------------------------------------------\n");
    printf("%s <input_file_address> <maxLen> "
          "[bitwise,arithmetic,shift,division]\n",
          argv[0]);
    printf("<input_file_address> may be a .json MbaInstance or a SyGuS-BV "
          ".sl file.\n");
    printf("The optional trailing argument, if given, replaces the default "
          "enabled op groups.\n");
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
  int max_size = atoi(argv[2]);

  // Bitwise and arithmetic ops are enabled by default, matching this
  // binary's original fixed ReducedOps set (Not, Neg, Add, Sub, And, Or,
  // Xor, Mul); shift and division are opt-in.
  OpFlags flags{};
  std::vector<uint32_t> constants;

  MbaInstance mba;
  bool is_sygus_file =
      file_name.size() >= 3 && file_name.substr(file_name.size() - 3) == ".sl";
  if (is_sygus_file) {
    SygusParseResult parsed = parse_sygus_bv_file(file_name);
    mba = parsed.instance;
    flags = parsed.flags;
    constants = parsed.constants;
  } else {
    mba = MbaInstance::from_json_file(file_name);
  }

  if (argc == 4)
    flags = parse_op_flags(argv[3]);

  // ------------------------------
  // Mixed Boolean Arithmetic (MBA)
  // ------------------------------

#ifdef MEASUREMENT_MODE
  auto start = chrono::high_resolution_clock::now();
#endif

  std::string output = NoCache::solve(mba, max_size, flags, constants);
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