#include "formula_cache.cuh"
#include "globals.cuh"
#include "hash.cuh"
#include "instance.h"
#include "mba.cuh"
#include "mba_ops.cuh"
#include "sygus_parser.h"
#include "utils.cuh"
#include <cstddef>
#include <cstdint>
#include <fmt/core.h>
#include <stack>
#include <vector>

// #define MEASUREMENT_MODE

using std::min;
using std::stack;
using std::string;
using std::to_string;
using std::unordered_map;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

using MFormulaCache = FormulaCache<max_formula_size>;
using MHashset = GpuHashset<max_n_examples>;

__global__ void process_variable(MFormulaCache &cache, MHashset hashset,
                                 const int n_examples, const int n_variables,
                                 const int n_constants) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= n_variables + n_constants)
    return;

  char *new_formula = cache.build_leaf(tid, n_variables);
  uint32_t CS[max_n_examples];
  for (int i = 0; i < n_examples; ++i) {
    CS[i] = evaluateRPN(new_formula, i, n_variables);
  }
  if (is_solution(CS, n_examples)) {
    cache.report_solution(tid);
  } else {
    bool inserted = hashset.insert(CS);
    cache.report_unique(tid, inserted);
  }
}

bool process_variables(MFormulaCache &cache, MHashset hashset,
                       const int n_examples, const int n_variables,
                       const int n_constants) {

  process_variable<<<1, n_variables + n_constants>>>(
      cache, hashset, n_examples, n_variables, n_constants);

  checkCuda(cudaPeekAtLastError());
  int solution = cache.get_solution();
  return solution != -1;
}

template <class Op>
__device__ void process_formula(int index, char *left_formula,
                                char *right_formula, MFormulaCache &cache,
                                MHashset hashset, const int n_examples,
                                const int n_variables) {

  char *new_formula =
      cache.build_formula<Op>(index, left_formula, right_formula);
  uint32_t CS[max_n_examples];
  for (int i = 0; i < n_examples; ++i) {
    CS[i] = evaluateRPN(new_formula, i, n_variables);
  }
  if (is_solution(CS, n_examples)) {
    cache.report_solution(index);
  } else {
    bool inserted = hashset.insert(CS);
    cache.report_unique(index, inserted);
  }
}

template <class Op>
__global__ void process_formulas(const int n_left, const int n_right,
                                 const int n_examples, const int n_variables,
                                 MFormulaCache &cache, MHashset hashset) {

  constexpr bool non_commutative = !Op::commutes;
  const int thread_id_x = (blockDim.x * blockIdx.x + threadIdx.x);
  const int thread_id_y = (blockDim.y * blockIdx.y + threadIdx.y);
  int thread_id = thread_id_y * n_right + thread_id_x;
  thread_id = non_commutative ? 2 * thread_id : thread_id;

  if (thread_id_x < n_left && thread_id_y < n_right) {
    char *left_formula = cache.get_formula(thread_id_x);
    char *right_formula = cache.get_formula(thread_id_y);
    process_formula<Op>(thread_id, left_formula, right_formula, cache, hashset,
                        n_examples, n_variables);

    if constexpr (non_commutative) {
      thread_id += 1;
      process_formula<Op>(thread_id, right_formula, left_formula, cache,
                          hashset, n_examples, n_variables);
    }
  }
}

template <class Op>
bool process_operator(const int n_examples, const int n_variables,
                      MFormulaCache &cache, MHashset hashset) {

  constexpr bool isUnary = !Op::is_binary;

  const int n_formulas = cache.get_n_formulas();

  dim3 block_size(isUnary ? 256 : 16, isUnary ? 1 : 16);
  dim3 grid_size((n_formulas + block_size.x - 1) / block_size.x,
                 isUnary ? 1
                         : ((n_formulas + block_size.y - 1) / block_size.y));

  const int n_right = isUnary ? 1 : n_formulas;
  process_formulas<Op><<<grid_size, block_size>>>(
      n_formulas, n_right, n_examples, n_variables, cache, hashset);
  checkCuda(cudaPeekAtLastError());

  int solution = cache.get_solution();

  if (solution == -1) {
    size_t temp_cache_size = n_formulas * n_right;
    cache.store_unique_mba(temp_cache_size);
  }

  return solution != -1;
}

template <class... Ops>
bool process_operators(const int n_examples, const int n_variables,
                       MFormulaCache &cache, MHashset hashset,
                       const OpFlags &flags) {
  bool res_found =
      ((in_group<Ops>(flags) &&
        process_operator<Ops>(n_examples, n_variables, cache, hashset)) ||
       ...);

  return res_found;
}

template <class... Ops>
string mba_to_string(const char *MBAFormula,
                     const std::vector<uint32_t> &constants) {

  std::stack<std::string> stack;
  unordered_map<char, string> op_to_str = {
      {Ops::unique_char, std::string(1, Ops::char_representation)}...};
  unordered_map<char, bool> op_is_unary = {
      {Ops::unique_char, !Ops::is_binary}...};

  for (int i = 0; MBAFormula[i] != '\0'; ++i) {

    char token = MBAFormula[i];

    if (token >= '0' && token <= '9') {
      stack.push("v" + string(1, token));
    } else if (token >= 'A' && token < 'A' + static_cast<int>(constants.size())) {
      stack.push(to_string(constants[token - 'A']));
    } else {
      if (op_is_unary[token]) {
        if (!stack.empty()) {
          string operand = stack.top();
          stack.pop();
          stack.push("(" + op_to_str[token] + operand + ")");
        }
      } else {
        if (stack.size() < 2)
          return "Error : Incorrect formula.\n";
        string right = stack.top();
        stack.pop();
        string left = stack.top();
        stack.pop();
        stack.push("(" + left + " " + op_to_str[token] + " " + right + ")");
      }
    }
  }

  if (stack.size() != 1)
    return "Error : Incorrect Formula.\n";
  return stack.top();
}

class NoCache {
public:
  static string solve(const MbaInstance &mba, const int max_size,
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

    // Number of generated formulas
    uint64_t allMBAs{};

    // Index of the last free position in the cache
    uint64_t lastIdx{};

#ifndef MEASUREMENT_MODE
    printf("Length %-2d | Vars  | StoredMBAs: %-11lu | ToBeChecked: %-10d \n",
           1, allMBAs, mba.n_variables);
#endif

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

    const int MBACacheCapacity = maxAllocationSize / sizeof(char);
    const int temp_MBACacheCapacity = MBACacheCapacity / 2;

    MFormulaCache cache(MBACacheCapacity, temp_MBACacheCapacity,
                        mba.n_examples);

    MHashset hashset(mba.n_examples, 2 * MBACacheCapacity);

    // ----------------------------
    // Enumeration of the next MBAs
    // ----------------------------

    int n_constants = static_cast<int>(constants.size());
    bool res_found = process_variables(cache, hashset, mba.n_examples,
                                       mba.n_variables, n_constants);

    for (int MBALen = 2; !res_found && MBALen <= max_size; ++MBALen) {
      res_found = process_operators<Not, And, Or, Xor, LShift, RShift, AShr,
                                    Neg, Add, Sub, Mul, Div, Mod, SDiv, SRem>(
          mba.n_examples, mba.n_variables, cache, hashset, flags);
    }

    if (res_found) {
      auto MBAFormula = cache.get_solution_ptr();
      return mba_to_string<Not, And, Or, Xor, LShift, RShift, AShr, Neg, Add,
                           Sub, Mul, Div, Mod, SDiv, SRem>(MBAFormula.data(),
                                                           constants);
    } else {
      return "Not Found";
    }
  }
};

int main(int argc, char *argv[]) {

  // -----------------
  // Reading the input
  // -----------------

  if (argc != 3 && argc != 4) {
    printf("Arguments should be in the form of\n");
    printf("-----------------------------------------------------------------"
           "\n");
    printf("%s <input_file_address> <max_size> "
           "[bitwise,arithmetic,shift,division]\n",
           argv[0]);
    printf("<input_file_address> may be a .json MbaInstance or a SyGuS-BV "
           ".sl file.\n");
    printf("The optional trailing argument, if given, replaces the default "
           "enabled op groups.\n");
    printf("-----------------------------------------------------------------"
           "\n");
    return 0;
  }

  if (atoi(argv[2]) < 1 || atoi(argv[2]) > 50) {
    printf("Argument max_size = %s should be between 1 and 50", argv[2]);
    return 0;
  }

  string file_name = argv[1];
  int max_size = atoi(argv[2]);

  // Every op is enabled by default, matching this binary's behavior before
  // op groups were introduced.
  OpFlags flags{/*bitwise=*/true, /*arithmetic=*/true, /*shift=*/true,
               /*division=*/true};
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

  string output = NoCache::solve(mba, max_size, flags, constants);
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