// full-cache.cu
// GPU-accelerated Mixed Boolean-Arithmetic (MBA) synthesiser.

// Uncomment to suppress all progress output and enable timing.
// #define MEASUREMENT_MODE

#ifdef MEASUREMENT_MODE
#include <chrono>
#endif

#include "compat.h"
#include "cs_cache.cuh"
#include "globals.cuh"
#include "instance.h"
#include "mba_ops.cuh"

#include <cassert>
#include <climits> // INT_MAX
#include <cstdint>
#include <cstdio>
#include <string>

#include <warpcore/hash_set.cuh>

using namespace std;

using MCsCache = CsCache; //<max_n_examples>;

// TODO: handle edge cases like below?
// else if constexpr (op == Op::Div) {
//   // Guard against division by zero: if any RHS sample is 0, fall back
//   // to the identity (copy LHS) to keep the result vector well-defined.
//   bool noZeros = true;
//   for (int i = 0; i < n_examples; ++i) {
//     if (R[i] == 0) {
//       noZeros = false;
//       break;
//     }
//     CS[i] = L[i] / R[i];
//   }
//   if (!noZeros)
//     for (int i = 0; i < n_examples; ++i)
//       CS[i] = L[i];
// } else if constexpr (op == Op::Mod) {
//   // Same zero-divisor guard as Div.
//   bool noZeros = true;
//   for (int i = 0; i < n_examples; ++i) {
//     if (R[i] == 0) {
//       noZeros = false;
//       break;
//     }
//     CS[i] = L[i] % R[i];
//   }
//   if (!noZeros)
//     for (int i = 0; i < n_examples; ++i)
//       CS[i] = L[i];
// }

// ---------------------------------------------------------------------------
// Main per-operator enumeration kernel
// ---------------------------------------------------------------------------

// Each CUDA thread handles one (left, right) pair of cache rows
// Non-commutative ops (Minus, Div, Mod, shifts) also evaluate the swapped pair
// in the same thread, stored at modTid+1 in the temp cache.
template <class Op>
__global__ void operator_kernel(const int start_left, const int n_left,
                                const int start_right, const int n_right,
                                const int n_examples, MCsCache &cache) {

  const int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int thread_id_y = blockDim.y * blockIdx.y + threadIdx.y;
  constexpr bool notCommut = !Op::commutes;

  if (thread_id_x >= n_left || thread_id_y >= n_right)
    return;

  // Non-commutative ops claim two consecutive temp slots per thread.
  const int tid = thread_id_y * n_left + thread_id_x;
  int formula_index = notCommut ? 2 * tid : tid;

  cache.insert_formula<Op>(formula_index, thread_id_x, thread_id_y);

  if (notCommut) {
    formula_index += 1;
    cache.insert_formula<Op>(formula_index, thread_id_y, thread_id_x);
  }
}

template <class Op>
bool process_operator(const int left_start_index, const int left_end_index,
                      const int right_start_index, const int right_end_index,
                      const int n_examples, MCsCache &cache) {

#ifndef MEASUREMENT_MODE
  printf(" (%c) | StoredMBAs: %-10lu | ", Op::char_representation,
         cache.get_size());
#endif

  constexpr bool isUnary = !Op::is_binary;

  const int n_formulas_left = left_end_index - left_start_index + 1;
  const int n_formulas_right =
      isUnary ? 1 : (right_end_index - right_start_index + 1);

  dim3 block_size(isUnary ? 256 : 16, isUnary ? 1 : 16);
  dim3 grid_size(
      (n_formulas_left + block_size.x - 1) / block_size.x,
      isUnary ? 1 : ((n_formulas_right + block_size.y - 1) / block_size.y));

  const int start_right = isUnary ? 0 : right_start_index;
  operator_kernel<Op><<<grid_size, block_size>>>(
      left_start_index, n_formulas_left, start_right, n_formulas_right,
      n_examples, cache);
  checkCuda(cudaPeekAtLastError());

  int solution = cache.get_res_index();

  if (solution == -1) {
    size_t temp_cache_size = n_formulas_left * n_formulas_left;
    cache.store_unique(temp_cache_size);
  }

  return solution != -1;
}

template <class... Ops>
int process_operators(const int left_start_index, const int left_end_index,
                      const int right_start_index, const int right_end_index,
                      const int n_examples, MCsCache &cache) {

  bool res_found = (process_operator<Ops>(left_start_index, left_end_index,
                                          right_start_index, right_end_index,
                                          n_examples, cache) ||
                    ...);

  return res_found;
}

class FullCache {
public:
  static string solve(const MbaInstance &mba, const int max_search_size) {
    if (mba.n_examples > max_n_examples) {
      throw compat_fmt::format("This version supports at most {} examples.",
                               max_n_examples);
    }

    // Copy outputs into constant memory so every thread can read them cheaply.
    checkCuda(cudaMemcpyToSymbol(d_outputs, mba.outputs.data(),
                                 mba.outputs.size() * sizeof(uint32_t)));

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------
    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

    const int MBACacheCapacity =
        maxAllocationSize / (mba.n_examples * sizeof(uint32_t));
    const int temp_MBACacheCapacity = MBACacheCapacity / 2;

    // All device buffers in one RAII object; destructor calls cudaFree for all.
    CsCache cache(MBACacheCapacity, temp_MBACacheCapacity, mba.n_examples);

    insert_variables<<<1, mba.n_variables>>>(&cache, mba.n_variables);

    // ----------------------------
    // Enumeration of the next MBAs
    // ----------------------------

    std::vector<int> start_index(max_search_size + 1, 0);
    start_index[1] = cache.get_size();

    bool done = false;
    for (int formula_size = 2; (!done) && formula_size <= max_search_size;
         ++formula_size) {

      for (int left_size = 1; left_size < formula_size - 1; ++left_size) {
        int right_size = formula_size - 1 - left_size;
        if (right_size < left_size)
          break;

        /// TODO: separate unary and binary operators?
        done = process_operators<Not, And, Or, Xor, LShift, RShift, Neg, Add,
                                 Sub, Mul, Div, Mod>(
            start_index[left_size - 1], start_index[left_size],
            start_index[right_size - 1], start_index[right_size],
            mba.n_examples, cache);

        if (done)
          break;
      }
    }

    auto res = cache.rebuild_formula();
    return string(res.begin(), res.end());
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

  if (atoi(argv[2]) < 1 || atoi(argv[2]) > 100) {
    printf("Argument maxLen = %s should be between 1 and 100", argv[2]);
    return 0;
  }

  string fileName = argv[1];
  int max_search_size = atoi(argv[2]);

  MbaInstance instance = MbaInstance::from_json_file(fileName);

  // ------------------------------
  // Mixed Boolean Arithmetic (MBA)
  // ------------------------------

#ifdef MEASUREMENT_MODE
  auto start = chrono::high_resolution_clock::now();
#endif

  string output = FullCache::solve(instance, max_search_size);

#ifdef MEASUREMENT_MODE
  auto stop = chrono::high_resolution_clock::now();
#endif

  // -------------------
  // Printing the output
  // -------------------

  printf("Input file : %s\n", fileName.c_str());
  printf("Max length : %d\n", max_search_size);
  if (output == "Not found !")
    printf("Result     : no expression found within length %d\n",
           max_search_size);
  else
    printf("Result     : expression found\n");

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