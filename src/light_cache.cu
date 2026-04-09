#include "globals.cuh"
#include "instance.h"
#include "mba.cuh"
#include "ops.cuh"
#include "utils.cuh"
#include <fmt/core.h>
#include <stack>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <warpcore/hash_set.cuh>

// #define MEASUREMENT_MODE

using std::min;
using std::stack;
using std::string;
using std::to_string;
using std::unordered_map;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

__device__ uint32_t simpleHash(uint32_t input) {
  input ^= input >> 16;
  input *= 0x85ebca6b;
  input ^= input >> 13;
  input *= 0xc2b2ae35;
  input ^= input >> 16;
  return input;
}

__device__ void makeUnqChk(uint32_t *CS, uint64_t &seed, const int n_examples) {

  for (int i = 0; i < n_examples; ++i) {
    uint64_t val = simpleHash(CS[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    val = (i & 1) ? (val >> 32 | val << 32) : val;
    seed ^= val;
  }
}

template <class hash_set_t>
__global__ void hashSetInit(const int n_examples, const int n_variables,
                            hash_set_t hashSet) {

  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t CS[maxNumOfSamples];

  for (int i = 0; i < n_examples; ++i) {
    CS[i] = d_inputs[i * n_variables + tid];
  }

  uint64_t seed{};
  makeUnqChk(CS, seed, n_examples);

  const auto group =
      warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
  hashSet.insert(seed, group);
}

template <class Op>
__device__ void evaluateFormula(uint32_t *CS, char *left_formula,
                                char *right_formula, const int num_samples,
                                const int n_variables) {

  if constexpr (Op::is_binary) {
    for (int i = 0; i < num_samples; ++i) {
      uint32_t left = evaluateRPN(left_formula, i, n_variables);
      uint32_t right = evaluateRPN(right_formula, i, n_variables);
      CS[i] = Op::evaluate(left, right);
    }
  } else {
    for (int i = 0; i < num_samples; ++i) {
      uint32_t left = evaluateRPN(left_formula, i, n_variables);
      CS[i] = Op::evaluate(left);
    }
  }
}

template <class hash_set_t>
__device__ bool processUniqueCS(uint32_t *CS, const int n_examples,
                                hash_set_t &hashSet) {

  uint64_t seed{};
  makeUnqChk(CS, seed, n_examples);

  const auto group =
      warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
  return (hashSet.insert(seed, group) > 0) ? false : true;
}

template <class Op>
__device__ void insertInCache(uint32_t *CS, bool CSUniq, int tid, int ldx,
                              int rdx, int llen, int rlen, int len,
                              const int n_examples, char *d_MBACache,
                              char *d_temp_MBACache, char *d_MBAFormula) {

  if (CSUniq) {

    for (int i = 0; i < llen - 1; ++i)
      d_temp_MBACache[tid * len + i] = d_MBACache[ldx + i];
    for (int i = 0; i < rlen - 1; ++i)
      d_temp_MBACache[tid * len + i + llen - 1] = d_MBACache[rdx + i];
    d_temp_MBACache[tid * len + len - 2] = Op::char_representation;
    d_temp_MBACache[tid * len + len - 1] = '\0';

    bool found = true;
    for (int i = 0; found && i < n_examples; ++i)
      found = (CS[i] == d_outputs[i]);
    if (found) {
      for (int i = 0; i < len; ++i)
        d_MBAFormula[i] = d_temp_MBACache[tid * len + i];
    }

  } else {

    for (int i = 0; i < len; ++i)
      d_temp_MBACache[tid * len + i] = '?';
  }
}

template <class Op, class hash_set_t>
__global__ void processOperator(char *d_MBACache, char *d_temp_MBACache,
                                const int ldx1, const int ldx2, const int llen,
                                const int rdx1, const int rdx2, const int rlen,
                                const int n_examples, const int n_variables,
                                char *d_MBAFormula, hash_set_t hashSet) {

  const int lnumOfFormulas = (ldx2 - ldx1 + 1) / llen;
  const int rnumOfFormulas = (rdx2 - rdx1 + 1) / rlen;
  const int tid = (blockDim.x * blockIdx.x + threadIdx.x);
  constexpr bool isUnary = !Op::is_binary;
  constexpr bool notCommut = !Op::commutes;
  int maxTid = isUnary ? lnumOfFormulas : lnumOfFormulas * rnumOfFormulas;

  if (tid < maxTid) {

    int ldx = isUnary ? ldx1 + tid * llen : ldx1 + tid / rnumOfFormulas * llen;
    int rdx = isUnary ? 0 : rdx1 + tid % rnumOfFormulas * rlen;
    int modTid = notCommut ? 2 * tid : tid;
    int len = llen + rlen;

    uint32_t CS[maxNumOfSamples];
    evaluateFormula<Op>(CS, d_MBACache + ldx, d_MBACache + rdx, n_examples,
                        n_variables);
    bool CSUniq = processUniqueCS(CS, n_examples, hashSet);
    insertInCache<Op>(CS, CSUniq, modTid, ldx, rdx, llen, rlen, len, n_examples,
                      d_MBACache, d_temp_MBACache, d_MBAFormula);

    if constexpr (notCommut) {

      modTid += 1;
      evaluateFormula<Op>(CS, d_MBACache + rdx, d_MBACache + ldx, n_examples,
                          n_variables);
      CSUniq = processUniqueCS(CS, n_examples, hashSet);
      insertInCache<Op>(CS, CSUniq, modTid, rdx, ldx, rlen, llen, len,
                        n_examples, d_MBACache, d_temp_MBACache, d_MBAFormula);
    }
  }
}

bool storeUniqueMBAs(int N, uint64_t &lastIdx, int MBALen, const int n_examples,
                     const int MBACacheCapacity, char *d_MBACache,
                     char *d_temp_MBACache) {

  int space = N * (MBALen + 1);

  thrust::device_ptr<char> new_end_ptr;
  thrust::device_ptr<char> d_MBACache_ptr(d_MBACache + lastIdx);
  thrust::device_ptr<char> d_temp_MBACache_ptr(d_temp_MBACache);

  new_end_ptr =
      thrust::remove(d_temp_MBACache_ptr, d_temp_MBACache_ptr + space, '?');
  space = static_cast<int>(new_end_ptr - d_temp_MBACache_ptr);

  if (lastIdx + space > MBACacheCapacity) {
    return true;
  } else {
    thrust::copy_n(d_temp_MBACache_ptr, space, d_MBACache_ptr);
    lastIdx += space;
    return false;
  }
}

template <class Op, class hash_set_t>
bool generateMBAs(int MBALen, int *startPoints, const int MBACacheCapacity,
                  const int temp_MBACacheCapacity, uint64_t &allMBAs,
                  uint64_t &lastIdx, const int n_examples,
                  const int n_variables, char *d_MBACache,
                  char *d_temp_MBACache, char *d_MBAFormula, char *MBAFormula,
                  hash_set_t hashSet) {

  constexpr bool isUnary = !Op::is_binary;
  constexpr bool notCommut = !Op::commutes;

  if constexpr (isUnary) {
    int idx1 = startPoints[MBALen - 1];
    int idx2 = startPoints[MBALen] - 1;
    int N = (idx2 - idx1 + 1) / MBALen;
    int lim = temp_MBACacheCapacity / (MBALen + 1);

    if (N) {
      int x = idx1, y;
      do {
        y = x + min(lim * MBALen - 1, idx2 - x);
        N = (y - x + 1) / MBALen;
#ifndef MEASUREMENT_MODE
        printf(
            "Length %-2d | (%c) | StoredMBAs: %-11lu | ToBeChecked: %-10d \n",
            MBALen, Op::char_representation, allMBAs, N);
#endif
        int block_size = (N + 1023) / 1024;
        processOperator<Op><<<block_size, 1024>>>(
            d_MBACache, d_temp_MBACache, x, y, MBALen, 0, 0, 1, n_examples,
            n_variables, d_MBAFormula, hashSet);
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaMemcpy(MBAFormula, d_MBAFormula, 64 * sizeof(char),
                             cudaMemcpyDeviceToHost));
        if (MBAFormula[0] != '\0')
          return true;
        uint64_t oldIdx = lastIdx;
        if (storeUniqueMBAs(N, lastIdx, MBALen, n_examples, MBACacheCapacity,
                            d_MBACache, d_temp_MBACache))
          return true;
        allMBAs += (lastIdx - oldIdx) / (MBALen + 1);
        x = y + 1;
      } while (y < idx2);
    }

  } else {

    for (int i = 1; 2 * i <= MBALen - 1; ++i) {

      int idx1 = startPoints[i];
      int idx2 = startPoints[i + 1] - 1;
      int idx3 = startPoints[MBALen - i - 1];
      int idx4 = startPoints[MBALen - i] - 1;
      int N =
          ((idx4 - idx3 + 1) / (MBALen - i)) * ((idx2 - idx1 + 1) / (i + 1));
      int lim = temp_MBACacheCapacity /
                (((idx2 - idx1 + 1) / (i + 1)) * (MBALen + 1));
      if (notCommut)
        lim /= 2;

      if (N) {
        int x = idx3, y;
        do {
          y = x + min(lim * (MBALen - i) - 1, idx4 - x);
          N = (y - x + 1) * (idx2 - idx1 + 1) / ((i + 1) * (MBALen - i));
          int modN = notCommut ? 2 * N : N;
#ifndef MEASUREMENT_MODE
          printf(
              "Length %-2d | (%c) | StoredMBAs: %-11lu | ToBeChecked: %-10d \n",
              MBALen, Op::char_representation, allMBAs, modN);
#endif
          int Blc = (N + 1023) / 1024;
          processOperator<Op><<<Blc, 1024>>>(
              d_MBACache, d_temp_MBACache, idx1, idx2, i + 1, x, y, MBALen - i,
              n_examples, n_variables, d_MBAFormula, hashSet);
          checkCuda(cudaPeekAtLastError());
          checkCuda(cudaMemcpy(MBAFormula, d_MBAFormula, 64 * sizeof(char),
                               cudaMemcpyDeviceToHost));
          if (MBAFormula[0] != '\0')
            return true;
          uint64_t oldIdx = lastIdx;
          if (storeUniqueMBAs(modN, lastIdx, MBALen, n_examples,
                              MBACacheCapacity, d_MBACache, d_temp_MBACache))
            return true;
          allMBAs += (lastIdx - oldIdx) / (MBALen + 1);
          x = y + 1;
        } while (y < idx4);
      }
    }
  }

  startPoints[MBALen + 1] = lastIdx;
  return false;
}

string MBAToString(const char *MBAFormula) {

  stack<string> stack;
  unordered_map<char, string> opToStr = {
      {'~', "~"}, {'&', "&"}, {'|', "|"}, {'^', "^"}, {'<', "<<"}, {'>', ">>"},
      {'#', "-"}, {'+', "+"}, {'-', "-"}, {'*', "*"}, {'/', "/"},  {'%', "%"}};

  for (int i = 0; MBAFormula[i] != '\0'; ++i) {

    char token = MBAFormula[i];

    if (token >= '0' && token <= '9') {
      stack.push("v" + string(1, token));
    } else {
      if (token == '~' || token == '#') {
        if (!stack.empty()) {
          string operand = stack.top();
          stack.pop();
          stack.push("(" + opToStr[token] + operand + ")");
        }
      } else {
        if (stack.size() < 2)
          return "Error : Incorrect formula.\n";
        string right = stack.top();
        stack.pop();
        string left = stack.top();
        stack.pop();
        stack.push("(" + left + " " + opToStr[token] + " " + right + ")");
      }
    }
  }

  if (stack.size() != 1)
    return "Error : Incorrect Formula.\n";
  return stack.top();
}

class NoCache {
public:
  static string solve(const MbaInstance &mba, const int max_size) {
    if (max_size > maxFormulaSize) {
      throw fmt::format("This version supports formulas of size at most {}",
                        maxFormulaSize);
    }

    if (mba.n_examples > maxNumOfSamples) {
      throw fmt::format("This version supports at most {} samples",
                        maxNumOfSamples);
    }

    if (mba.n_variables > maxNumOfVars) {
      throw fmt::format("This version supports at most {} variables",
                        maxNumOfVars);
    }

    // Copying number of vars, inputs and outputs into the constant memory
    checkCuda(cudaMemcpyToSymbol(d_inputs, mba.inputs.data(),
                                 mba.inputs.size() * sizeof(uint32_t)));
    checkCuda(cudaMemcpyToSymbol(d_outputs, mba.outputs.data(),
                                 mba.outputs.size() * sizeof(uint32_t)));

    // Creating the cache
    char *MBACache = new char[mba.n_variables * 2];
    // Number of generated formulas
    uint64_t allMBAs{};

    // Index of the last free position in the cache
    uint64_t lastIdx{};

#ifndef MEASUREMENT_MODE
    printf("Length %-2d | Vars  | StoredMBAs: %-11lu | ToBeChecked: %-10d \n",
           1, allMBAs, mba.n_variables);
#endif

    // Initializing the cache with variables
    for (int i = 0; i < mba.n_variables; ++i) {
      MBACache[lastIdx++] = '0' + i;
      MBACache[lastIdx++] = '\0';
      bool found = true;
      for (int j = 0; j < mba.n_examples; j++) {
        if (!(mba.inputs[j * mba.n_variables + i] == mba.outputs[j]))
          found = false;
      }
      allMBAs++;
      if (found)
        return "v" + to_string(i);
    }

    // ------------------------------------
    // Memory allocation and initialisation
    // ------------------------------------

    int maxAllocationSize;
    cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

    const int MBACacheCapacity = maxAllocationSize / sizeof(char);
    const int temp_MBACacheCapacity = MBACacheCapacity / 2;

    int *startPoints = new int[max_size + 2]();
    startPoints[1] = 0;
    startPoints[2] = lastIdx;

    char *d_MBAFormula;
    char *MBAFormula = new char[64];
    MBAFormula[0] = '\0';
    checkCuda(cudaMalloc(&d_MBAFormula, 64 * sizeof(char)));

    char *d_MBACache, *d_temp_MBACache;
    checkCuda(cudaMalloc(&d_MBACache, MBACacheCapacity * sizeof(char)));
    checkCuda(
        cudaMalloc(&d_temp_MBACache, temp_MBACacheCapacity * sizeof(char)));
    checkCuda(cudaMemcpy(d_MBACache, MBACache,
                         2 * mba.n_variables * sizeof(char),
                         cudaMemcpyHostToDevice));

    using hash_set_t =
        warpcore::HashSet<uint64_t,        // Key type
                          uint64_t(0) - 1, // Empty key
                          uint64_t(0) - 2, // Tombstone key
                          warpcore::probing_schemes::QuadraticProbing<
                              warpcore::hashers::MurmurHash<uint64_t>>>;

    hash_set_t hashSet(MBACacheCapacity / 5);
    hashSet.init();
    hashSetInit<<<1, mba.n_variables>>>(mba.n_examples, mba.n_variables,
                                        hashSet);

    // ----------------------------
    // Enumeration of the next MBAs
    // ----------------------------

    bool lastRound = false;

    for (int MBALen = 2; MBALen <= max_size; ++MBALen) {

      lastRound = generateMBAs<Not>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<And>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<Or>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<Xor>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<LShift>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<RShift>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<Neg>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<Plus>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<Minus>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<Mul>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<Div>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;

      lastRound = generateMBAs<Mod>(
          MBALen, startPoints, MBACacheCapacity, temp_MBACacheCapacity, allMBAs,
          lastIdx, mba.n_examples, mba.n_variables, d_MBACache, d_temp_MBACache,
          d_MBAFormula, MBAFormula, hashSet);
      if (lastRound)
        break;
    }

    string output;

    if (MBAFormula[0] != '\0') {
      output = MBAToString(MBAFormula);
    } else {
      output = "Not found !";
    }

    // Cleanup
    cudaFree(d_MBACache);
    cudaFree(d_temp_MBACache);
    cudaFree(d_MBAFormula);

    return output;
  }
};

int main(int argc, char *argv[]) {

  // -----------------
  // Reading the input
  // -----------------

  if (argc != 3) {
    printf("Arguments should be in the form of\n");
    printf("-----------------------------------------------------------------"
           "\n");
    printf("%s <input_file_address> <max_size>\n", argv[0]);
    printf("-----------------------------------------------------------------"
           "\n");
    return 0;
  }

  if (atoi(argv[2]) < 1 || atoi(argv[2]) > 50) {
    printf("Argument max_size = %s should be between 1 and 50", argv[2]);
    return 0;
  }

  string file_name = argv[1];
  auto mba = MbaInstance::from_json_file(file_name);
  int max_size = atoi(argv[2]);

  // ------------------------------
  // Mixed Boolean Arithmetic (MBA)
  // ------------------------------

#ifdef MEASUREMENT_MODE
  auto start = chrono::high_resolution_clock::now();
#endif

  string output = NoCache::solve(mba, max_size);
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