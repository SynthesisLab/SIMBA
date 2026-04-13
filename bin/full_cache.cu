// full-cache.cu
// GPU-accelerated Mixed Boolean-Arithmetic (MBA) synthesiser.

// Uncomment to suppress all progress output and enable timing.
// #define MEASUREMENT_MODE

#include <cstddef>
#ifdef MEASUREMENT_MODE
#include <chrono>
#endif

#include "full_cache.cuh"
#include "globals.cuh"
#include "hash.cuh"

#include <algorithm> // std::min
#include <cassert>
#include <climits> // INT_MAX
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory> // std::unique_ptr / std::make_unique
#include <string>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/remove.h>

#include <warpcore/hash_set.cuh>

using namespace std;

// ---------------------------------------------------------------------------
// Operator enumeration and evaluation
// ---------------------------------------------------------------------------

// All supported operators.  The order here determines how startPoints[] is
// indexed (numOps entries per depth level, one per Op).
enum class Op {
  Not,
  And,
  Or,
  Xor,
  LShift,
  RShift,
  Neg,
  Plus,
  Minus,
  Mul,
  Div,
  Mod
};

// Apply a unary operator (Op::Not or Op::Neg) element-wise from cache row ldx
// into CS.
template <Op op>
__device__ void applyUnary(uint32_t *CS, const uint32_t *cache, int ldx,
                           int n_examples) {
  static_assert(op == Op::Not || op == Op::Neg,
                "applyUnary called with non-unary op");
  const uint32_t *L = cache + ldx * n_examples;
  if constexpr (op == Op::Not)
    for (int i = 0; i < n_examples; ++i)
      CS[i] = ~L[i];
  else
    for (int i = 0; i < n_examples; ++i)
      CS[i] = -(int32_t)L[i];
}

// Apply a binary operator (all ops except Not and Neg) element-wise from cache
// rows ldx and rdx into CS.
// Division and modulo fall back to copying LHS when any RHS sample is zero.
template <Op op>
__device__ void applyBinary(uint32_t *CS, const uint32_t *cache, int ldx,
                            int rdx, int n_examples) {
  static_assert(op != Op::Not && op != Op::Neg,
                "applyBinary called with unary op");
  const uint32_t *L = cache + ldx * n_examples;
  const uint32_t *R = cache + rdx * n_examples;
  if constexpr (op == Op::And)
    for (int i = 0; i < n_examples; ++i)
      CS[i] = L[i] & R[i];
  else if constexpr (op == Op::Or)
    for (int i = 0; i < n_examples; ++i)
      CS[i] = L[i] | R[i];
  else if constexpr (op == Op::Xor)
    for (int i = 0; i < n_examples; ++i)
      CS[i] = L[i] ^ R[i];
  else if constexpr (op == Op::LShift)
    for (int i = 0; i < n_examples; ++i)
      CS[i] = L[i] << R[i];
  else if constexpr (op == Op::RShift)
    for (int i = 0; i < n_examples; ++i)
      CS[i] = L[i] >> R[i];
  else if constexpr (op == Op::Plus)
    for (int i = 0; i < n_examples; ++i)
      CS[i] = L[i] + R[i];
  else if constexpr (op == Op::Minus)
    for (int i = 0; i < n_examples; ++i)
      CS[i] = L[i] - R[i];
  else if constexpr (op == Op::Mul)
    for (int i = 0; i < n_examples; ++i)
      CS[i] = L[i] * R[i];
  else if constexpr (op == Op::Div) {
    // Guard against division by zero: if any RHS sample is 0, fall back
    // to the identity (copy LHS) to keep the result vector well-defined.
    bool noZeros = true;
    for (int i = 0; i < n_examples; ++i) {
      if (R[i] == 0) {
        noZeros = false;
        break;
      }
      CS[i] = L[i] / R[i];
    }
    if (!noZeros)
      for (int i = 0; i < n_examples; ++i)
        CS[i] = L[i];
  } else if constexpr (op == Op::Mod) {
    // Same zero-divisor guard as Div.
    bool noZeros = true;
    for (int i = 0; i < n_examples; ++i) {
      if (R[i] == 0) {
        noZeros = false;
        break;
      }
      CS[i] = L[i] % R[i];
    }
    if (!noZeros)
      for (int i = 0; i < n_examples; ++i)
        CS[i] = L[i];
  } else {
    []<bool flag = false>() {
      static_assert(flag, "Unhandled binary operator");
    }();
  }
}

// Thin router: dispatch to applyUnary or applyBinary based on op at compile
// time.
template <Op op>
__device__ void applyOperator(uint32_t *CS, uint32_t *d_MBACache, int ldx,
                              int rdx, const int n_examples) {
  if constexpr (op == Op::Not || op == Op::Neg)
    applyUnary<op>(CS, d_MBACache, ldx, n_examples);
  else
    applyBinary<op>(CS, d_MBACache, ldx, rdx, n_examples);
}

// Fingerprint CS and attempt to insert into the hash set.
// Returns true if the fingerprint was NOT already present (i.e. CS is unique).
template <class hash_set_t>
__device__ bool processUniqueCS(uint32_t *CS, const int n_examples,
                                hash_set_t &hashSet) {
  uint64_t seed{};
  hash_cs(CS, seed, n_examples);

  const auto group =
      warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
  // insert() returns 0 on duplicate, > 0 on successful insertion.
  return hashSet.insert(seed, group) > 0;
}

// ---------------------------------------------------------------------------
// Main per-operator enumeration kernel
// ---------------------------------------------------------------------------

// Each CUDA thread handles one (left, right) pair of cache rows:
//   - Unary ops : one thread per left operand (idx1..idx2).
//   - Binary ops: one thread per (left, right) pair from idx1..idx2 ×
//   idx3..idx4.
// Non-commutative ops (Minus, Div, Mod, shifts) also evaluate the swapped pair
// in the same thread, stored at modTid+1 in the temp cache.
template <Op op, class hash_set_t>
__global__ void processOperator(
    const int idx1, const int idx2, // left operand range in d_MBACache
    const int idx3, const int idx4, // right operand range (ignored for unary)
    const int n_examples, uint32_t *d_MBACache, uint32_t *d_temp_MBACache,
    bool *d_temp_boolCache, int *d_temp_leftIdx, int *d_temp_rightIdx,
    int *d_FinalMBAIdx, hash_set_t hashSet) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  constexpr bool isUnary = (op == Op::Not || op == Op::Neg);
  constexpr bool notCommut =
      (op == Op::Minus || op == Op::Div || op == Op::Mod || op == Op::LShift ||
       op == Op::RShift);
  const int maxTid =
      isUnary ? (idx2 - idx1 + 1) : (idx4 - idx3 + 1) * (idx2 - idx1 + 1);

  if (tid >= maxTid)
    return;

  // Decode flat tid into operand indices.
  const int ldx = isUnary ? idx1 + tid : idx1 + tid / (idx4 - idx3 + 1);
  const int rdx = isUnary ? 0 : idx3 + tid % (idx4 - idx3 + 1);
  // Non-commutative ops claim two consecutive temp slots per thread.
  const int modTid = notCommut ? 2 * tid : tid;
  uint32_t CS[max_n_examples];

  applyOperator<op>(CS, d_MBACache, ldx, rdx, n_examples);
  bool CS_is_unique = processUniqueCS(CS, n_examples, hashSet);
  insertInCache(CS_is_unique, CS, modTid, ldx, rdx, n_examples, d_temp_MBACache,
                d_temp_boolCache, d_temp_leftIdx, d_temp_rightIdx,
                d_FinalMBAIdx);

  if (notCommut) {
    // Also try op(right, left) for non-commutative operators.
    applyOperator<op>(CS, d_MBACache, rdx, ldx, n_examples);
    bool CS_is_unique2 = processUniqueCS(CS, n_examples, hashSet);
    insertInCache(CS_is_unique2, CS, modTid + 1, rdx, ldx, n_examples,
                  d_temp_MBACache, d_temp_boolCache, d_temp_leftIdx,
                  d_temp_rightIdx, d_FinalMBAIdx);
  }
}

// ---------------------------------------------------------------------------
// Cache compaction
// ---------------------------------------------------------------------------

// Compact the temporary cache into the permanent d_MBACache.
// Uses thrust to remove_if over the bool mask, then copies the surviving rows.
// Returns true if the cache is full (capacity overflow) – the caller should
// then stop the search.
bool storeUniqueMBAs(int N, int &lastIdx, const int n_examples,
                     const int MBACacheCapacity, FormulaCache &bufs) {
  thrust::device_ptr<uint32_t> d_MBACache_ptr(bufs.d_MBACache +
                                              lastIdx * n_examples);
  thrust::device_ptr<uint32_t> d_temp_MBACache_ptr(bufs.d_temp_MBACache);
  thrust::device_ptr<bool> d_temp_boolCache_ptr(bufs.d_temp_boolCache);
  thrust::device_ptr<int> d_leftIdx_ptr(bufs.d_leftIdx + lastIdx);
  thrust::device_ptr<int> d_rightIdx_ptr(bufs.d_rightIdx + lastIdx);
  thrust::device_ptr<int> d_temp_leftIdx_ptr(bufs.d_temp_leftIdx);
  thrust::device_ptr<int> d_temp_rightIdx_ptr(bufs.d_temp_rightIdx);

  auto new_end_ptr = thrust::remove_if(
      d_temp_MBACache_ptr, d_temp_MBACache_ptr + N * n_examples,
      d_temp_boolCache_ptr, thrust::logical_not<bool>());
  thrust::remove(d_temp_leftIdx_ptr, d_temp_leftIdx_ptr + N, -1);
  thrust::remove(d_temp_rightIdx_ptr, d_temp_rightIdx_ptr + N, -1);

  int nNew = static_cast<int>(new_end_ptr - d_temp_MBACache_ptr) / n_examples;
  if (lastIdx + nNew > MBACacheCapacity) {
    return true;
  } else {
    thrust::copy_n(d_temp_MBACache_ptr, nNew * n_examples, d_MBACache_ptr);
    thrust::copy_n(d_temp_leftIdx_ptr, nNew, d_leftIdx_ptr);
    thrust::copy_n(d_temp_rightIdx_ptr, nNew, d_rightIdx_ptr);
    lastIdx += nNew;
    return false;
  }
}

// ---------------------------------------------------------------------------
// Result reconstruction
// ---------------------------------------------------------------------------

// BFS from `index` through the left/right index tree stored in d_leftIdx /
// d_rightIdx.  Writes a flat list of (node, left, right) triples into
// d_FinalMBAIdx so the host can reconstruct the expression string.

// Maximum number of nodes that can be visited / recorded in one BFS call.
static const int MAX_RES_NODES = 100;

__global__ void generateResIndices(const int n_variables, const int index,
                                   const int *d_leftIdx, const int *d_rightIdx,
                                   int *d_FinalMBAIdx) {

  int resIdx = 0;
  while (d_FinalMBAIdx[resIdx] != -1)
    resIdx++;
  int queue[MAX_RES_NODES];
  queue[0] = index;
  int head = 0;
  int tail = 1;
  while (head < tail) {
    // Overflow guard: stop before writing out of bounds or overfilling queue.
    if (resIdx + 3 > 3 * MAX_RES_NODES || tail + 2 > MAX_RES_NODES)
      return;
    int mba = queue[head];
    int l = d_leftIdx[mba];
    int r = d_rightIdx[mba];
    d_FinalMBAIdx[resIdx++] = mba;
    d_FinalMBAIdx[resIdx++] = l;
    d_FinalMBAIdx[resIdx++] = r;
    if (l >= n_variables)
      queue[tail++] = l;
    if (r >= n_variables)
      queue[tail++] = r;
    head++;
  }
}

// String representation of each Op in declaration order.
const string opStrs[] = {"~", "&", "|", "^", "<<", ">>",
                         "-", "+", "-", "*", "/",  "%"};

// X-table of all operators in the order they are enumerated.
// Each row: (Op enum value, display label, startPoints column index 1..numOps).
// To add or remove an operator, edit only this table.
#define SIMBA_OPS(X)                                                           \
  X(Op::Not, "(~)  ", 1)                                                       \
  X(Op::And, "(&)  ", 2)                                                       \
  X(Op::Or, "(|)  ", 3)                                                        \
  X(Op::Xor, "(^)  ", 4)                                                       \
  X(Op::LShift, "(<<) ", 5)                                                    \
  X(Op::RShift, "(>>) ", 6)                                                    \
  X(Op::Neg, "(Neg)", 7)                                                       \
  X(Op::Plus, "(+)  ", 8)                                                      \
  X(Op::Minus, "(-)  ", 9)                                                     \
  X(Op::Mul, "(*)  ", 10)                                                      \
  X(Op::Div, "(/)  ", 11)                                                      \
  X(Op::Mod, "(%)  ", 12)

// Count the number of operators by expanding SIMBA_OPS as a sum of 1s.
#define COUNT_OP(op, lbl, idx) +1
constexpr int numOps = 0 SIMBA_OPS(COUNT_OP);
#undef COUNT_OP

// Recursively build the expression string for the node at `index`.
// - Leaf nodes (index < numVar) are printed as v0, v1, …
// - Internal nodes look up their operator from startPoints[] depth/offset,
//   then recurse on left and right children stored in indicesMap.
string toString(int index, map<int, pair<int, int>> &indicesMap,
                const int *startPoints) {

  if (index < numVar) {
    return "v" + to_string(index);
  }

  int i = 0;
  while (index >= startPoints[i]) {
    i++;
  }
  i--;

  int op = i % numOps;
  bool isUnary = (op == 0 || op == 6); // Op::Not or Op::Neg

  if (isUnary) {
    string res = toString(indicesMap[index].first, indicesMap, startPoints);
    return opStrs[op] + "(" + res + ")";
  }

  else {
    string left = toString(indicesMap[index].first, indicesMap, startPoints);
    string right = toString(indicesMap[index].second, indicesMap, startPoints);
    return "(" + left + ")" + opStrs[op] + "(" + right + ")";
  }
}

// Bring the left/right index tree for the solution from device to host and
// call toString() to produce the human-readable expression.
string MBAToString(const int FinalMBAIdx, const int *startPoints,
                   FormulaCache &bufs) {
  // Fetch the two immediate operand indices of the top-level node.
  int LIdx{}, RIdx{};
  checkCuda(cudaMemcpy(&LIdx, bufs.d_temp_leftIdx + FinalMBAIdx, sizeof(int),
                       cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(&RIdx, bufs.d_temp_rightIdx + FinalMBAIdx, sizeof(int),
                       cudaMemcpyDeviceToHost));

  int *d_resIndices;
  checkCuda(cudaMalloc(&d_resIndices, MAX_RES_NODES * sizeof(int)));
  thrust::device_ptr<int> d_resIndices_ptr(d_resIndices);
  thrust::fill(d_resIndices_ptr, d_resIndices_ptr + MAX_RES_NODES, -1);

  // BFS through the index tree for each child that is not a leaf.
  if (LIdx >= numVar)
    generateResIndices<<<1, 1>>>(numVar, LIdx, bufs.d_leftIdx, bufs.d_rightIdx,
                                 d_resIndices);
  if (RIdx >= numVar)
    generateResIndices<<<1, 1>>>(numVar, RIdx, bufs.d_leftIdx, bufs.d_rightIdx,
                                 d_resIndices);

  int resIndices[MAX_RES_NODES];
  checkCuda(cudaMemcpy(resIndices, d_resIndices, MAX_RES_NODES * sizeof(int),
                       cudaMemcpyDeviceToHost));
  cudaFree(d_resIndices);

  map<int, pair<int, int>> indicesMap;
  indicesMap.insert(make_pair(INT_MAX - 1, make_pair(LIdx, RIdx)));

  int i = 0;
  while (resIndices[i] != -1 && i + 2 < MAX_RES_NODES) {
    indicesMap.insert(make_pair(
        resIndices[i], make_pair(resIndices[i + 1], resIndices[i + 2])));
    i += 3;
  }

  if (i + 2 >= MAX_RES_NODES)
    return "Size of the output is too big !";
  return toString(INT_MAX - 1, indicesMap, startPoints);
}

// ---------------------------------------------------------------------------
// Per-operator enumeration (host)
// ---------------------------------------------------------------------------

// Accessor for the flat startPoints array, which is logically a 2D table:
//   startPoints[depth * numOps + opIdx]
// depth  = expression tree depth (1-based)
// opIdx  = 0 for the base entry of that depth, 1..numOps for each Op (see
// SIMBA_OPS)
inline int &startPt(int *sp, int depth, int opIdx = 0) {
  return sp[depth * numOps + opIdx];
}

template <Op op, class hash_set_t>
bool generateMBAs(int MBALen, int *startPoints, const int temp_MBACacheCapacity,
                  const int MBACacheCapacity, uint64_t &allMBAs, int &lastIdx,
                  const int n_examples, FormulaCache &bufs, int *FinalMBAIdx,
                  hash_set_t hashSet, const char *opStr, int opIdx) {
  constexpr bool isUnary = (op == Op::Not || op == Op::Neg);
  constexpr bool notCommut =
      (op == Op::Minus || op == Op::Div || op == Op::Mod || op == Op::LShift ||
       op == Op::RShift);

  if (isUnary) {

    int idx1 = startPt(startPoints, MBALen - 1);
    int idx2 = startPt(startPoints, MBALen) - 1;
    int N = idx2 - idx1 + 1;

    if (N) {
      int x = idx1, y;
      do {
        y = x + min(temp_MBACacheCapacity - 1, idx2 - x);
        N = y - x + 1;
#ifndef MEASUREMENT_MODE
        printf("Length %-2d | %s | AllMBAs: %-11lu | StoredMBAs: %-10d | "
               "ToBeChecked: %-10d \n",
               MBALen, opStr, allMBAs, lastIdx, N);
#endif
        int Blc = (N + 1023) / 1024;
        processOperator<op, hash_set_t><<<Blc, 1024>>>(
            x, y, 0, 0, n_examples, bufs.d_MBACache, bufs.d_temp_MBACache,
            bufs.d_temp_boolCache, bufs.d_temp_leftIdx, bufs.d_temp_rightIdx,
            bufs.d_FinalMBAIdx, hashSet);
        checkCuda(cudaPeekAtLastError());
        checkCuda(cudaMemcpy(FinalMBAIdx, bufs.d_FinalMBAIdx, sizeof(int),
                             cudaMemcpyDeviceToHost));
        allMBAs += N;
        if (*FinalMBAIdx != -1) {
          startPt(startPoints, MBALen, opIdx) = INT_MAX;
          return true;
        }
        if (storeUniqueMBAs(N, lastIdx, n_examples, MBACacheCapacity, bufs))
          return true;
        x = y + 1;
      } while (y < idx2);
    }
    startPt(startPoints, MBALen, opIdx) = lastIdx;

  } else {

    for (int i = 1; 2 * i <= MBALen - 1; ++i) {

      int idx1 = startPt(startPoints, i);
      int idx2 = startPt(startPoints, i + 1) - 1;
      int idx3 = startPt(startPoints, MBALen - i - 1);
      int idx4 = startPt(startPoints, MBALen - i) - 1;
      int N = (idx4 - idx3 + 1) * (idx2 - idx1 + 1);
      int modCap = notCommut ? 2 * (idx2 - idx1 + 1) : idx2 - idx1 + 1;

      if (N) {
        int x = idx3, y;
        do {
          y = x + min(temp_MBACacheCapacity / modCap - 1, idx4 - x);
          N = (y - x + 1) * (idx2 - idx1 + 1);
          int modN = notCommut ? 2 * N : N;
#ifndef MEASUREMENT_MODE
          printf("Length %-2d | %s | AllMBAs: %-11lu | StoredMBAs: %-10d | "
                 "ToBeChecked: %-10d \n",
                 MBALen, opStr, allMBAs, lastIdx, modN);
#endif
          int Blc = (N + 1023) / 1024;
          processOperator<op, hash_set_t><<<Blc, 1024>>>(
              idx1, idx2, x, y, n_examples, bufs.d_MBACache,
              bufs.d_temp_MBACache, bufs.d_temp_boolCache, bufs.d_temp_leftIdx,
              bufs.d_temp_rightIdx, bufs.d_FinalMBAIdx, hashSet);
          checkCuda(cudaPeekAtLastError());
          checkCuda(cudaMemcpy(FinalMBAIdx, bufs.d_FinalMBAIdx, sizeof(int),
                               cudaMemcpyDeviceToHost));
          allMBAs += modN;
          if (*FinalMBAIdx != -1) {
            startPt(startPoints, MBALen, opIdx) = INT_MAX;
            return true;
          }
          if (storeUniqueMBAs(modN, lastIdx, n_examples, MBACacheCapacity,
                              bufs))
            return true;
          x = y + 1;
        } while (y < idx4);
      }
    }

    startPt(startPoints, MBALen, opIdx) = lastIdx;
  }

  return false;
}

string MBA(const int maxLen, const int n_examples, uint32_t *inputData,
           uint32_t *outputData) {
  // ---------------------------------
  // Validate inputs
  // ---------------------------------

  if (n_examples > max_n_examples) {
    printf("This version supports at most %d samples.\n", max_n_examples);
    return "see_the_error";
  }

  // Copy outputs into constant memory so every thread can read them cheaply.
  checkCuda(cudaMemcpyToSymbol(d_outputData, outputData,
                               n_examples * sizeof(uint32_t)));

  // Host-side cache: each row i stores the n_examples output values for
  // variable i.
  auto MBACache = std::make_unique<uint32_t[]>(numVar * n_examples);
  uint64_t allMBAs{}; // total number of formulas considered
  int lastIdx{};      // first free row in the permanent device cache

#ifndef MEASUREMENT_MODE
  printf("Length %-2d | (V)   | AllMBAs: %-11lu | StoredMBAs: %-10d | "
         "ToBeChecked: %-10d \n",
         1, allMBAs, 0, numVar);
#endif

  // Initialise the cache with the raw input variables.
  int index{};
  for (int i = 0; i < numVar; ++i) {
    bool found = true;
    for (int j = 0; j < n_examples; j++) {
      MBACache[index++] = inputData[j * numVar + i];
      if (inputData[j * numVar + i] != outputData[j])
        found = false;
    }
    allMBAs++;
    lastIdx++;
    if (found)
      return "v" + to_string(i);
  }

  // ------------------------------------
  // Memory allocation and initialisation
  // ------------------------------------

  int maxAllocationSize;
  cudaDeviceGetAttribute(&maxAllocationSize, cudaDevAttrMaxPitch, 0);

  const int MBACacheCapacity =
      maxAllocationSize / (n_examples * sizeof(uint32_t));
  const int temp_MBACacheCapacity = MBACacheCapacity / 2;

  // startPoints[depth][opIdx] records the first cache row belonging to
  // (depth, operator).  Zero-initialised; see startPt() for the accessor.
  auto startPoints = std::make_unique<int[]>((maxLen + 2) * numOps);
  startPt(startPoints.get(), 1, numOps - 1) =
      lastIdx;                             // last op slot of depth 1
  startPt(startPoints.get(), 2) = lastIdx; // first free entry at depth 2

  auto FinalMBAIdx = std::make_unique<int>(-1);

  // All device buffers in one RAII object; destructor calls cudaFree for all.
  FormulaCache bufs(MBACacheCapacity, temp_MBACacheCapacity, n_examples);

  const int minusOne = -1;
  checkCuda(cudaMemcpy(bufs.d_FinalMBAIdx, &minusOne, sizeof(int),
                       cudaMemcpyHostToDevice));

  using hash_set_t =
      warpcore::HashSet<uint64_t,        // Key type
                        uint64_t(0) - 1, // Empty key
                        uint64_t(0) - 2, // Tombstone key
                        warpcore::probing_schemes::QuadraticProbing<
                            warpcore::hashers::MurmurHash<uint64_t>>,
                        warpcore::defaults::temp_memory_bytes()>;

  hash_set_t hashSet(2 * MBACacheCapacity);

  checkCuda(cudaMemcpy(bufs.d_MBACache, MBACache.get(),
                       numVar * n_examples * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
  hashSetInit<hash_set_t><<<1, numVar>>>(n_examples, hashSet, bufs.d_MBACache);

  // ----------------------------
  // Enumeration of the next MBAs
  // ----------------------------

  for (int MBALen = 2; MBALen <= maxLen; ++MBALen) {
    bool done = false;
#define TRY_OP(op, label, idx)                                                 \
  if (!done)                                                                   \
    done = generateMBAs<op>(MBALen, startPoints.get(), temp_MBACacheCapacity,  \
                            MBACacheCapacity, allMBAs, lastIdx, n_examples,    \
                            bufs, FinalMBAIdx.get(), hashSet, label, idx);
    SIMBA_OPS(TRY_OP)
#undef TRY_OP
    if (done)
      break;
  }

  if (*FinalMBAIdx != -1)
    return MBAToString(*FinalMBAIdx, startPoints.get(), bufs);
  return "Not found !";
  // DeviceBuffers destructor frees all GPU memory automatically.
}

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
  int maxLen = atoi(argv[2]);

  // Wrap the raw pointers returned by readJsonFile in unique_ptrs so they
  // are automatically freed when main() returns (including early returns).
  auto [inputData_raw, outputData_raw, n_examples, numVarResult] =
      readJsonFile(fileName);
  numVar = numVarResult; // set the global used by toString() and MBA()
  auto inputData = std::unique_ptr<uint32_t[]>(inputData_raw);
  auto outputData = std::unique_ptr<uint32_t[]>(outputData_raw);

  // ------------------------------
  // Mixed Boolean Arithmetic (MBA)
  // ------------------------------

#ifdef MEASUREMENT_MODE
  auto start = chrono::high_resolution_clock::now();
#endif

  string output = MBA(maxLen, n_examples, inputData.get(), outputData.get());
  if (output == "see_the_error")
    return 0;

#ifdef MEASUREMENT_MODE
  auto stop = chrono::high_resolution_clock::now();
#endif

  // -------------------
  // Printing the output
  // -------------------

  printf("Input file : %s\n", fileName.c_str());
  printf("Max length : %d\n", maxLen);
  if (output == "Not found !")
    printf("Result     : no expression found within length %d\n", maxLen);
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