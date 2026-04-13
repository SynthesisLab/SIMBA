#ifndef SIMBA_DEVICE_BUFFERS
#define SIMBA_DEVICE_BUFFERS

#include "globals.cuh"
#include "utils.cuh"

// ---------------------------------------------------------------------------
// Device buffer management
// ---------------------------------------------------------------------------

// All GPU allocations bundled in one struct.
// Constructor allocates; destructor frees.
struct FormulaCache {
  // Permanent cache – stores all unique candidate solutions found so far.
  uint32_t *d_MBACache{};
  int *d_leftIdx{};
  int *d_rightIdx{};
  // Temporary scratch – filled by each kernel batch, then compacted.
  uint32_t *d_temp_MBACache{};
  bool *d_temp_boolCache{};
  int *d_temp_leftIdx{};
  int *d_temp_rightIdx{};
  // Scalar: cache index of the first matching candidate (-1 = not found).
  int *d_FinalMBAIdx{};

  FormulaCache(size_t MBACap, size_t tempCap, size_t n_examples) {
    checkCuda(cudaMalloc(&d_MBACache, MBACap * n_examples * sizeof(uint32_t)));
    checkCuda(
        cudaMalloc(&d_temp_MBACache, tempCap * n_examples * sizeof(uint32_t)));
    checkCuda(
        cudaMalloc(&d_temp_boolCache, tempCap * n_examples * sizeof(bool)));
    checkCuda(cudaMalloc(&d_leftIdx, MBACap * sizeof(int)));
    checkCuda(cudaMalloc(&d_rightIdx, MBACap * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_leftIdx, tempCap * sizeof(int)));
    checkCuda(cudaMalloc(&d_temp_rightIdx, tempCap * sizeof(int)));
    checkCuda(cudaMalloc(&d_FinalMBAIdx, sizeof(int)));
  }

  ~FormulaCache() {
    cudaFree(d_MBACache);
    cudaFree(d_temp_MBACache);
    cudaFree(d_temp_boolCache);
    cudaFree(d_leftIdx);
    cudaFree(d_rightIdx);
    cudaFree(d_temp_leftIdx);
    cudaFree(d_temp_rightIdx);
    cudaFree(d_FinalMBAIdx);
  }

  FormulaCache(const FormulaCache &) = delete;
  FormulaCache &operator=(const FormulaCache &) = delete;

  // Write a candidate solution into the temporary cache.
  // If unique: store the output vector, mark it as valid, record its
  // operand indices, and signal a match if it equals the target.
  // If duplicate: mark the slot invalid so storeUniqueMBAs() can filter it
  // out.
  inline __device__ void insertInCache(bool CS_is_unique, uint32_t *CS, int tid,
                                       int ldx, int rdx, const int n_examples) {
    if (CS_is_unique) {
      for (int i = 0; i < n_examples; ++i) {
        d_temp_MBACache[tid * n_examples + i] = CS[i];
        d_temp_boolCache[tid * n_examples + i] = true;
      }
      d_temp_leftIdx[tid] = ldx;
      d_temp_rightIdx[tid] = rdx;

      // Check whether this CS matches the target output vector.
      bool found = true;
      for (int i = 0; found && i < n_examples; ++i)
        found = (CS[i] == d_outputData[i]);
      // atomicCAS: only the first thread that finds a match writes its tid.
      if (found)
        atomicCAS(d_FinalMBAIdx, -1, tid);

    } else {
      // Mark slot invalid; the bool mask drives thrust::remove_if later.
      for (int i = 0; i < n_examples; ++i)
        d_temp_boolCache[tid * n_examples + i] = false;
      d_temp_leftIdx[tid] = -1;
      d_temp_rightIdx[tid] = -1;
    }
  }
};

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

#endif