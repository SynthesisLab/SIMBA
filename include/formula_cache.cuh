#ifndef SIMBA_DEVICE_BUFFERS
#define SIMBA_DEVICE_BUFFERS

#include "utils.cuh"
#include <cstddef>
#include <iterator>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/remove.h>

#include <cstdint>

template <int MaxFormulaLen> struct FormulaCache {
  // Permanent cache – stores all unique MBAs found so far.
  char *d_mba_cache{};
  size_t end_index = 0;
  // Temporary scratch – filled by each kernel batch, then compacted.
  char *d_temp_mba_cache{};
  bool *d_is_unique{};
  // Scalar: temp cache index of the first matching candidate (-1 = not found).
  int *res_index{};

  int n_examples;

  FormulaCache(size_t MBACap, size_t tempCap, size_t n_examples)
      : n_examples(n_examples) {
    checkCuda(cudaMalloc(&d_mba_cache, MBACap * n_examples * sizeof(uint32_t)));
    checkCuda(
        cudaMalloc(&d_temp_mba_cache, tempCap * n_examples * sizeof(uint32_t)));
    checkCuda(cudaMalloc(&d_is_unique, tempCap * n_examples * sizeof(bool)));
    checkCuda(cudaMalloc(&res_index, sizeof(int)));

    int minus_one = -1;
    checkCuda(
        cudaMemcpy(res_index, &minus_one, sizeof(int), cudaMemcpyHostToDevice));
  }

  ~FormulaCache() {
    cudaFree(d_mba_cache);
    cudaFree(d_temp_mba_cache);
    cudaFree(d_is_unique);
    cudaFree(res_index);
  }

  FormulaCache(const FormulaCache &) = delete;
  FormulaCache &operator=(const FormulaCache &) = delete;

  __host__ size_t get_n_formulas() { return end_index; }

  // `index` ranges over [0, n_variables + n_constants). Indices below
  // n_variables build a variable leaf ('0'+index, read from d_inputs);
  // the rest build a constant leaf ('A'+(index-n_variables), read from
  // d_constants). See mba.cuh for the matching evaluation-side decoding.
  __device__ char *build_leaf(size_t index, int n_variables) {
    char *new_formula = d_temp_mba_cache + index * MaxFormulaLen;

    new_formula[0] = (index < static_cast<size_t>(n_variables))
                         ? '0' + index
                         : 'A' + (index - n_variables);
    new_formula[1] = '\0';

    return new_formula;
  }

  template <class Op>
  __device__ char *build_formula(size_t index, char *left_formula,
                                 char *right_formula) {
    char *new_formula = d_temp_mba_cache + index * MaxFormulaLen;
    int idx = 0;

    for (int i = 0; left_formula[i] != '\0'; ++i) {
      new_formula[idx++] = left_formula[i];
    }

    if constexpr (Op::is_binary) {
      for (int i = 0; right_formula[i] != '\0'; ++i) {
        new_formula[idx++] = right_formula[i];
      }
    }

    new_formula[idx++] = Op::unique_char;
    new_formula[idx] = '\0';

    return new_formula;
  }

  __device__ char *get_formula(int index) { return d_mba_cache + index; }

  /// Write whether the formula currently in the temp cache at the given
  /// index should be kept when compressing the cache.
  __device__ void report_unique(size_t index, bool cs_unique) {
    for (int i = 0; i < MaxFormulaLen; ++i) {
      d_is_unique[index * MaxFormulaLen] = cs_unique;
    }
  }

  __device__ void report_solution(size_t index) {
    atomicCAS(res_index, -1, index);
  }

  /// Returns -1 if no solution has been found
  __host__ int get_solution() {
    int sol;
    checkCuda(cudaMemcpy(&sol, res_index, sizeof(int), cudaMemcpyDeviceToHost));
    return sol;
  }

  /// Returns -1 if no solution has been found
  __host__ std::vector<char> get_solution_ptr() {
    std::vector<char> res(MaxFormulaLen, '\0');

    int sol;
    checkCuda(cudaMemcpy(&sol, res_index, sizeof(int), cudaMemcpyDeviceToHost));
    if (sol != -1) {

      checkCuda(cudaMemcpy(res.data(), d_temp_mba_cache + sol,
                           MaxFormulaLen * sizeof(char),
                           cudaMemcpyDeviceToHost));
    }

    return res;
  }

  void store_unique_mba(int temp_cache_size) {
    size_t range = temp_cache_size * MaxFormulaLen;

    thrust::device_ptr<char> cache(d_mba_cache + end_index);
    thrust::device_ptr<char> temp_cache(d_temp_mba_cache);
    thrust::device_ptr<bool> is_unique(d_is_unique);

    thrust::device_ptr<char> cache_end =
        thrust::copy_if(temp_cache, temp_cache + range, is_unique, cache,
                        [] __device__(bool x) { return x; });

    int nb_new_formulas = std::distance(cache, cache_end) / MaxFormulaLen;

    end_index += nb_new_formulas;
  }
};

#endif