#ifndef SIMBA_UTILS
#define SIMBA_UTILS
#include <cassert>
#include <cstdio>

inline cudaError_t checkCuda(cudaError_t res) {
#ifndef MEASUREMENT_MODE
  if (res != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
    assert(res == cudaSuccess);
  }
#endif
  return res;
}

#endif // SIMBA_UTILS