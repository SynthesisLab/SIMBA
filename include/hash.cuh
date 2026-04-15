#ifndef SIMBA_HASH
#define SIMBA_HASH

#include <cstdint>
#include <warpcore/hash_set.cuh>

// Murmur-inspired avalanche hash – spreads bits so that similar 32-bit values
// produce very different 64-bit contributions to the fingerprint.
inline __device__ uint32_t simple_hash(uint32_t input) {
  input ^= input >> 16;
  input *= 0x85ebca6b;
  input ^= input >> 13;
  input *= 0xc2b2ae35;
  input ^= input >> 16;
  return input;
}

// Fold all n_examples output values of a characteristic sequence (CS) into a
// single 64-bit fingerprint.
inline __device__ uint64_t hash_cs(uint32_t *CS, const int n_examples) {
  uint64_t res = 0;
  for (int i = 0; i < n_examples; ++i) {
    uint64_t val = simple_hash(CS[i]) + 0x9e3779b9 + (res << 6) + (res >> 2);
    // Alternate byte-swap on odd indices for extra diffusion.
    val = (i & 1) ? (val >> 32 | val << 32) : val;
    res ^= val;
  }

  return res;
}

using hash_set_t =
    warpcore::HashSet<uint64_t,        // Key type
                      uint64_t(0) - 1, // Empty key
                      uint64_t(0) - 2, // Tombstone key
                      warpcore::probing_schemes::QuadraticProbing<
                          warpcore::hashers::MurmurHash<uint64_t>>,
                      warpcore::defaults::temp_memory_bytes()>;

template <int MaxNExamples> class GpuHashset {
  hash_set_t hash_set;
  int n_examples;

public:
  GpuHashset(const int n_examples, int capacity)
      : n_examples(n_examples), hash_set(capacity) {}

  __device__ bool insert(uint32_t *CS) {

    uint64_t seed = hash_cs(CS, n_examples);

    const auto group =
        warpcore::cg::tiled_partition<1>(warpcore::cg::this_thread_block());
    return (hash_set.insert(seed, group) > 0) ? false : true;
  }
};

#endif