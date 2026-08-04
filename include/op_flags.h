#ifndef SIMBA_OP_FLAGS
#define SIMBA_OP_FLAGS

#include <sstream>
#include <string>

// Runtime toggles for groups of primitives. Each binary picks its own
// defaults so that omitting the CLI override reproduces today's behavior
// exactly; see nocache.cu and light_cache.cu. Deliberately CUDA-free (unlike
// mba_ops.cuh) so host-only code such as the SyGuS parser can use it without
// pulling in __device__ code that only nvcc can compile.
struct OpFlags {
  bool bitwise = true;
  bool arithmetic = true;
  bool shift = false;
  bool division = false;
};

// Parses a comma-separated list of group names (bitwise, arithmetic, shift,
// division) into an OpFlags with only those groups enabled. Used to let a
// binary's default op set be overridden from the command line.
inline OpFlags parse_op_flags(const std::string &csv) {
  OpFlags flags{false, false, false, false};
  std::stringstream ss(csv);
  std::string group;
  while (std::getline(ss, group, ',')) {
    if (group == "bitwise")
      flags.bitwise = true;
    else if (group == "arithmetic")
      flags.arithmetic = true;
    else if (group == "shift")
      flags.shift = true;
    else if (group == "division")
      flags.division = true;
  }
  return flags;
}

#endif // SIMBA_OP_FLAGS
