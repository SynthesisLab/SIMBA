#ifndef SIMBA_SYGUS_PARSER
#define SIMBA_SYGUS_PARSER

#include "instance.h"
#include "op_flags.h"
#include <cstdint>
#include <string>
#include <vector>

struct SygusParseResult {
  MbaInstance instance;
  OpFlags flags;
  std::vector<uint32_t> constants;
};

// Parses a SyGuS-BV PBE-style .sl file (the SyGuS-Comp PBE_BV_Track format:
// a synth-fun with a grammar built from bvXXX primitives and define-fun
// macros, plus a set of (constraint (= (f <literal...>) <literal>))
// examples).
//
// Detects which primitive groups the grammar uses (mapped to an OpFlags),
// collects the constant literals it references (deduplicated, capped at
// max_n_constants - see simba_limits.h), and extracts the literal examples
// into an MbaInstance.
//
// (BitVec 8) values are zero-extended (exact). (BitVec 64) values are
// truncated to their low 32 bits, since SIMBA's engine is uint32_t-only;
// this is exact for bitwise/add/sub/mul but not guaranteed for shift/div/rem,
// and a warning is printed to stderr when it happens.
//
// Throws std::string on malformed or unsupported input: no literal examples
// found (e.g. a specification-style file with no concrete constraints),
// more parameters than max_n_variables, or an unsupported bit width.
SygusParseResult parse_sygus_bv_file(const std::string &path);

#endif // SIMBA_SYGUS_PARSER
