#ifndef SIMBA_LIMITS
#define SIMBA_LIMITS

// Plain (CUDA-free) numeric caps, shared between device code (globals.cuh,
// which sizes its __constant__ arrays from these) and host-only code (the
// SyGuS parser) that cannot include CUDA-specific headers.

const int max_n_variables = 10;
const int max_n_examples = 100;
const int max_formula_size = 32;

// Constant leaves (e.g. 0, 1, or SyGuS-grammar-detected shift amounts),
// disjoint from the variable leaves above. See mba_ops.cuh / mba.cuh for the
// token ranges used to tell leaf kinds apart in an RPN formula string.
const int max_n_constants = 16;

// Upper bound on the number of simultaneously-active ops, used by nocache.cu
// to size its runtime (flag-dependent) op-numbering tables.
const int max_n_ops = 16;

#endif // SIMBA_LIMITS
