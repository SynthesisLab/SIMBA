#ifndef SIMBA_OPS
#define SIMBA_OPS

#include "op_flags.h"
#include <array>
#include <cstdint>
#include <type_traits>

class UnaryOp {
public:
  static constexpr bool is_binary = false;
  static constexpr bool commutes = false;
};
class BinaryOp {
public:
  static constexpr bool is_binary = true;
};

class CommutativeOp {
public:
  static constexpr bool commutes = true;
};
class NonCommutativeOp {
public:
  static constexpr bool commutes = false;
};

class Not : public UnaryOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x) { return ~x; }
  static constexpr char char_representation = '~';
  static constexpr char unique_char = char_representation;
};
class Neg : public UnaryOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x) { return -x; }
  static constexpr char char_representation = '-';
  static constexpr char unique_char = '#';
};

class And : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x & y;
  }
  static constexpr char char_representation = '&';
  static constexpr char unique_char = char_representation;
};
class Or : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x | y;
  }
  static constexpr char char_representation = '|';
  static constexpr char unique_char = char_representation;
};

class Xor : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x ^ y;
  }
  static constexpr char char_representation = '^';
  static constexpr char unique_char = char_representation;
};
class LShift : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x << y;
  }
  static constexpr char char_representation = '<';
  static constexpr char unique_char = char_representation;
};
class RShift : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x >> y;
  }
  static constexpr char char_representation = '>';
  static constexpr char unique_char = char_representation;
};
class Add : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x + y;
  }
  static constexpr char char_representation = '+';
  static constexpr char unique_char = char_representation;
};
class Sub : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x - y;
  }
  static constexpr char char_representation = '-';
  static constexpr char unique_char = char_representation;
};
class Mul : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x * y;
  }
  static constexpr char char_representation = '*';
  static constexpr char unique_char = char_representation;
};
class Div : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x / y;
  }
  static constexpr char char_representation = '/';
  static constexpr char unique_char = char_representation;
};
class Mod : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x % y;
  }
  static constexpr char char_representation = '%';
  static constexpr char unique_char = char_representation;
};

// Arithmetic (sign-extending) shift right: bvashr.
class AShr : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return static_cast<uint32_t>(static_cast<int32_t>(x) >> y);
  }
  static constexpr char char_representation = 'a';
  static constexpr char unique_char = char_representation;
};
// Signed division: bvsdiv.
class SDiv : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return static_cast<uint32_t>(static_cast<int32_t>(x) / static_cast<int32_t>(y));
  }
  static constexpr char char_representation = 'q';
  static constexpr char unique_char = char_representation;
};
// Signed remainder (sign of dividend): bvsrem.
class SRem : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return static_cast<uint32_t>(static_cast<int32_t>(x) % static_cast<int32_t>(y));
  }
  static constexpr char char_representation = 'r';
  static constexpr char unique_char = char_representation;
};

template <class... Ops> class OpGroup {
public:
  static constexpr std::array<char, sizeof...(Ops)> char_representation = {
      {Ops::char_representation}...};

  static constexpr std::array<char, sizeof...(Ops)> unique_char = {
      {Ops::unique_char}...};

  static constexpr std::array<bool, sizeof...(Ops)> commutes = {
      {Ops::commutes}...};

  static constexpr std::array<bool, sizeof...(Ops)> is_binary = {
      {Ops::is_binary}...};

  // Whether Op is one of the ops in this group; used to check group
  // membership when a single Op type is gated by an OpFlags value.
  template <class Op>
  static constexpr bool contains = (std::is_same_v<Op, Ops> || ...);
};

using ReducedOps = OpGroup<Not, Neg, Add, Sub, And, Or, Xor, Mul>;

// Named groups of ops, for use with OpFlags below.
using BitwiseOps = OpGroup<Not, And, Or, Xor>;
using ArithmeticOps = OpGroup<Neg, Add, Sub, Mul>;
using ShiftOps = OpGroup<LShift, RShift, AShr>;
using DivisionOps = OpGroup<Div, Mod, SDiv, SRem>;

// Whether Op should be enumerated given the active OpFlags.
template <class Op> inline bool in_group(const OpFlags &flags) {
  return (flags.bitwise && BitwiseOps::contains<Op>) ||
         (flags.arithmetic && ArithmeticOps::contains<Op>) ||
         (flags.shift && ShiftOps::contains<Op>) ||
         (flags.division && DivisionOps::contains<Op>);
}

#endif // SIMBA_OPS