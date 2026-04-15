#ifndef SIMBA_OPS
#define SIMBA_OPS

#include <cstdint>

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
  static constexpr char unique_char = '~';
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
  static constexpr char unique_char = '&';
};
class Or : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x | y;
  }
  static constexpr char char_representation = '|';
  static constexpr char unique_char = '|';
};

class Xor : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x ^ y;
  }
  static constexpr char char_representation = '^';
  static constexpr char unique_char = '^';
};
class LShift : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x << y;
  }
  static constexpr char char_representation = '<';
  static constexpr char unique_char = '<';
};
class RShift : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x >> y;
  }
  static constexpr char char_representation = '>';
  static constexpr char unique_char = '>';
};
class Plus : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x + y;
  }
  static constexpr char char_representation = '+';
  static constexpr char unique_char = '+';
};
class Minus : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x - y;
  }
  static constexpr char char_representation = '-';
  static constexpr char unique_char = '-';
};
class Mul : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x * y;
  }
  static constexpr char char_representation = '*';
  static constexpr char unique_char = '*';
};
class Div : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x / y;
  }
  static constexpr char char_representation = '/';
  static constexpr char unique_char = '/';
};
class Mod : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x % y;
  }
  static constexpr char char_representation = '%';
  static constexpr char unique_char = '%';
};

#endif // SIMBA_OPS