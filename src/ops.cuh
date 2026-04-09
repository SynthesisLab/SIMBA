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
};
class Neg : public UnaryOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x) { return -x; }
  static constexpr char char_representation = '-';
};

class And : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x & y;
  }
  static constexpr char char_representation = '&';
};
class Or : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x | y;
  }
  static constexpr char char_representation = '|';
};

class Xor : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x ^ y;
  }
  static constexpr char char_representation = '^';
};
class LShift : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x << y;
  }
  static constexpr char char_representation = '<';
};
class RShift : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x >> y;
  }
  static constexpr char char_representation = '>';
};
class Plus : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x + y;
  }
  static constexpr char char_representation = '+';
};
class Minus : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x - y;
  }
  static constexpr char char_representation = '-';
};
class Mul : public BinaryOp, public CommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x * y;
  }
  static constexpr char char_representation = '*';
};
class Div : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x / y;
  }
  static constexpr char char_representation = '/';
};
class Mod : public BinaryOp, public NonCommutativeOp {
public:
  static __device__ constexpr uint32_t evaluate(uint32_t x, uint32_t y) {
    return x % y;
  }
  static constexpr char char_representation = '%';
};

#endif // SIMBA_OPS