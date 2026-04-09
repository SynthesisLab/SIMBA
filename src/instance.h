#ifndef SIMBA_INSTANCE
#define SIMBA_INSTANCE

#include <cstdint>
#include <fstream>
#include <iostream>
#include <istream>
#include <vector>

class MbaInstance {
public:
  std::vector<uint32_t> inputs;
  std::vector<uint32_t> outputs;
  int n_variables;
  int n_examples;

  friend std::istream &operator>>(std::istream &, MbaInstance &);

  static inline MbaInstance from_json_file(const std::string &filename) {

    std::ifstream file(filename);
    if (!file) {
      throw "Error: Could not open file " + filename;
    }

    MbaInstance mba;
    file >> mba;
    return mba;
  }
};

#endif