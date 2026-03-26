#ifndef SIMBA_UTILS
#define SIMBA_UTILS

#include <cstdint>
#include <fstream>
#include <iostream>
#include <istream>
#include <vector>

using namespace std;

class MbaInstance {
public:
  vector<uint32_t> inputs;
  vector<uint32_t> outputs;
  int n_variables;
  int n_examples;

  friend istream &operator>>(istream &, MbaInstance &);

  static inline MbaInstance from_json_file(const string &filename) {

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