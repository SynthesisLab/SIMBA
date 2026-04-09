#include "instance.h"
#include "json.hpp"

using json = nlohmann::json;

// Reads a json-formatted MBA instance for `is` into `mba`.
std::istream &operator>>(std::istream &is, MbaInstance &mba) {
  json j;
  is >> j;

  mba.inputs.clear();
  mba.outputs.clear();

  mba.n_variables = j.at("num_variables").get<uint32_t>();
  mba.n_examples = j.at("num_examples").get<uint32_t>();

  for (const auto &sample : j.at("examples")) {
    uint32_t output_value =
        sample.at("output").at(0).at("value").get<uint32_t>();
    mba.outputs.push_back(output_value);
    for (const auto &input : sample["inputs"]) {
      uint32_t input_value = input.at("value").get<uint32_t>();
      mba.inputs.push_back(input_value);
    }
  }
  return is;
}
