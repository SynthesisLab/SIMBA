#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>

using json = nlohmann::json;

// Returns {inputData, outputData, numExamples, numVariables}.
// inputData is laid out as [example0_var0, example0_var1, ..., example1_var0,
// ...]. Caller owns the two heap-allocated arrays and must delete[] them.
std::tuple<uint32_t *, uint32_t *, int, int>
readJsonFile(const std::string &filename) {

  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return {nullptr, nullptr, 0, 0};
  }

  json j;
  file >> j;

  const int numVar = j["num_variables"];
  const int numExamples = j["num_examples"];

  uint32_t *inputData = new uint32_t[numExamples * numVar];
  uint32_t *outputData = new uint32_t[numExamples];

  // idx tracks the flat position in inputData across all examples × variables.
  int idx = 0;
  for (const auto &example : j["examples"]) {
    const int exampleIdx = idx / numVar;
    outputData[exampleIdx] = std::stoul(
        example["output"][0]["value"].get<std::string>(), nullptr, 16);
    for (const auto &value : example["inputs"]) {
      inputData[idx++] =
          std::stoul(value["value"].get<std::string>(), nullptr, 16);
    }
  }

  return {inputData, outputData, numExamples, numVar};
}
