#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include "utils.h"

int main(int argc, char* argv[]) {
    const std::string filename = (argc > 1) ? argv[1] : "benchmarks/test.json";

    auto [inputs, outputs, numExamples, numVar] = readJsonFile(filename);

    if (numExamples == 0) {
        std::cerr << "Failed to read file or no examples found.\n";
        return 1;
    }

    std::cout << "Read " << numExamples << " example(s) from " << filename << "\n";
    for (int i = 0; i < numExamples; ++i) {
        std::cout << "  example " << i
                  << "  input=0x"  << std::hex << std::setw(8) << std::setfill('0') << inputs[i]
                  << "  output=0x" << std::hex << std::setw(8) << std::setfill('0') << outputs[i]
                  << "\n";
    }

    delete[] inputs;
    delete[] outputs;
    return 0;
}
