# 🦁 SIMBA :  SImple MBA expression learning

This repository contains the source code for SIMBA, as well as the code for reproducing the experiments and obtaining the data for the companion paper ("GPU-Accelerated Synthesis of Mixed-Boolean Arithmetic: Beyond Caching", Gabriel Bathie and Nathanaël Fijalkow, NeurIPS 2026).

## TL;DR

Create the executables, the instances and run all the experiments with
```bash
./build_all.sh
./run_all.sh
```

## Building the executables and generating the instances

SIMBA and the VFB algorithm are both written in C++ and Cuda.
Building the executables requires `CUDA` (tested with version `13.2.78`), `g++` (tested with version `Ubuntu 13.3.0-6ubuntu2~24.04.1`) and `GNU Make` (tested with version `4.3`).
The source codes for VFB and SIMBA are available in the respective directories (`vfb/` and `simba/`).
Both Makefiles target compute capability 9.0 by default (`sm_90`); set the `CUDA_ARCH` environment variable to match your GPU, e.g. `CUDA_ARCH=sm_80 ./build_all.sh`.
We use cvc5's Python bindings, through a script given in the `cvc5/` directory.

We use Python scripts to generate MBA expressions and sample them at random values to generate MBA learning instances. These scripts are available in the `scripts/` folder.
The `scripts/` folder also contains `convert_to_xyntia.py`, which converts our JSON instances to the input format of Xyntia.
The Python scripts (both in `scripts/` and `cvc5/`) require Python >= 3.13 and are run through [`uv`](https://docs.astral.sh/uv/), which installs their dependencies automatically.

The `build_all.sh` builds SIMBA, SIMBA-RTid (used in the ablation study) and VFB, by calling their respective Makefiles, and generates the MBA instances used in the paper (reproducibility is ensured by using a fixed random seed). Instances are output as individual JSON files to the `instances/` folder.

## Running the experiments

SIMBA, VFB and our cvc5 script all print their result and runtime in CSV format to the standard output; debug information is sent to `stderr`.
We provide a `run_all.sh` script that runs all four programs (including SIMBA-RTid) on all generated instances, and aggregates them into four files called `res_simba.csv`, `res_simba_rtid.csv`, `res_vfb.csv` and `res_cvc5.csv`.
Each run is given a timeout of 60 seconds, and SIMBA and VFB search for expressions of size at most 20. Runs that time out produce no output, and therefore have no corresponding line in the result files.
This script assumes that `build_all.sh` was run first, and requires a CUDA-compatible GPU to be available.
The `run_all.sh` script can be used to generate the data for the comparison, stability and ablation experiments.

Results for the scalability experiment can be obtained by uncommenting the following line at the top of the `vfb/src/full-cache.cu` file, and then rebuilding VFB (e.g. by running `build_all.sh` again):
```cpp
#define PRINT_CACHE_SIZE
```
In this mode, VFB prints the cache size (as `size,cache_size` lines) instead of its runtime, so the header of `res_vfb.csv` written by `run_all.sh` no longer matches its content.
