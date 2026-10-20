CXX      := c++
CXXFLAGS := -O2 -std=c++20 -Wall -Wextra -Iinclude -Isrc

# CUDA / nvcc settings.
# Set CUDA_ARCH to match your GPU, e.g. `make CUDA_ARCH=sm_80`.
CUDA_ARCH  ?= sm_90
NVCC       := nvcc
NVCCFLAGS  := -O2 -std=c++20 -Iinclude -Isrc --extended-lambda -arch=$(CUDA_ARCH)

TEST_SRC       := src/test_utils.cpp
TEST_BIN       := build/test_utils

FULLCACHE_SRC  := src/full-cache.cu
FULLCACHE_BIN  := build/full-cache

# Instance used by the test targets, e.g. `make test TEST_INSTANCE=../instances/<file>.json`.
TEST_INSTANCE ?=

.PHONY: all test test-full-cache check-test-instance clean

all: $(FULLCACHE_BIN)

$(TEST_BIN): $(TEST_SRC) src/utils.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC)

$(FULLCACHE_BIN): $(FULLCACHE_SRC) src/utils.h
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ $(FULLCACHE_SRC)

check-test-instance:
	@test -n "$(TEST_INSTANCE)" || { echo "Set TEST_INSTANCE to an instance file, e.g. make $(MAKECMDGOALS) TEST_INSTANCE=../instances/<file>.json"; exit 1; }

test: $(TEST_BIN) check-test-instance
	./$(TEST_BIN) $(TEST_INSTANCE)

test-full-cache: $(FULLCACHE_BIN) check-test-instance
	./$(FULLCACHE_BIN) $(TEST_INSTANCE) 5

clean:
	rm -rf build
