CXX      := c++
CXXFLAGS := -std=c++20 -Wall -Wextra -Iinclude -Isrc

# CUDA / nvcc settings.
# Adjust WARPCORE_DIR to point to your warpcore installation.
NVCC       := nvcc
NVCCFLAGS  := -std=c++20 -Iinclude -Isrc --extended-lambda -arch=sm_90
WARPCORE_DIR ?= $(HOME)/warpcore
NVCCFLAGS  += -I$(WARPCORE_DIR)/include

JSON_URL := https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
JSON_HDR := include/nlohmann/json.hpp

TEST_SRC       := src/test_utils.cpp
TEST_BIN       := build/test_utils

FULLCACHE_SRC  := src/full-cache.cu
FULLCACHE_BIN  := build/full-cache

.PHONY: all test test-full-cache clean

all: $(TEST_BIN) $(FULLCACHE_BIN)

# Download nlohmann/json single-header if not present
$(JSON_HDR):
	@mkdir -p $(dir $@)
	@echo "Downloading nlohmann/json → $@"
	curl -fsSL $(JSON_URL) -o $@

$(TEST_BIN): $(TEST_SRC) src/utils.h $(JSON_HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC)

$(FULLCACHE_BIN): $(FULLCACHE_SRC) src/utils.h $(JSON_HDR)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o $@ $(FULLCACHE_SRC)

test: $(TEST_BIN)
	./$(TEST_BIN) benchmarks/test.json

test-full-cache: $(FULLCACHE_BIN)
	./$(FULLCACHE_BIN) benchmarks/test.json 5

clean:
	rm -rf build
