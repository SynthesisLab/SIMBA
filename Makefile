# Compilers
NVCC := nvcc
CXX  := g++

# Flags
NVCC_FLAGS := -O3 -std=c++17
CXX_FLAGS  := -O3 -std=c++17
LIBS       := -lfmt

# Target names
TARGET := simba
OBJS   := src/nocache.o src/utils.o

all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET) $(LIBS)

# Compile CUDA source
src/nocache.o: src/nocache.cu src/globals.cuh src/mba.cuh src/utils.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source (This bypasses the std::function error)
src/utils.o: src/utils.cpp src/utils.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean