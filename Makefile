# Compilers
NVCC := nvcc
CXX  := g++

# Flags
DEPFLAGS := -MMD -MP
NVCC_FLAGS := -O2 -std=c++20 -Iincludes --extended-lambda
CXX_FLAGS  := -O2 -std=c++20 -Wall -Wextra

CPPS := $(wildcard src/*.cpp)
CUS := $(wildcard src/*.cu)

OBJS := $(CPPS:.cpp=.o) $(CUS:.cu=.o)
DEPS := $(OBJS:.o=.d)

all: simba_nocache simba_rtid

simba_nocache: src/nocache.o src/instance.o
	$(NVCC) $^ -o $@ $(LIBS)

simba_rtid: src/nocache_rtid.o src/instance.o
	$(NVCC) $^ -o $@ $(LIBS)

%.o: %.cu
	$(NVCC) $(DEPFLAGS) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(DEPFLAGS) $(CXX_FLAGS) -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(OBJS)
	rm -f $(DEPS)

.PHONY: all clean