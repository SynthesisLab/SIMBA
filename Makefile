# Compilers
NVCC := nvcc
CXX  := g++-12

# Flags
DEPFLAGS := -MMD -MP
NVCC_FLAGS := -O3 -std=c++17 -arch=sm_90 -Iinclude --extended-lambda -ccbin=$(CXX)
CXX_FLAGS  := -O3 -std=c++17 -Iinclude -Wall -Wextra
HAS_STD_FORMAT := $(shell $(CXX) $(CXXFLAGS) -E -dM -x c++ /dev/null | grep __cpp_lib_format)
ifneq ($(HAS_STD_FORMAT),)
    LIBS = 
else
    LIBS = -lfmt
endif

CPPS := $(wildcard src/*.cpp)
CUS  := $(wildcard src/*.cu)
BINS_SRC := $(wildcard bin/*.cu)
BINS := $(BINS_SRC:bin/%.cu=%)

OBJS := $(CPPS:.cpp=.o) $(CUS:.cu=.o)
DEPS := $(OBJS:.o=.d)


all: $(BINS)

%: bin/%.cu $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

%.o: %.cu
	$(NVCC) $(DEPFLAGS) $(NVCC_FLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(DEPFLAGS) $(CXX_FLAGS) -c $< -o $@

-include $(DEPS)

clean:
	$(RM) -f $(OBJS)
	$(RM) -f $(DEPS)
	$(RM) -rf build/*

mrproper: clean
	$(RM) $(BINS)

.PHONY: all clean mrproper
