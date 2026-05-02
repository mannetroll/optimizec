# Makefile for dns_cuda (RTX 3090, sm_86)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
CUDA_PATH ?= /usr/local/cuda

NVCC      ?= $(CUDA_PATH)/bin/nvcc
ARCH      ?= sm_86

INCLUDES  := -Icuda -I$(CUDA_PATH)/include
LIBS      := -L$(CUDA_PATH)/lib64 -lcufft -lcudart

NVCCFLAGS := -std=c++14 \
             -O3 \
             --use_fast_math \
             -lineinfo \
             -Xptxas=-v \
             -maxrregcount=255 \
             -arch=$(ARCH)

# -------------------------------------------------------------------
# Sources / objects
# -------------------------------------------------------------------
SOURCES  := cuda/mem_cuda.cu \
            cuda/color_cuda.cu \
            cuda/dns_cuda.cu \
            cuda/pao_km3_cuda.cu \
            cuda/calcom_cuda.cu \
            cuda/om2phys_cuda.cu \
            cuda/step2a_cuda.cu \
            cuda/step2b_cuda.cu \
            cuda/step3_cuda.cu \
            cuda/vfft_cuda.cu

OBJECTS  := $(SOURCES:.cu=.o)
TARGET   := mem_cuda

# -------------------------------------------------------------------
# Rules
# -------------------------------------------------------------------
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
