cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dns_cuda LANGUAGES C CXX CUDA)

# ================================================================
# CUDA configuration
# ================================================================
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target NVIDIA RTX 3090 (SM86)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Maximize optimization for compute work
set(CMAKE_BUILD_TYPE Release)

# ================================================================
# Source files
# ================================================================
set(SOURCES
        cuda/calcom_cuda.cu
        cuda/color_cuda.cu
        cuda/dns_cuda.cu
        cuda/mem_cuda.cu
        cuda/om2phys_cuda.cu
        cuda/pao_km3_cuda.cu
        cuda/step2a_cuda.cu
        cuda/step2b_cuda.cu
        cuda/step3_cuda.cu
        cuda/vfft_cuda.cu
)

# ================================================================
# Executable target
# ================================================================
add_executable(dns_cuda ${SOURCES})

# ================================================================
# Include CUDA source directory (cuda_dns.h)
# ================================================================
target_include_directories(dns_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

# ================================================================
# CUDA compilation flags
# ================================================================
target_compile_options(dns_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        -lineinfo
        -Xptxas=-v
        -maxrregcount=255
        >
)

# ================================================================
# Link CUDA libraries
# ================================================================
find_package(CUDAToolkit REQUIRED)

target_link_libraries(dns_cuda PRIVATE
        CUDA::cufft
        CUDA::cudart
)

# Win10 Pro
# mkdir build
# cd build
# cmake .. -G "Visual Studio 17 2022" -A x64
# cmake --build . --config Release

# WSL
# mkdir build
# cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# cmake --build . -j$(nproc)
