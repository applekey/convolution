cmake_minimum_required(VERSION 2.8.9)
project (wave)
find_package(CUDA QUIET REQUIRED)

file(GLOB wave_SRC
    "*.h"
    "*.cpp"
    "*.cuh"
    "*.cu"
)

cuda_add_executable(wave ${wave_SRC})
