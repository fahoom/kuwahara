cmake_minimum_required(VERSION 3.25)

project(kuwuhara LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        10.2.1
)

FetchContent_MakeAvailable(fmt)

find_package(CUDAToolkit REQUIRED)

add_executable(kuwuhara "src/main.cu")
target_link_libraries(kuwuhara fmt::fmt CUDA::cudart)

