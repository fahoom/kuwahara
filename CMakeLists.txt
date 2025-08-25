cmake_minimum_required (VERSION 3.25)

project (kuwuhara LANGUAGES CXX CUDA)

add_executable (kuwuhara src/kernel.cu)
