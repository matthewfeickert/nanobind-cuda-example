cmake_minimum_required(VERSION 3.26...4.3)

# Build for all major GPU architectures by default so the resulting conda
# package runs on any reasonably modern NVIDIA GPU. Use "native" while
# iterating locally for faster compiles:
#   config.extra-args = ["-C", "cmake.define.CMAKE_CUDA_ARCHITECTURES=native"]
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(gpu-pairwise LANGUAGES CXX CUDA)

# Development.Module (not Development) so we only need Python headers and
# never link against libpython, which is what an extension module wants.
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC src/pairwise.cu)
target_compile_features(_core PRIVATE cxx_std_17 cuda_std_17)

# scikit-build-core sets SKBUILD_PROJECT_NAME; install the module next to the
# pure-Python package so `import gpu_pairwise._core` resolves.
install(TARGETS _core LIBRARY DESTINATION gpu_pairwise)
