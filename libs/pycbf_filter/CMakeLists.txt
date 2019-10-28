cmake_minimum_required(VERSION 3.5)
project(pycbf_filter)

# Python Interp Options ------------------------------------------------------------------------------------------------
set(PYTHON_EXECUTABLE /home/luwei/anaconda3/bin/python)
set(PYBIND11_PYTHON_VERSION 3.6)

# Compiler Options -----------------------------------------------------------------------------------------------------
SET(CMAKE_CONFIGURATION_TYPES RELEASE; DEBUG)
SET(CMAKE_BUILD_TYPE RELEASE CACHE STRING "Build Type: Release" FORCE)

if (CMAKE_COMPILER_IS_GNUCC)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
            OUTPUT_VARIABLE GCC_VERSION)
    string(REGEX MATCHALL "[0-9]+" GCC_VERSION_COMPONENTS ${GCC_VERSION})
    list(GET GCC_VERSION_COMPONENTS 0 GCC_MAJOR)
    # list(GET GCC_VERSION_COMPONENTS 1 GCC_MINOR)
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
    if (GCC_VERSION VERSION_GREATER 4.7)
        add_definitions(-std=c++11)
    elseif(GCC_VERSION VERSION_GREATER 5.1)
        add_definitions(-std=c++14)
    endif()
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    # extend release-profile with fast-math
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -ffast-math -march=native")
endif()
set(PYBIND11_CPP_STANDARD -std=c++11)

find_package(pybind11 REQUIRED)
pybind11_add_module(pycbf_filter cbf_interface.cpp cbf.h cbf.cpp)

