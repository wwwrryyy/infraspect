cmake_minimum_required(VERSION 3.16)
project(battlefield_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 1. Locate pybind11 cmake directory via Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT PYBIND11_CMAKE_DIR)
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

if(PYBIND11_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()

# 2. Add optional Miniforge prefix if present
if(EXISTS "/opt/homebrew/Caskroom/miniforge/base")
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/Caskroom/miniforge/base")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

pybind11_add_module(battlefield_core
    src/pybind_bridge.cpp
    src/video_ingestor.cpp
    src/anomaly_calculus.cpp
)

target_link_libraries(battlefield_core PRIVATE ${OpenCV_LIBS})

# Output shared library directly into src_python for import
set_target_properties(battlefield_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src_python"
)
