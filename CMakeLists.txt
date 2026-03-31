cmake_minimum_required(VERSION 3.22)

project(plate VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

set(PLATE_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${PLATE_BIN_DIR})

add_library(plate_lib
    src/detector/license_plate_detector.cpp
    src/draw/draw.cpp
    src/filter/npp_filters.cu
    src/image/image_io.cpp
)

target_include_directories(plate_lib
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(plate_lib
    PUBLIC
        CUDA::cudart
        CUDA::nppc
        CUDA::nppicc
        CUDA::nppif
        ${OpenCV_LIBS}
)

target_compile_features(plate_lib PUBLIC cxx_std_17)

set_target_properties(plate_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
)

add_executable(plate src/app/main.cpp)
target_link_libraries(plate PRIVATE plate_lib)

set_target_properties(plate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PLATE_BIN_DIR}
)

if(CMAKE_CONFIGURATION_TYPES)
    foreach(config_name ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${config_name}" config_name_upper)
        set_target_properties(plate PROPERTIES
            "RUNTIME_OUTPUT_DIRECTORY_${config_name_upper}" ${PLATE_BIN_DIR}
        )
    endforeach()
endif()

install(TARGETS plate RUNTIME DESTINATION bin)
