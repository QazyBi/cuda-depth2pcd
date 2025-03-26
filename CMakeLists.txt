cmake_minimum_required(VERSION 3.10)
project(MultiNodePointCloud LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Important: Set CUDA flags for proper device linking
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true")

# Create the shared library directly from the source file
add_library(pointcloud SHARED src/pointcloud.cu)
set_target_properties(pointcloud PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(pointcloud ${CUDA_LIBRARIES})

# Create the main executable
add_executable(main src/main.cpp)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(main PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(main pointcloud ${MPI_CXX_LIBRARIES} ${CUDA_LIBRARIES})
