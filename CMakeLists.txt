cmake_minimum_required(VERSION 3.20)

project(geometrySLAM)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -g")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})

find_package(PCL REQUIRED QUIET)
find_package(Glog REQUIRED QUIET)

include_directories(${PCL_INCLUDE_DIRS})

add_executable(planeDetect main.cpp planeDetect.h)
target_link_libraries(planeDetect
        ${PCL_LIBRARIES}
        ${GLOG_LIBRARIES})
