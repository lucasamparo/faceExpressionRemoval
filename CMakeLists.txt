cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(normalize)

find_package(PCL 1.2 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable (normalize normalize.cpp)
target_link_libraries (normalize ${PCL_LIBRARIES} ${OpenCV_LIBS})
