cmake_minimum_required(VERSION 3.8)
project(final_project)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES main.cpp)
add_executable(wheres_waldo ${SOURCE_FILES})
target_link_libraries(wheres_waldo ${OpenCV_LIBS})