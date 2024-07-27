cmake_minimum_required(VERSION 3.22)

project(mission3)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mission3 main.cpp)

target_link_libraries(mission3 ${OpenCV_LIBS})

