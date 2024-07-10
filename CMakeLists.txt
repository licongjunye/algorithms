cmake_minimum_required(VERSION 3.0)
project(kalmanfilter)
set(CMAKE_BUILD_TYPE Debug)
find_package(OpenCV REQUIRED )


# Eigen
include_directories("/usr/include/eigen3")


# add_executable(kalmanfilter kalmanfilter.cpp)
add_executable(gaussnewton gaussnewton.cpp)

target_link_libraries(gaussnewton ${OpenCV_LIBS})