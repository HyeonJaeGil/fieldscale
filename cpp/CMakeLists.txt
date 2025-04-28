cmake_minimum_required(VERSION 3.10)
project(FieldscaleDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


find_package(OpenCV REQUIRED)

add_library(Fieldscale STATIC
    Fieldscale.cpp
    Fieldscale.h
)
target_include_directories(Fieldscale PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(fieldscale_demo main.cpp)
target_link_libraries(fieldscale_demo
    Fieldscale
    ${OpenCV_LIBS}
)

