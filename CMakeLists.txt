cmake_minimum_required(VERSION 3.16)
project(kproto_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(kproto STATIC
  src/distances.cpp
  src/kprototypes.cpp
)
target_include_directories(kproto PUBLIC include)
