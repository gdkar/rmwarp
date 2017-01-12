project(rmwarp)

cmake_minimum_required(VERSION 2.8)

set(target_dir "bin")

set(CMAKE_CXX_STANDARD 14)
set_property(GLOBAL PROPERTY CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set( CMAKE_AUTOMOC ON)
set( CMAKE_AUTORCC ON)

set( CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories("." "./include" )

find_package(PkgConfig REQUIRED)
find_package(FFTW REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
