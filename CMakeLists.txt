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
include_directories( "${PROJECT_SOURCE_DIR}" )
include_directories( "${PROJECT_SOURCE_DIR}/rmwarp" )
include_directories( "${PROJECT_SOURCE_DIR}/src" )

find_package(Threads REQUIRED)
find_package(Boost.SIMD REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(FFTW REQUIRED)

set ( CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib" )
set ( CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib" )
set ( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin" )

set ( rmwarp_LIBRARY_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set ( rmwarp_PROGRAM_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin")
set ( rmwarp_HEADERS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include")
set ( rmwarp_PKGCONF_INSTALL_DIR "${rmwarp_LIBRARY_INSTALL_DIR}/pkgconfig")

add_library(rmwarp-lib INTERFACE)
set_property(TARGET rmwarp-lib PROPERTY INTERFACE_LINK_LANGUAGE CXX )
target_compile_options(rmwarp-lib INTERFACE -g -ggdb -O3 -Ofast -march=native )
list(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,-z,relro,-z,now,-z,combreloc -Wl,--build-id ")
target_link_libraries(rmwarp-lib INTERFACE ${FFTW_LIBRARIES})

add_subdirectory(src)
add_subdirectory(test)


install( DIRECTORY rmwarp DESTINATION "${rmwarp_HEADERS_INSTALL_DIR}" )
