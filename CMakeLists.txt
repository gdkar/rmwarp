project(rmwarp)
set(PROJECT_VERSION 0.0.0.0)

cmake_minimum_required(VERSION 3.0)

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

set ( PROJECT_CMAKE_CONFIG          ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake)
set ( PROJECT_CMAKE_CONFIG_VERSION  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake)
set ( PROJECT_CMAKE_FIND            ${PROJECT_BINARY_DIR}/Find${PROJECT_NAME}.cmake)
set ( PROJECT_PKG_CONFIG ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pc)

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

string(REPLACE "." ";" PROJECT_VERSION_COMPONENTS ${PROJECT_VERSION})
list(GET PROJECT_VERSION_COMPONENTS 0 PROJECT_MAJOR_VERSION)
list(GET PROJECT_VERSION_COMPONENTS 1 PROJECT_MINOR_VERSION)
list(GET PROJECT_VERSION_COMPONENTS 2 PROJECT_SUBMINOR_VERSION)
list(GET PROJECT_VERSION_COMPONENTS 3 PROJECT_SUBSUBMINOR_VERSION)

configure_file(cmake/rmwarpConfig.cmake.in ${PROJECT_CMAKE_CONFIG} @ONLY)
configure_file(cmake/rmwarpConfigVersion.cmake.in ${PROJECT_CMAKE_CONFIG_VERSION} @ONLY)
configure_file(cmake/Findrmwarp.cmake.in ${PROJECT_CMAKE_FIND} @ONLY)
configure_file(rmwarp.pc.in ${PROJECT_PKG_CONFIG} @ONLY)
install( FILES ${PROJECT_PKG_CONFIG}
    DESTINATION ${rmwarp_PKGCONF_INSTALL_DIR})
if (WIN32)
    set(CMAKE_CONFIG_INSTALL_DIR ${PROJECT_NAME}/CMake)
else ()
    set(CMAKE_CONFIG_INSTALL_DIR share/cmake/${PROJECT_NAME})
endif()
install(
    FILES ${PROJECT_CMAKE_CONFIG} ${PROJECT_CMAKE_CONFIG_VERSION} ${PROJECT_CMAKE_FIND}
    DESTINATION ${CMAKE_CONFIG_INSTALL_DIR}
)
install( DIRECTORY rmwarp DESTINATION "${rmwarp_HEADERS_INSTALL_DIR}" )
