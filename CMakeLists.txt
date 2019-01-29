cmake_minimum_required(VERSION 3.4)
project(rnnlite CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIB_NAME ${PROJECT_NAME})
set(VERSION_MAJOR  0)
set(VERSION_MINOR  0)
set(VERSION_PATCH  1)
set(LIB_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

option(ENABLE_COVERAGE "Enable the code coverage" ON)

if (ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -fprofile-arcs -coverage")
endif(ENABLE_COVERAGE)

add_library(${LIB_NAME} INTERFACE)
target_sources(${LIB_NAME} INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_include_directories(${LIB_NAME} INTERFACE)
target_include_directories(${LIB_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)
target_include_directories(${LIB_NAME} SYSTEM INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)

file(GLOB_RECURSE FPHUB_PUBLIC_HEADERS
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}-config-version.cmake"
        VERSION ${LIB_VERSION}
        COMPATIBILITY ExactVersion
)

install(TARGETS ${LIB_NAME} EXPORT "${LIB_NAME}-targets" DESTINATION lib)
install(EXPORT "${LIB_NAME}-targets" FILE "${LIB_NAME}-config.cmake" DESTINATION "lib/cmake/${LIB_NAME}")
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME}-config-version.cmake"
        DESTINATION "lib/cmake/${LIB_NAME}")
install(DIRECTORY include/ DESTINATION include)


enable_testing()
add_subdirectory(test)