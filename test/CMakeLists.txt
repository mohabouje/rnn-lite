cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(rnn-lite-test VERSION 0.0.0 LANGUAGES CXX)

find_package(GTest REQUIRED)
set(SOURCE_FILES
        testing_gtest.cpp
        activation/testing_logistic.cpp
        activation/testing_softsign.cpp
        activation/testing_tanh.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} rnnlite ${GTEST_LIBRARIES} -pthread)

target_include_directories(${PROJECT_NAME} PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_definitions(${PROJECT_NAME} PRIVATE CURRENT_TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME ${PROJECT_NAME}
        COMMAND ${PROJECT_NAME})