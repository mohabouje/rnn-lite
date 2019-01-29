#!/bin/bash

RED='\033[0;31m' # Red
BB='\033[0;34m'  # Blue
NC='\033[0m' # No Color
BG='\033[0;32m' # Green

error() { >&2 echo -e "${RED}$1${NC}"; }
showinfo() { echo -e "${BG}$1${NC}"; }
workingprocess() { echo -e "${BB}$1${NC}"; }
allert () { echo -e "${RED}$1${NC}"; }

showinfo "Installing google benmchmark"
cd ${TRAVIS_BUILD_DIR} && cd ..
rm -rf benchmark
git clone https://github.com/google/benchmark.git
cd benchmark
git clone https://github.com/google/googletest.git
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j6 && sudo make install
cd .. & cd ..
cd rnn-lite

showinfo "Building and installing extensions"
cd ${TRAVIS_BUILD_DIR} && cd extension
mkdir -p build && cd build
cmake .. && make -j8
sudo make install
cd .. && cd ..


showinfo "Building the library..."
cd ${TRAVIS_BUILD_DIR}
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
make -j8
if [ $? -ne 0 ]; then
    error "Error: there are compile errors!"
    exit 3
fi

showinfo "Running the tests..."
make test
if [ $? -ne 0 ]; then
    error "Error: there are some tests that failed!"
    exit 4
fi

showinfo "Success: All tests compile and pass."