/*
* rnnlite, Yet another framework for building deep RNN networks written in modern C++.
*
* The 2-Clause BSD License
*
* Copyright (c) 2019 Mohammed Boujemaoui Boulaghmoudi,
*
* Redistribution and use in source and binary forms, with or without modification,
* are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
* this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation and/or
* other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
* OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
* OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Filename: testing_tensor.cpp
* Author: Mohammed Boujemaoui
* Date: 30/01/19
*/

#include <rnnlite/core/tensor.hpp>
#include <gtest/gtest.h>

TEST(Tensor, TestingEmpty) {
    rnn::tensor1_t<float> example_1;
    EXPECT_EQ(example_1.size(), 0);
    for (auto dim : example_1.dimensions()) {
        EXPECT_EQ(dim, 0);
    }

    rnn::tensor2_t<float> example_2;
    EXPECT_EQ(example_1.size(), 0);
    for (auto dim : example_1.dimensions()) {
        EXPECT_EQ(dim, 0);
    }

    rnn::tensor3_t<float> example_3;
    EXPECT_EQ(example_1.size(), 0);
    for (auto dim : example_1.dimensions()) {
        EXPECT_EQ(dim, 0);
    }

    rnn::tensor4_t<float> example_4;
    EXPECT_EQ(example_1.size(), 0);
    for (auto dim : example_1.dimensions()) {
        EXPECT_EQ(dim, 0);
    }
}

TEST(Tensor, InitializeWithShape) {
    constexpr auto first  = 4;
    constexpr auto second = 10;
    constexpr auto third  = 41;

    rnn::tensor2_t<float> tensor(first, second);
    EXPECT_EQ(tensor.size(), first * second);
    EXPECT_EQ(tensor.dimension(0), first);
    EXPECT_EQ(tensor.dimension(1), second);

    rnn::tensor3_t<float> tensor2(first, second, third);
    EXPECT_EQ(tensor2.size(), first * second * third);
    EXPECT_EQ(tensor2.dimension(0), first);
    EXPECT_EQ(tensor2.dimension(1), second);
    EXPECT_EQ(tensor2.dimension(2), third);
}

TEST(Tensor, RandomAccess) {
    constexpr auto row = 4;
    constexpr auto col = 10;

    rnn::tensor2_t<float> tensor(row, col);
    std::iota(tensor.data(), tensor.data() + tensor.size(), 0);

    for (auto i = 0ul; i < col; ++i) {
        for (auto j = 0ul; j < row; ++j) {
            EXPECT_EQ(tensor(j, i), i * row + j);
        }
    }
}

TEST(Tensor, ArithmeticOperation) {
    constexpr auto row = 4;
    constexpr auto col = 10;

    rnn::tensor2_t<float> tensor(row, col);
    std::iota(tensor.data(), tensor.data() + tensor.size(), 0);

    rnn::tensor2_t<float> tensor2(row, col);
    std::fill(tensor2.data(), tensor2.data() + tensor2.size(), 7);

    rnn::tensor2_t<float> result = tensor + tensor2;

    for (auto i = 0ul; i < row; ++i) {
        for (auto j = 0ul; j < col; ++j) {
            EXPECT_EQ(result(i, j), tensor(i, j) + 7);
        }
    }
}