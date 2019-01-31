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
* Filename: testing_edge.cpp
* Author: Mohammed Boujemaoui
* Date: 30/01/19
*/

#include <rnnlite/core/node.hpp>
#include <rnnlite/core/tensor_ref.hpp>
#include <gtest/gtest.h>

TEST(Edge, InitializeWithGivenShape) {
    constexpr auto Rank = 3;
    constexpr auto X    = 42;
    constexpr auto Y    = 60;
    constexpr auto Z    = 42;
    rnn::edge<float, Rank> edge(nullptr, X, Y, Z);
    EXPECT_EQ(Rank, edge.rank());

    const auto& shape = edge.shape();
    EXPECT_EQ(shape.size(), Rank);
    EXPECT_EQ(X, shape[0]);
    EXPECT_EQ(Y, shape[1]);
    EXPECT_EQ(Z, shape[2]);

    const auto size = X * Y * Z;
    EXPECT_EQ(size, edge.size());
}

TEST(Edge, AccessingInternalData) {
    constexpr auto Rank = 3;
    constexpr auto X    = 42;
    constexpr auto Y    = 60;
    constexpr auto Z    = 42;
    rnn::edge<float, Rank> edge(nullptr, X, Y, Z);

    const auto data = edge.data();
    rnn::tensor_ref<float, Rank> reference(data, X, Y, Z);

    for (auto i = 0ul; i < X; ++i) {
        for (auto j = 0ul; j < Y; ++j) {
            for (auto k = 0ul; k < Z; ++k) {
                reference(i, j, k) = 4;
            }
        }
    }

    EXPECT_EQ(std::count(data, data + edge.size(), 4), edge.size());
}