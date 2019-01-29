/*
* rnn-lite, Yet another framework for building deep RNN networks written in modern C++.
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
* Filename: testing_logistic.cpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/

#include <rnnlite/activation/logistic.hpp>
#include <rnnlite/util/derivative.hpp>
#include <gtest/gtest.h>

TEST(LogisticFunction, LogisticFunction_PositiveNumber_Test) {
    auto function = rnn::activation::logistic<double>{};

    EXPECT_NEAR(function(0.012), 0.5029999640005184, 1e-9);
    EXPECT_NEAR(function(0.458), 0.6125396134409151, 1e-9);
    EXPECT_NEAR(function(2.0), 0.8807970779778823, 1e-9);
    EXPECT_NEAR(function(75.52), 1.0, 1e-9);
}

TEST(LogisticFunction, LogisticFunction_NegativeNumber_Test) {
    auto function = rnn::activation::logistic<double>{};

    EXPECT_NEAR(function(-0.012), 0.49700003599948167, 1e-9);
    EXPECT_NEAR(function(-0.458), 0.3874603865590849, 1e-9);
    EXPECT_NEAR(function(-2.0), 0.11920292202211755, 1e-9);
    EXPECT_NEAR(function(-75.52), 1.5925047143473614e-33, 1e-9);
}

TEST(LogisticFunction, LogisticFunction_FirstDerivative) {
    auto function = rnn::activation::logistic<double>{};

    EXPECT_NEAR(rnn::derivative<1>(function, -0.012), 0.24999100021599557, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, -0.458), 0.23733483540656938, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, -2.757), 0.056129333373867094, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, 0.075), 0.24964876682739415, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, 0.857), 0.2091824699158776, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, 3.857), 0.02026576524796812, 1e-9);
}

TEST(LogisticFunction, LogisticFunction_SecondDerivative) {
    auto function = rnn::activation::logistic<double>{};

    EXPECT_NEAR(rnn::derivative<2>(function, -0.012), 0.001499928002203113, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, -0.458), 0.053419141265437065, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, -2.757), 0.04942832934856418, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, 0.075), -0.009357442865844565, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, 0.857), -0.08452372785509665, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, 3.857), -0.01942700546482088, 1e-9);
}