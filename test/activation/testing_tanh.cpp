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
* Filename: testing_tanh.cpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/

#include <rnnlite/activation/tanh.hpp>
#include <rnnlite/util/derivative.hpp>
#include <gtest/gtest.h>

TEST(TanHFunction, TanHFunction_PositiveNumber_Test) {
    auto function = rnn::activation::tanh<double>{};

    EXPECT_NEAR(function(0.012), 0.01199942403317566633272, 1e-9);
    EXPECT_NEAR(function(0.458), 0.4284527551078611044764, 1e-9);
    EXPECT_NEAR(function(2.0), 0.9640275800758168839464, 1e-9);
    EXPECT_NEAR(function(75.52), 1.0, 1e-9);
}

TEST(TanHFunction, TanHFunction_NegativeNumber_Test) {
    auto function = rnn::activation::tanh<double>{};

    EXPECT_NEAR(function(-0.012), -0.01199942403317566633272, 1e-9);
    EXPECT_NEAR(function(-0.458), -0.4284527551078611044764, 1e-9);
    EXPECT_NEAR(function(-2.0), -0.9640275800758168839464, 1e-9);
    EXPECT_NEAR(function(-75.52), -1.0, 1e-9);
}

TEST(TanHFunction, TanHFunction_FirstDerivative) {
    auto function = rnn::activation::tanh<double>{};

    EXPECT_NEAR(rnn::derivative<1>(function, -0.012), 0.9998560138228720462253, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, -0.458), 0.8164282366404832002479, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, -2.757), 0.01599067798614975229775, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, 0.075), 0.994396026710171960691, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, 0.857), 0.5173797300818266907667, 1e-9);
    EXPECT_NEAR(rnn::derivative<1>(function, 3.857), 0.001784533034861266535389, 1e-9);
}

TEST(TanHFunction, TanHFunction_SecondDerivative) {
    auto function = rnn::activation::tanh<double>{};

    EXPECT_NEAR(rnn::derivative<2>(function, -0.012), 0.02399539256396278415151, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, -0.458), 0.6996018546729356460143, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, -2.757), 0.03172462372309459834778, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, 0.075), -0.1488803579608031302542, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, 0.857), -0.7188564679459184261572, 1e-9);
    EXPECT_NEAR(rnn::derivative<2>(function, 3.857), -0.003565880089563623781903, 1e-9);
}
