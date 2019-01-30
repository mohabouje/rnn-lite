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
* Filename: testing_optimizer.cpp
* Author: Mohammed Boujemaoui
* Date: 30/01/19
*/

#include <rnnlite/optimizer/adagrad_optimizer.hpp>
#include <rnnlite/optimizer/adamax_optimizer.hpp>
#include <rnnlite/optimizer/momentum_optimizer.hpp>
#include <rnnlite/optimizer/nesterov_optimizer.hpp>
#include <rnnlite/optimizer/rmsprop_optimizer.hpp>
#include <rnnlite/optimizer/sgd_optimizer.hpp>
#include <gtest/gtest.h>

TEST(Optimizers, AdagradOptimizer) {
    using array_t = std::vector<double>;
    rnn::optimizer::adagrad_optimizer<array_t> optimizer;

    array_t weights   = {0.20, 0.40, 0.006, -0.77, -0.010};
    array_t gradients = {1.00, -3.24, -0.600, 2.79, 1.820};

    // Defining the expected updates

    array_t first_update  = {0.1900, 0.4100, 0.0160, -0.7800, -0.0200};
    array_t second_update = {0.1829, 0.4170, 0.0230, -0.7870, -0.0270};

    // Testing
    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);
    }

    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
    }
}

TEST(Optimizers, RMSPropOptimizer) {
    using array_t = std::vector<double>;
    rnn::optimizer::rmsprop_optimizer<array_t> optimizer;

    array_t weights   = {-0.021, 1.03, -0.05, -.749, 0.009};
    array_t gradients = {1.000, -3.24, -0.60, 2.79, 1.820};

    // Defining the expected updates

    array_t first_update  = {-0.0220, 1.0310, -0.0490, -0.7500, 0.0080};
    array_t second_update = {-0.0227, 1.0317, -0.0482, -0.7507, 0.0072};

    // Testing
    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);
    }

    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
    }
}

TEST(Optimizers, AdamaxOptimizer) {
    using array_t = std::vector<double>;
    rnn::optimizer::adamax_optimizer<array_t> optimizer;

    array_t weights   = {1.00, 0.081, -0.6201, 0.96, -0.007};
    array_t gradients = {6.45, -3.240, -0.6000, 2.79, 1.820};

    // Defining the expected updates

    array_t first_update  = {0.9980, 0.0830, -0.6181, 0.9580, -0.0090};
    array_t second_update = {0.9960, 0.0850, -0.6161, 0.9560, -0.0109};

    // Testing
    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);
    }

    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
    }
}

TEST(Optimizers, SGDOptimizer) {
    using array_t = std::vector<double>;
    rnn::optimizer::sgd_optimizer<array_t> optimizer;

    array_t weights   = {-0.001, -0.90, 0.005, -0.74, 0.003};
    array_t gradients = {-2.240, -3.24, 0.600, 0.39, 0.820};

    // Defining the expected updates

    array_t first_update  = {0.0214, -0.8676, -0.0010, -0.7439, -0.0052};
    array_t second_update = {0.0438, -0.8352, -0.0070, -0.7478, -0.0134};

    // Testing
    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);
    }

    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
    }
}

TEST(Optimizers, MomentumOptimizer) {
    using array_t = std::vector<double>;
    rnn::optimizer::momentum_optimizer<array_t> optimizer;

    array_t weights   = {-0.001, -0.90, 0.005, -0.74, 0.003};
    array_t gradients = {-2.240, -3.24, 0.600, 0.39, 0.820};

    // Defining the expected updates

    array_t first_update  = {0.0214, -0.8676, -0.0010, -0.7439, -0.0052};
    array_t second_update = {0.0639, -0.8060, -0.0124, -0.7513, -0.0207};

    // Testing
    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);
    }

    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
    }
}

TEST(Optimizers, NesterovOptimizer) {
    using array_t = std::vector<double>;
    rnn::optimizer::nesterov_optimizer<array_t> optimizer;

    array_t weights   = {0.1, 0.30, 0.005, -0.74, -0.008};
    array_t gradients = {1.0, -3.24, -0.600, 2.79, 1.820};

    // Defining the expected updates
    array_t first_update  = {0.0810, 0.3615, 0.0164, -0.7930, -0.0425};
    array_t second_update = {0.0539, 0.4493, 0.0326, -0.8686, -0.0919};

    // Testing
    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(first_update[i], weights[i], 1e-3);
    }

    optimizer(weights, gradients);

    for (size_t i = 0; i < weights.size(); i++) {
        EXPECT_NEAR(second_update[i], weights[i], 1e-3);
    }
}
