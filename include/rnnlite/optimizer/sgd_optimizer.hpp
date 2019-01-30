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
* Filename: sgd_optimizer.hpp
* Author: Mohammed Boujemaoui
* Date: 30/01/19
*/

#ifndef RNNLITE_MOMENTUM_HPP
#define RNNLITE_MOMENTUM_HPP

#include <algorithm>

namespace rnn { inline namespace optimizer {
    template <typename Weight>
    struct sgd_optimizer {
        using value_type = typename Weight::value_type;

        explicit sgd_optimizer(value_type learning_rate = 0.01) : learning_rate_(learning_rate) {}

        void reset() {}

        void operator()(Weight& W, const Weight& dW) {
            for (auto i = 0ul, size = W.size(); i < size; ++i) {
                W[i] -= learning_rate_ * (dW[i] + weight_decay_ * W[i]);
            }
        }

    private:
        value_type learning_rate_;
        value_type weight_decay_{0};
    };
}} // namespace rnn::optimizer

#endif //RNNLITE_MOMENTUM_HPP
