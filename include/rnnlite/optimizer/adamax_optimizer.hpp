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
* Filename: adamax_optimizer.hpp
* Author: Mohammed Boujemaoui
* Date: 30/01/19
*/

#ifndef RNNLITE_ADAMAX_OPTIMIZER_HPP
#define RNNLITE_ADAMAX_OPTIMIZER_HPP

#include <rnnlite/optimizer/optimizer_cache.hpp>
#include <algorithm>


namespace rnn { inline namespace optimizer {


    template <typename Weigth>
    struct adamax_optimizer {
        using value_type = typename Weigth::value_type;

        explicit adamax_optimizer(value_type learning_rate = 0.002) :
                learning_rate_(learning_rate)
        {

        }

        void operator()(Weigth& W, const Weigth& dW) const {
            constexpr auto epsilon = 1e-8;
            auto &mt = cache_.get<0>(W);
            auto &ut = cache_.get<1>(W);

            for (auto i = 0ul, size = W.size(); i < size; ++i) {
                mt[i] = b1_ * mt[i] + (1.0 - b1_) * dW[i];
                ut[i] = std::max(b2_ * ut[i], std::abs(dW[i]));

                // Lp norm based update rule
                W[i] -= (learning_rate_ / (1.0 - b1_t_)) * (mt[i] / (ut[i] + epsilon));
            }

            b1_t_ *= b1_;
        }

    private:
        value_type learning_rate_;
        value_type b1_{0.9};
        value_type b2_{0.999};
        value_type b1_t_{0.9};
        mutable optimizer_cache<2, Weigth, Weigth> cache_;
    };

}}


#endif //RNNLITE_ADAMAX_OPTIMIZER_HPP
