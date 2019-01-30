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
* Filename: glorot_weight_initializer.hpp
* Author: Mohammed Boujemaoui
* Date: 30/01/19
*/

#ifndef RNNLITE_GLOROT_WEIGHT_INITIALIZER_HPP
#define RNNLITE_GLOROT_WEIGHT_INITIALIZER_HPP

#include <rnnlite/util/random.hpp>
#include <functional>
#include <algorithm>

namespace rnn { inline namespace weight {

    /**
     * @cite X. Glorot, Y Bengio, Understanding the difficulty of training deep feedforward neural networks
     * Proc. AISTATS 10, May 2010, vol.9, pp249-256
     */
    template <typename T>
    struct glorot_weight_initializer {
        using value_type = T;

        /**
         * @brief Creates a random generator following X. Glorot approach.
         * @param fan_in Number of input weight for each neuron
         * @param fan_out Number of output weight for each neuron
         */
        glorot_weight_initializer(value_type fan_in, value_type fan_out) :
            fan_in_(fan_in),
            fan_out_(fan_out),
            generator_(0, 0) {
            scale(1);
        }

        /**
         * @brief Initialize the given weights.
         * @tparam ForwardIt Forward iterator of the buffer containing the weights.
         * @param first Iterator pointing to the beginning of the buffer.
         * @param last Iterator pointing to the ending of the buffer.
         */
        template <typename ForwardIt>
        void operator()(ForwardIt first, ForwardIt last) const {
            std::generate(first, last, std::ref(generator_));
        }

        /**
         * @brief Updates the scaling value.
         * @param value Scale value.
         */
        void scale(value_type value) {
            const auto base = value / std::sqrt(fan_in_ + fan_out_);
            generator_      = generator{-base, base};
        }

    private:
        using generator = rnn::random_generator<rnn::distribution_type::Uniform, T>;
        value_type fan_in_{1}, fan_out_{1};
        generator generator_;
    };

}} // namespace rnn::weight

#endif //RNNLITE_GLOROT_WEIGHT_INITIALIZER_HPP
