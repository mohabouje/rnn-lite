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
* Filename: softsign.hpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/

#ifndef RNNLITE_SOFTSIGN_HPP
#define RNNLITE_SOFTSIGN_HPP

#include <rnnlite/util/math.hpp>
#include <rnnlite/util/limits.hpp>

namespace rnn { inline namespace activation {

    /**
     * A softsign function is an activation function that converges polynomially, with equation:
     *
     * \f[
     * {\displaystyle f(x)={\frac {x}{1+|x|}}}
     * \f]

     * @tparam T Numeric type.
     */
    template <typename T>
    struct softsign {
        using value_type = T;

        /**
         * @brief Range of the possible output values.
         */
        inline static constexpr auto range = std::make_pair<value_type, value_type>(-1, 1);

        /**
         * @brief Evaluates the softsign function of the input value.
         * @param x Input value.
         * @return Returns the result of applying the softsign function to the input value.
         */
        constexpr value_type operator()(value_type x) const {
            if (x < std::numeric_limits<value_type>::max()) {
                if (x > -std::numeric_limits<value_type>::max()) {
                    return x / (1.0 + std::fabs(x));
                }
                return range.first;
            }
            return range.second;
        }

        /**
         * @brief Computes the nth-derivative.
         * @tparam N Order of the derivative.
         * @param y Input value, obtained from the execution of the softsign function y = f(x)
         * @return Returns the nth-derivative of the softsign function.
         */
        template <std::size_t N>
        constexpr value_type derivative(value_type y) const {
            static_assert(N > 0 && N < 3, "Not implemented yet");
            if constexpr (N == 1) {
                return square(1.0 - std::fabs(y));
            } else if constexpr (N == 2) {
                return -2 * sign(y) * std::pow((1 - std::fabs(y)), 3);
            }
        }
    };

}} // namespace rnn::activation

#endif //RNNLITE_SOFTSIGN_HPP
