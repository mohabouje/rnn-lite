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
* Filename: tanh.hpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/

#ifndef RNNLITE_TANH_HPP
#define RNNLITE_TANH_HPP

#include <rnnlite/util/limits.hpp>
#include <rnnlite/util/math.hpp>

namespace rnn { inline namespace activation {

    /**
     * A tanh (Hyperbolic Tangent Activation Function) function like the logistic sigmoid,
     * is also sigmoidal (“s”-shaped), but instead outputs values that range (-1, 1), with equation:
     *
     * \f[
     * \Large{\begin{array}{rcl} g_{\text{tanh}}(z) &=& \frac{\text{sinh}(z)}{\text{cosh}(z)}
     * \\  &=& \frac{\mathrm{e}^z - \mathrm{e}^{-z}}{\mathrm{e}^z + \mathrm{e}^{-z}}\end{array}}
     * \f]

     * @tparam T Numeric type.
     */
    template <typename T>
    struct tanh {
        using value_type = T;

        /**
         * @brief Range of the possible output values.
         */
        inline static constexpr auto range = std::make_pair<value_type, value_type>(-1, 1);

        /**
         * @brief Evaluates the tanh function of the input value.
         * @param x Input value.
         * @return Returns the result of applying the tanh function to the input value.
         */
        constexpr value_type operator()(value_type x) const {
            if (x < log_traits<value_type>::maximum()) {
                if (x > log_traits<value_type>::minimum()) {
                    const auto negative = std::exp(-x);
                    const auto positive = std::exp(x);
                    return (positive - negative) / (positive + negative);
                }
                return range.first;
            }
            return range.second;
        }

        /**
         * @brief Computes the nth-derivative.
         * @tparam N Order of the derivative.
         * @param y Input value, obtained from the execution of the tanh function y = f(x)
         * @return Returns the nth-derivative of the tanh function.
         */
        template <std::size_t N>
        constexpr value_type derivative(value_type y) const {
            if constexpr (N == 1) {
                return 1 - square(y);
            } else {
                return -2 * y * derivative<1>(y);
            }
        }

    };


}}


#endif //RNNLITE_TANH_HPP
