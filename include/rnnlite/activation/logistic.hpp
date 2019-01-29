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
* Filename: logistic.hpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/

#ifndef RNN_LITE_LOGISTIC_HPP
#define RNN_LITE_LOGISTIC_HPP


#include <rnnlite/util/limits.hpp>

namespace rnn { inline namespace activation {

    /**
     * A logistic function or logistic curve is a common "S" shape (sigmoid curve), with equation:
     *
     * \f[
     * {\displaystyle f(x)={\frac {L}{1+e^{-k(x-x_{0})}}}} {\displaystyle f(x)={\frac {L}{1+e^{-k(x-x_{0})}}}}
     * \f]
     *
     * where:
     *
     * - e = the natural logarithm base (also known as Euler's number)
     * - x0 = the x-value of the sigmoid's midpoint,
     * - L = the curve's maximum value
     * - k = the logistic growth rate or steepness of the curve.
     *
     * This class implements a standard logistic function (k = 1, x0 = 0, L = 1) which yields
     *
     * \f[
     * {\displaystyle {\begin{aligned}f(x)&={\frac {1}{1+e^{-x}}}\\&={\frac {e^{x}}{e^{x}+1}}\\&
     *          ={\tfrac {1}{2}}+{\tfrac {1}{2}}\tanh({\tfrac {x}{2}})\\\end{aligned}}}￼
     * \f]
     * @tparam T Numeric type.
     */
    template <typename T>
    struct logistic {
        using value_type = T;

        /**
         * @brief Evaluates the logistic function of the input value.
         * @param x Input value.
         * @return Returns the result of applying the logistic function to the input value.
         */
        constexpr value_type operator()(value_type x) const {
            if (x < log_traits<value_type>::maximum()) {
                if (x > log_traits<value_type>::minimum()) {
                    return 1.0 / (1.0 + std::exp(-x));
                }
                return 0;
            }
            return 1;
        }

        /**
         * @brief Computes the nth-derivative.
         * @tparam N Order of the derivative.
         * @param y Input value, obtained from the execution of the logistic function y = f(x)
         * @return Returns the nth-derivative of the logistic function.
         */
        template <std::size_t N>
        constexpr value_type derivative(value_type y) const {
            static_assert(N > 0 && N < 3, "Not implemented yet");
            if constexpr (N == 1) {
                return y * (1.0 - y);
            } else if constexpr (N == 2) {
                return derivative<1>(y) * (1 - (2 * y));
            }
        }

    };


}}

#endif //RNN_LITE_LOGISTIC_HPP
