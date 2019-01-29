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
* Filename: identity.hpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/

#ifndef RNNLITE_IDENTITY_HPP
#define RNNLITE_IDENTITY_HPP

#include <cstddef>
#include <limits>
#include <utility>

namespace rnn { inline namespace activation {

        /**
         * A identity function is the simplest activation function, with equation:
         *
         * \f[
         * {\displaystyle f(x) = x}
         * \f]

         * @tparam T Numeric type.
         */
        template <typename T>
        struct identity {
            using value_type = T;

            /**
             * @brief Range of the possible output values.
             */
            inline static constexpr auto range = std::make_pair<value_type, value_type>(
                    std::numeric_limits<value_type>::infinity(),
                    -std::numeric_limits<value_type>::infinity());

            /**
             * @brief Evaluates the identity function of the input value.
             * @param x Input value.
             * @return Returns the result of applying the identity function to the input value.
             */
            constexpr value_type operator()(value_type x) const {
                return x;
            }

            /**
             * @brief Computes the nth-derivative.
             * @tparam N Order of the derivative.
             * @param y Input value, obtained from the execution of the identity function y = f(x)
             * @return Returns the nth-derivative of the identity function.
             */
            template <std::size_t N>
            constexpr value_type derivative(value_type y) const {
                if constexpr (N == 1) {
                    return 1;
                } else {
                    return 0;
                }
            }

        };


    }}

#endif //RNNLITE_IDENTITY_HPP
