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
* Filename: mse.hpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/

#ifndef RNNLITE_MSE_HPP
#define RNNLITE_MSE_HPP

#include <rnnlite/util/math.hpp>

namespace rnn { inline namespace loss {

    template <typename T>
    struct mse {
        using value_type = T;

        /**
         * @brief Computes the Mean Square Error (L2 Loss)
         * @tparam InputIt Input iterator
         * @param first1 Input iterator storing the beginning of the estimated buffer.
         * @param last1 Input iterator storing the ending of the estimated buffer.
         * @param first2 Input iterator storing the beginning of the buffer.
         * @return Results of the cost-function
         */
        template <typename InputIt>
        constexpr value_type operator()(InputIt first1, InputIt last1, InputIt first2) const {
            const auto N      = static_cast<value_type>(std::distance(first1, last1));
            value_type result = 0;
            for (; first1 != last1; ++first1, ++first2) {
                result += square(*first1 - *first2);
            }
            return result / N;
        }

        /**
         * @brief Computes the first derivative of the Mean Square Error (L2 Loss)
         * @tparam InputIt Input iterator
         * @tparam OutputIt Output iterator
         * @param first1 Input iterator storing the beginning of the estimated buffer.
         * @param last1 Input iterator storing the ending of the estimated buffer.
         * @param first2 Input iterator storing the beginning of the buffer.
         * @param d_first Output iterator storing the derivative of the loss function.
         */
        template <typename InputIt, typename OutputIt>
        constexpr void operator()(InputIt first1, InputIt last1, InputIt first2, OutputIt d_first) const {
            const auto factor = 2.0 / static_cast<value_type>(std::distance(first1, last1));
            for (; first1 != last1; ++first1, ++first2, ++d_first) {
                *d_first = factor * (*first1 - *first2);
            }
        }
    };

}} // namespace rnn::loss

#endif //RNNLITE_MSE_HPP
