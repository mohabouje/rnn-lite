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
* Filename: traits.hpp
* Author: Mohammed Boujemaoui
* Date: 01/02/19
*/

#ifndef RNNLITE_TRAITS_HPP
#define RNNLITE_TRAITS_HPP

#include <vector>

namespace rnn { inline namespace core {

    enum class data_type {
        data = 0,
        weight,
        bias
    };

    struct tensor_descriptor {

        template <typename... Args>
        explicit tensor_descriptor(data_type type, Args&&... args) :
                type_(type), shape_(std::forward<Args>(args)...) {}

        /**
         * @return Array describing the shape of the tensor
         */
        const auto& shape() const {
            return shape_;
        }

        /**
         * @return Flag describing the type of information that the tensor stores.
         * @see data_type
         */
        auto type() const {
            return type_;
        }

    private:
        data_type type_{data_type::data};
        std::vector<std::size_t> shape_;
    };

}}


#endif //RNNLITE_TRAITS_HPP
