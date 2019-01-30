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
* Filename: tensor_ref.hpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/
#ifndef RNNLITE_TENSOR_REF_HPP
#define RNNLITE_TENSOR_REF_HPP

#include <rnnlite/core/tensor.hpp>

namespace rnn { inline namespace types {

    template <typename T, std::size_t N>
    class tensor_ref : public Eigen::TensorRef<tensor<T, N>> {};

    template <typename T>
    using tensor_ref1_t = tensor_ref<T, 1>;

    template <typename T>
    using tensor_ref2_t = tensor_ref<T, 2>;

    template <typename T>
    using tensor_ref3_t = tensor_ref<T, 3>;

    template <typename T>
    using tensor_ref4_t = tensor_ref<T, 4>;
}} // namespace rnn::types

#endif //RNNLITE_TENSOR_REF_HPP
