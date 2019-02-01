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
* Filename: connect.hpp
* Author: Mohammed Boujemaoui
* Date: 01/02/19
*/

#ifndef RNNLITE_CONNECT_HPP
#define RNNLITE_CONNECT_HPP

#include <rnnlite/core/layer.hpp>

namespace rnn { inline namespace core {

    template <typename T>
    inline void connect(const layer_interface<T>* head,
                        const layer_interface<T>* tail,
                        std::size_t head_index,
                        std::size_t tail_index) {
        const auto& head_shape = head->output_shape(head_index);
        const auto& tail_shape = tail->input_shape(tail_index);

        if (head_shape.size() != tail_shape.size()) {
            throw std::runtime_error("Whatever");
        }

        const auto& head_edge = head->output(head_index);
        if (!head_edge) {
            throw std::runtime_error("Another error");
        }

        head_edge->connect(tail);
        tail->input(tail_index) = head_edge;
    }

    template <class Layer1, class Layer2>
    inline void connect(const std::shared_ptr<Layer1>& head,
                        const std::shared_ptr<Layer2>& tail,
                        std::size_t head_index,
                        std::size_t tail_index) {
        connect(head.get(), tail.get(), head_index, tail_index);
    };

}}

#endif //RNNLITE_CONNECT_HPP
