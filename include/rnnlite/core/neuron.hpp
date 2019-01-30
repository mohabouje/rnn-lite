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
* Filename: edge.hpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/
#ifndef RNNLITE_NEURON_HPP
#define RNNLITE_NEURON_HPP

#include <rnnlite/core/edge.hpp>

namespace rnn { inline namespace core {

    template <typename ActivationFunction>
    class neuron : public std::enable_shared_from_this<neuron<ActivationFunction>> {
    public:
        using base_class = neuron<ActivationFunction>;
        using value_type = typename ActivationFunction::value_type;
        using neuron_ptr = std::shared_ptr<base_class>;
        using map        = std::unordered_map<unique_id, neuron_ptr>;

        template <typename... Args>
        explicit neuron(value_type bias, Args... arg) {}

    private:
        unique_id id_;
        value_type bias_;
        value_type activation_;
        value_type derivative_;
        value_type state_;
        value_type previous_state_;
        ActivationFunction function_;
    };

}}     // namespace rnn::core
#endif //RNNLITE_NEURON_HPP
