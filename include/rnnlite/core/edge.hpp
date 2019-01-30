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
#ifndef RNNLITE_EDGE_HPP
#define RNNLITE_EDGE_HPP

#include <rnnlite/core/unique_id.hpp>

#include <memory>
#include <unordered_map>

namespace rnn {
    inline namespace core {
        {
            template <class Neuron>
            class edge : public std::enable_shared_from_this<edge<Neuron>> {
            public:
                using value_type = typename Neuron::value_type;
                using edge_ptr   = std::shared_ptr<edge>;
                using neuron_ptr = std::weak_ptr<Neuron>;
                using map        = std::unordered_map<unique_id, edge_ptr>;

                explicit edge(const neuron_ptr& input = nullptr, const neuron_ptr& output = nullptr);

                const unique_id& id() const {
                    return id_;
                }

                const neuron_ptr& input() const {
                    return input_neuron_;
                }

                const neuron_ptr& output() const {
                    return output_neuron_;
                }

                void connect(const neuron_ptr& input, const neuron_ptr& outpur) {}

            private:
                unique_id id_{};
                value_type weight_{0.0};
                value_type gain_{1.0};
                neuron_ptr input_neuron_{nullptr};
                neuron_ptr output_neuron_{nullptr};
            };

            template <class Neuron>
            edge<Neuron>::edge(const edge::neuron_ptr& input, const edge::neuron_ptr& output) :
                input_neuron_(input),
                output_neuron_(output) {}
        }
    }  // namespace core::core
#endif //RNNLITE_EDGE_HPP
