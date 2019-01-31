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
* Filename: layer.hpp
* Author: Mohammed Boujemaoui
* Date: 31/01/19
*/

#ifndef RNNLITE_LAYER_HPP
#define RNNLITE_LAYER_HPP

#include <rnnlite/core/node.hpp>
#include <rnnlite/core/tensor_ref.hpp>
#include <rnnlite/core/matrix.hpp>


namespace rnn { inline namespace core {


    class layer_interface : public node {
    public:
        layer_interface(std::size_t fan_in, std::size_t fan_out) :
                node(fan_in, fan_out) {}

        virtual bool trainable() const = 0;
        virtual const std::string& type() const = 0;

        virtual void setup() = 0;
        virtual void optimize() = 0;
        virtual void reset() = 0;
        virtual void initialize_weights() = 0;
    };

    template <typename T,
            std::size_t Rank,
            class MetaInformation,
            class WeightInitializer,
            class Optimizer>
    class layer : public layer_interface {
    public:

        /**
         * @brief Creates a layer with N-input and M-output
         * @param input_information Array storing any meta information about the input edges.
         * @param output_information Array storing any meta information about the output edges.
         */
        layer(const std::vector<MetaInformation>& input_information,
              const std::vector<MetaInformation>& output_information) :
            layer_interface(input_information.size(), output_information.size()),
            weight_initializer_(input_information.size(), output_information.size()),
            bias_initializer_(input_information.size(), output_information.size()),
            inputs_information_(input_information),
            outputs_information_(output_information) {
            // TODO: maybe initialize all the inputs + outputs.
        }

        /**
         * @brief Returns the input-edge at the given index.
         * @param index Index of the input edge.
         * @return Smart pointer holding an input edge.
         */
        auto& input(std::size_t index) {
            if (!input_edges_[index]) {
                allocate_input(index);
            }
            return input_edges_[index];
        }

        /**
         * @brief Returns the output-edge at the given index.
         * @param index Index of the output edge.
         * @return Smart pointer holding an output edge.
         */
        auto& output(std::size_t index) {
            if (!output_edges_[index]) {
            }
            return output_edges_[index];
        }

        /**
         * @brief Returns the input-edge at the given index.
         * @param index Index of the input edge.
         * @return Smart pointer holding an input edge.
         */
        const auto& input(std::size_t index) const {
            return input_edges_[index];
        }

        /**
         * @brief Returns the output-edge at the given index.
         * @param index Index of the output edge.
         * @return Smart pointer holding an output edge.
         */
        const auto& output(std::size_t index) const {
            return output_edges_[index];
        }


        /**
         * @brief Updates the optimizer
         * @param optimizer Optimizer to be used.
         */
        void set_optimizer(Optimizer optimizer) {
            optimizer_ = std::move(optimizer);
        }

        /**
         * @brief Creates an edge with the input shape at the given channel
         * @param index Index of the input channel
         */
        void allocate_input(std::size_t index) {
            const auto& shape = inputs_information_[index].shape();
            if constexpr (Rank == 1) {
                input_edges_[index] = std::make_shared<edge<T, Rank>>(nullptr, shape[0]);
            } else if constexpr (Rank == 2) {
                input_edges_[index] = std::make_shared<edge<T, Rank>>(nullptr, shape[0], shape[1]);
            } else if constexpr (Rank == 3) {
                input_edges_[index] = std::make_shared<edge<T, Rank>>(nullptr, shape[0], shape[1], shape[2]);
            } else if constexpr (Rank == 4) {
                input_edges_[index] = std::make_shared<edge<T, Rank>>(nullptr, shape[0], shape[1], shape[2], shape[3]);
            }
        }

        /**
         * @brief Creates an edge with the output shape at the given channel
         * @param index Index of the output channel
         */
        void allocate_output(std::size_t index) {
            const auto& shape = outputs_information_[index].shape();
            if constexpr (Rank == 1) {
                output_edges_[index] = std::make_shared<edge<T, Rank>>(dynamic_cast<node*>(this), shape[0]);
            } else if constexpr (Rank == 2) {
                output_edges_[index] = std::make_shared<edge<T, Rank>>(dynamic_cast<node*>(this), shape[0], shape[1]);
            } else if constexpr (Rank == 3) {
                output_edges_[index] = std::make_shared<edge<T, Rank>>(dynamic_cast<node*>(this), shape[0], shape[1], shape[2]);
            } else if constexpr (Rank == 4) {
                output_edges_[index] = std::make_shared<edge<T, Rank>>(dynamic_cast<node*>(this), shape[0], shape[1], shape[2], shape[3]);
            }
        }

        /**
         * @brief Reset the information of the different input channels.
         */
        void reset() override {
            for (auto i = 0ul, size = fan_in(); i < size; ++i) {
                input(i)->reset();
            }
        }

        void initialize_weights() override {
            if (trainable()) {
                for (auto i = 0ul, size = fan_in(); i < size; ++i) {
                    auto weights = input(i)->data();
                    weight_initializer_(weights, weights + input(i)->size());
                }
            }
            initialized_ = true;
        }

        void setup() override {
            // TODO: implement this part
        }

        void optimize() override {
            // TODO: implement this part
        }

    private:
        bool initialized_{false};
        Optimizer optimizer_{};
        WeightInitializer weight_initializer_{};
        WeightInitializer bias_initializer_{};
        std::vector<MetaInformation> inputs_information_;
        std::vector<MetaInformation> outputs_information_;
        unique_id weights_{0}; // TODO: try to put here the right thing!
    };

}}

#endif //RNNLITE_LAYER_HPP
