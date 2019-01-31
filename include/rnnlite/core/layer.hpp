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

    template <typename T>
    class layer_interface : public node<T> {
    public:
        using edge_ptr = typename node<T>::edge_ptr;
        layer_interface(std::size_t fan_in, std::size_t fan_out) : node<T>(fan_in, fan_out) {}

        /**
         * @brief Returns the input-edge at the given index.
         * @param index Index of the input edge.
         * @return Smart pointer holding an input edge.
         */
        virtual edge_ptr& input(std::size_t index) = 0;

        /**
         * @brief Returns the output-edge at the given index.
         * @param index Index of the output edge.
         * @return Smart pointer holding an output edge.
         */
        virtual edge_ptr& output(std::size_t index) = 0;

        /**
         * @brief Returns the input-edge at the given index.
         * @param index Index of the input edge.
         * @return Smart pointer holding an input edge.
         */
        virtual const edge_ptr& input(std::size_t index) const = 0;

        /**
         * @brief Returns the output-edge at the given index.
         * @param index Index of the output edge.
         * @return Smart pointer holding an output edge.
         */
        virtual const edge_ptr& output(std::size_t index) const = 0;

        virtual void set_input_data(const std::vector<rnn::array_view<T>>& data)      = 0;
        virtual void set_output_gradient(const std::vector<rnn::array_view<T>>& data) = 0;
        virtual std::vector<rnn::array_view<T>> output() const                        = 0;

        virtual bool trainable() const          = 0;
        virtual const std::string& type() const = 0;

        virtual void setup(bool initialize_weights) = 0;
        virtual void optimize()                     = 0;
        virtual void reset()                        = 0;
        virtual void initialize_weights()           = 0;

        virtual void backward() = 0;
        virtual void forward()  = 0;
    };

    template <typename T, std::size_t Rank, class MetaInformation, class WeightInitializer, class Optimizer>
    class layer : public layer_interface<T> {
    public:
        using edge_ptr = typename layer_interface<T>::edge_ptr;

        /**
         * @brief Creates a layer with N-input and M-output
         * @param input_information Array storing any meta information about the input edges.
         * @param output_information Array storing any meta information about the output edges.
         */
        layer(const std::vector<MetaInformation>& input_information,
              const std::vector<MetaInformation>& output_information) :
            layer_interface<T>(input_information.size(), output_information.size()),
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
        edge_ptr& input(std::size_t index) override {
            if (!this->input_edges_[index]) {
                allocate_input(index);
            }
            return this->input_edges_[index];
        }

        /**
         * @brief Returns the output-edge at the given index.
         * @param index Index of the output edge.
         * @return Smart pointer holding an output edge.
         */
        edge_ptr& output(std::size_t index) override {
            if (!this->output_edges_[index]) {
                allocate_output(index);
            }
            return this->output_edges_[index];
        }

        /**
         * @brief Returns the input-edge at the given index.
         * @param index Index of the input edge.
         * @return Smart pointer holding an input edge.
         */
        const edge_ptr& input(std::size_t index) const override {
            return this->input_edges_[index];
        }

        /**
         * @brief Returns the output-edge at the given index.
         * @param index Index of the output edge.
         * @return Smart pointer holding an output edge.
         */
        const edge_ptr& output(std::size_t index) const override {
            return this->output_edges_[index];
        }

        /**
         * @brief Updates the optimizer
         * @param optimizer Optimizer to be used.
         */
        void set_optimizer(Optimizer optimizer) {
            optimizer_ = std::move(optimizer);
        }

        /**
         * @brief Updates the weight initializer
         * @param optimizer Initializer to be used.
         */
        void set_weight_initializer(WeightInitializer initializer) {
            weight_initializer_ = std::move(initializer);
        }

        /**
         * @brief Updates the bias initializer
         * @param optimizer Initializer to be used.
         */
        void set_bias_initializer(WeightInitializer initializer) {
            bias_initializer_ = std::move(initializer);
        }

        /**
         * @brief Creates an edge with the input shape at the given channel
         * @param index Index of the input channel
         */
        void allocate_input(std::size_t index) {
            const auto& shape = inputs_information_[index].shape();
            if constexpr (Rank == 1) {
                this->input_edges_[index].reset(new edge<T, Rank>>(nullptr, shape[0]));
            } else if constexpr (Rank == 2) {
                this->input_edges_[index].reset(new edge<T, Rank>>(nullptr, shape[0], shape[1]));
            } else if constexpr (Rank == 3) {
                this->input_edges_[index].reset(new edge<T, Rank>>(nullptr, shape[0], shape[1], shape[2]));
            } else if constexpr (Rank == 4) {
                this->input_edges_[index].reset(new edge<T, Rank>>(nullptr, shape[0], shape[1], shape[2], shape[3]));
            }
        }

        /**
         * @brief Creates an edge with the output shape at the given channel
         * @param index Index of the output channel
         */
        void allocate_output(std::size_t index) {
            const auto& shape = outputs_information_[index].shape();
            if constexpr (Rank == 1) {
                this->output_edges_[index].reset(new edge<T, Rank>>(this, shape[0]));
            } else if constexpr (Rank == 2) {
                this->output_edges_[index].reset(new edge<T, Rank>>(this, shape[0], shape[1]));
            } else if constexpr (Rank == 3) {
                this->output_edges_[index].reset(new edge<T, Rank>>(this, shape[0], shape[1], shape[2]));
            } else if constexpr (Rank == 4) {
                this->output_edges_[index].reset(new edge<T, Rank>>(this, shape[0], shape[1], shape[2], shape[3]));
            }
        }

        void set_input_data(const std::vector<rnn::array_view<T>>& data) override {
            for (auto i = 0ul, j = 0ul; i < this->fan_in(); ++i) {
                if (inputs_information_[i].type()) {
                    auto& reference = data[j++];
                    std::copy(std::cbegin(reference), std::cend(reference), input(i)->data());
                }
            }
        }

        void set_output_gradient(const std::vector<rnn::array_view<T>>& data) override {
            for (auto i = 0ul, j = 0ul; i < this->fan_out(); ++i) {
                if (outputs_information_[i].type()) {
                    auto& reference = data[j++];
                    std::copy(std::cbegin(reference), std::cend(reference), output(i)->gradients());
                }
            }
        }

        std::vector<rnn::array_view<T>> output() const override {
            std::vector<rnn::array_view<T>> temporal;
            temporal.reserve(this->fan_out());

            for (auto i = 0ul; i < this->fan_out(); ++i) {
                if (outputs_information_[i].type()) {
                    const auto& reference = input(i);
                    temporal.emplace_back(reference->data(), reference->size());
                }
            }

            return temporal;
        }

        void forward() override {
            std::vector<rnn::array_view<T>> input_data, output_data;

            input_data.reserve(this->fan_in());
            for (auto& element : this->inputs()) {
                input_data.emplace_back(element->data(), element->size());
                element->reset();
            }

            // TODO: ensure the right size, if need resize everything.

            output_data.reserve(this->fan_out());
            for (auto& element : this->outputs()) {
                output_data.emplace_back(element->data(), element->size());
                element->reset();
            }

            // TODO: call the forward propagation
        }

        void backward() override {
            std::vector<rnn::array_view<T>> input_data, input_gradient, output_data, output_gradient;

            input_data.reserve(this->fan_in());
            input_gradient.reserve(this->fan_in());
            for (auto& element : this->inputs()) {
                input_data.emplace_back(element->data(), element->size());
                input_gradient.emplace_back(element->gradients(), element->size());
            }

            output_data.reserve(this->fan_out());
            output_gradient.reserve(this->fan_out());
            for (auto& element : this->outputs()) {
                output_data.emplace_back(element->data(), element->size());
                output_gradient.emplace_back(element->gradients(), element->size());
            }

            // TODO: call the backward propagation
        }

        /**
         * @brief Reset the information of the different input channels.
         */
        void reset() override {
            for (auto i = 0ul, size = this->fan_in(); i < size; ++i) {
                input(i)->reset();
            }
        }

        void initialize_weights() override {
            if (this->trainable()) {
                for (auto i = 0ul, size = this->fan_in(); i < size; ++i) {
                    auto weights = input(i)->data();
                    weight_initializer_(weights, weights + input(i)->size());
                }
            }
            initialized_ = true;
        }

        void setup(bool init_weights) override {
            for (auto i = 0ul; i < this->fan_out(); ++i) {
                if (!this->output_edges_[i]) {
                    allocate_output(i);
                }
            }

            if (!initialized_ || init_weights) {
                initialize_weights();
            }
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
    };

    template <typename T>
    inline void connect(const std::shared_ptr<layer_interface<T>>& head,
                        const std::shared_ptr<layer_interface<T>>& tail, std::size_t head_index,
                        std::size_t tail_index) {
        auto& head_shape = head->shape(head_index);
        auto& tail_shape = tail->shape(tail_index);

        if (head_shape.size() != tail_shape.size()) {
            throw std::runtime_error("Whatever");
        }

        auto head_edge = head->output(head_index);
        if (!head_edge) {
            throw std::runtime_error("Another error");
        }

        head_edge->connect(tail);
        tail->input(tail_index) = head_edge;
    }

}} // namespace rnn::core

#endif //RNNLITE_LAYER_HPP
