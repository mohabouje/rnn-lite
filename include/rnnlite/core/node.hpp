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
* Filename: node.hpp
* Author: Mohammed Boujemaoui
* Date: 31/01/19
*/

#ifndef RNNLITE_NODE_HPP
#define RNNLITE_NODE_HPP

#include <rnnlite/core/tensor.hpp>
#include <rnnlite/core/unique_id.hpp>

#include <memory>
#include <unordered_map>

namespace rnn { inline namespace core {

    template <typename T>
    class edge_interface;

    template <typename T>
    class node : std::enable_shared_from_this<node> {
    public:
        using edge_ptr = std::shared_ptr<edge_interface<T>>;

        /**
         * Creates a node with the given configuration.
         * @param fan_in Number of input channels.
         * @param fan_out Number of output channels.
         */
        node(std::size_t fan_in, std::size_t fan_out) :
            input_edges_(fan_in),
            output_edges_(fan_out) {

        }

        /**
         * @brief Returns the number of input channels
         * @return Number of input channels.
         */
        auto fan_in() const {
            return input_edges_.size();
        }

        /**
         * @brief Returns the number of output channels
         * @return Number of output channels.
         */
        auto fan_out() const {
            return output_edges_.size();
        }

        /**
         * @brief Returns an array storing all the input edges.
         * @return Array of smart pointers storing the input edges.
         */
        const auto& inputs() const {
            return input_edges_;
        }

        /**
         * @brief Returns an array storing all the output edges.
         * @return Array of smart pointers storing the output edges.
         */
        const auto& outputs() const {
            return output_edges_;
        }

        /**
         * @brief Returns the shape of the input edge at the given index.
         * @param index Index of the edge.
         * @return Array representing the shape of the edge.
         */
        const std::vector<std::size_t>& input_shape(std::size_t index) const {
            return input_edges_[index]->shape();
        }

        /**
         * @brief Returns the shape of the output edge at the given index.
         * @param index Index of the edge.
         * @return Array representing the shape of the edge.
         */
        const std::vector<std::size_t>& output_shape(std::size_t index) const {
            return output_edges_[index]->shape();
        }

        /**
         * @brief Returns the range of the input values.
         * @return Pair of number representing the range of the input values.
         */
        virtual std::pair<double, double> input_range() const {
            return {0, 1};
        }

        /**
         * @brief Returns the range of the output values.
         * @return Pair of number representing the range of the output values.
         */
        virtual std::pair<double, double> output_range() const {
            return {0, 1};
        }

    protected:
        std::vector<edge_ptr> input_edges_;
        std::vector<edge_ptr> output_edges_;
    };



    template <typename T>
    class edge_interface {
    public:
        using node_ptr = std::shared_ptr<node<T>>;

        /**
         * @brief Creates an edge with the given configuration
         * @param parent Parent node of the edge.
         */
        explicit edge_interface(node_ptr parent) :
                parent_(std::move(parent)) {}

        /**
         * @brief Append a node to the list of childrens.
         * @param other Node to be inserted.
         */
        void append(const node_ptr& other) {
            childs_.push_back(other);
        }

        /**
         * @return Returns an array storing the data of the node.
         */
        virtual const T* data() const = 0;

        /**
         * @return Returns an array storing the data of the node.
         */
        virtual T* data() = 0;

        /**
         * @return Returns an array storing the gradients value of the node.
         */
        virtual const T* gradients() const = 0;

        /**
         * @return Returns an array storing the gradients value of the node.
         */
        virtual T* gradients() = 0;

        /**
         * @return Size of the tensor.
         */
        virtual std::size_t size() const = 0;

        /**
         * @brief Returns the rank of the tensor
         */
        virtual std::size_t rank() const = 0;

        /**
         * @return Returns an array representing the shape of the tensor.
         */
        virtual const std::vector<std::size_t>& shape() const = 0;

        /**
         * @brief Resets the edge to the original state.
         */
        virtual void reset() = 0;

    private:
        node_ptr parent_;
        std::vector<node_ptr> childs_;
    };


    template <typename T, std::size_t Rank>
    class edge : public edge_interface<T> {
    public:
        template <typename... Args>
        explicit edge(const node_ptr& parent, Args... arg) :
                edge_interface(parent),
                data_(std::forward<Args>(arg)...),
                gradient_(std::forward<Args>(arg)...),
                shape_(Rank)
        {
            for (auto i = 0ul; i < Rank; ++i) {
                shape_[i] = static_cast<std::size_t>(data_.dimension(i));
            }
        }

        void reset() override {
            std::fill(data_.data(), data_.data() + data_.size(), static_cast<T>(0));
        }

        auto data() const override {
            return data_.data();
        }

        auto data() override {
            return data_.data();
        }

        auto gradients() override {
            return gradient_.data();
        }

        auto gradients() const override {
            return gradient_.data();
        }

        auto size() const override {
            return static_cast<std::size_t>(gradient_.size());
        }

        auto rank() const override {
            return Rank;
        }

        const std::vector<std::size_t>& shape() const override {
            return shape_;
        }

    private:
        tensor<T, Rank> data_;
        tensor<T, Rank> gradient_;
        std::vector<std::size_t> shape_{};
    };

}}

#endif //RNNLITE_NODE_HPP
