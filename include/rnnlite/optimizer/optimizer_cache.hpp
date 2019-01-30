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
* Filename: optimizer_cache.hpp
* Author: Mohammed Boujemaoui
* Date: 30/01/19
*/

#ifndef RNNLITE_OPTIMIZER_CACHE_HPP
#define RNNLITE_OPTIMIZER_CACHE_HPP

#include <array>
#include <vector>
#include <unordered_map>

namespace rnn { inline namespace optimizer {

    template <std::size_t N, class Key, class Value>
    struct optimizer_cache {
        template <std::size_t M>
        Value& get(const Key& key) {
            //static_assert(M > N, "Out of bounds");
            auto& map = cache_[M];

            auto existing = map.find(&key);
            if (existing != map.end()) {
                existing->second.resize(key.size());
                return existing->second;
            }

            map[&key] = Key(key.size(), 0);
            return map[&key];
        }

        void reset() {
            for (auto& element : cache_) {
                element.clear();
            }
        }

    private:
        using cache_type = std::unordered_map<const Key*, Value>;
        std::array<cache_type, N> cache_;
    };
}} // namespace rnn::optimizer

#endif //RNNLITE_OPTIMIZER_CACHE_HPP
