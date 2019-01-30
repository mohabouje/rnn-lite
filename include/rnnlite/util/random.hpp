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
* Filename: random.hpp
* Author: Mohammed Boujemaoui
* Date: 30/01/19
*/

#ifndef RNNLITE_RANDOM_HPP
#define RNNLITE_RANDOM_HPP

#include <cstdint>
#include <random>
#include <chrono>

namespace rnn {

    /**
     * @brief The DistributionType enum represents all the available distributions in the pseudo-random number
     * generation library.
     */
    enum class distribution_type {
        Uniform,           /*<! Produces real values evenly distributed across a range */
        Bernoulli,         /*<! Produces random values on a Bernoulli distribution */
        Binomial,          /*<! Produces random values on a binomial distribution */
        Geometric,         /*<! Produces random values on a geometric distribution */
        Poisson,           /*<! Produces random values on a Poisson distribution */
        Exponential,       /*<! Produces random values on a exponential distribution */
        Gamma,             /*<! Produces random values on a Gamma distribution */
        Weibull,           /*<! Produces random values on a Weibull distribution */
        ExtremeValue,      /*<! Produces random values on a extreme value distribution */
        Normal,            /*<! Produces random values on a normal distribution */
        LogNormal,         /*<! Produces random values on a logarithmic normal distribution */
        ChiSquared,        /*<! Produces random values on a chi-squared distribution */
        Cauchy,            /*<! Produces random values on a Cauchy distribution */
        Fisher,            /*<! Produces random values on a Fisher distribution */
        Student,           /*<! Produces random values on a Student distribution */
        Discrete,          /*<! Produces random values on a discrete distribution */
        PieceWiseConstant, /*<! Produces real values distributed on constant sub-intervals.  */
        PieceWiseLinear    /*<! Produces real values distributed on defined sub-intervals.  */
    };

    namespace internal {

        template <typename Distribution, typename Engine = std::mt19937>
        struct RandomGeneratorImpl {
            using value_type = typename Distribution::value_type;
            template <typename... Args>
            explicit RandomGeneratorImpl(Args... arg) :
                generator_(
                    Engine(static_cast<std::size_t>(std::chrono::system_clock::now().time_since_epoch().count()))),
                distribution_(Distribution(std::forward(arg...))) {}

            inline value_type operator()() {
                return static_cast<value_type>(distribution_(generator_));
            }

        private:
            Engine generator_;
            Distribution distribution_;
        };

        template <distribution_type Type, typename T>
        struct _RandomGenerator;

        template <typename T>
        struct _RandomGenerator<distribution_type::Uniform, T>
            : public RandomGeneratorImpl<std::uniform_real_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Bernoulli, T>
            : public RandomGeneratorImpl<std::bernoulli_distribution, T> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Binomial, T>
            : public RandomGeneratorImpl<std::binomial_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Geometric, T>
            : public RandomGeneratorImpl<std::geometric_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Poisson, T>
            : public RandomGeneratorImpl<std::poisson_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Exponential, T>
            : public RandomGeneratorImpl<std::exponential_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Gamma, T> : public RandomGeneratorImpl<std::gamma_distribution<T>> {
        };

        template <typename T>
        struct _RandomGenerator<distribution_type::Weibull, T>
            : public RandomGeneratorImpl<std::weibull_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::ExtremeValue, T>
            : public RandomGeneratorImpl<std::extreme_value_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Normal, T>
            : public RandomGeneratorImpl<std::normal_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::LogNormal, T>
            : public RandomGeneratorImpl<std::lognormal_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::ChiSquared, T>
            : public RandomGeneratorImpl<std::chi_squared_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Fisher, T>
            : public RandomGeneratorImpl<std::fisher_f_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Student, T>
            : public RandomGeneratorImpl<std::student_t_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Discrete, T>
            : public RandomGeneratorImpl<std::discrete_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::PieceWiseConstant, T>
            : public RandomGeneratorImpl<std::piecewise_constant_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::PieceWiseLinear, T>
            : public RandomGeneratorImpl<std::piecewise_linear_distribution<T>> {};

        template <typename T>
        struct _RandomGenerator<distribution_type::Cauchy, T>
            : public RandomGeneratorImpl<std::cauchy_distribution<T>> {};

    } // namespace internal

    /**
    * @class random_generator
    * @brief This class implements a random generator according to one of the discrete probability functions available
    * in the c++ standard.
    *
    * @see Distribution
    */
    template <distribution_type dist, typename T>
    struct random_generator {
        using value_type = T;

        /**
         * @brief Creates a random generator.
         * @param arg Arguments parameters to initialize the internal engine. It depends of the chosen distribution.
         * @see https://en.cppreference.com/w/cpp/numeric/random
         */
        template <typename... Args>
        explicit random_generator(Args... arg) : generator_(arg...) {}

        /**
         * @brief Generates a random number based in the chosen distribution.
         * @return The generated random number.
         */
        value_type operator()() {
            return generator_.operator()();
        }

    private:
        internal::_RandomGenerator<dist, T> generator_;
    };

} // namespace rnn

#endif //RNNLITE_RANDOM_HPP
