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
* Filename: testing_rectifier.cpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/

#include <rnnlite/activation/relu.hpp>
#include <rnnlite/activation/selu.hpp>
#include <rnnlite/activation/elu.hpp>
#include <rnnlite/activation/leaky_relu.hpp>

#include <vector>
#include <gtest/gtest.h>

TEST(TestingRectifier, TestingReLU) {

    const std::vector<double> input = {
            -2.000000000000, -1.973154306412, -1.946308732033, -1.919463038445,
            -1.892617464066, -1.865771770477, -1.838926196098, -1.812080502510,
            -1.785234928131, -1.758389234543, -1.731543660164, -1.704697966576,
            -1.677852392197, -1.651006698608, -1.624161005020, -1.597315430641,
            -1.570469856262, -1.543624162674, -1.516778469086, -1.489932894707,
            -1.463087320328, -1.436241626740, -1.409395933151, -1.382550358772,
            -1.355704665184, -1.328859090805, -1.302013397217, -1.275167703629,
            -1.248322129250, -1.221476554871, -1.194630861282, -1.167785167694,
            -1.140939593315, -1.114094018936, -1.087248325348, -1.060402631760,
            -1.033557057381, -1.006711483002, -0.979865789413, -0.953020095825,
            -0.926174521446, -0.899328827858, -0.872483253479, -0.845637559891,
            -0.818791985512, -0.791946291924, -0.765100717545, -0.738255023956,
            -0.711409330368, -0.684563755989, -0.657718062401, -0.630872488022,
            -0.604026794434, -0.577181220055, -0.550335526466, -0.523489952087,
            -0.496644258499, -0.469798684120, -0.442952990532, -0.416107416153,
            -0.389261722565, -0.362416148186, -0.335570454597, -0.308724880219,
            -0.281879186630, -0.255033493042, -0.228187918663, -0.201342225075,
            -0.174496650696, -0.147650957108, -0.120805382729, -0.093959689140,
            -0.067114114761, -0.040268421173, -0.013422846794,  0.013422727585,
            0.040268421173,  0.067114114761,  0.093959808350,  0.120805263519,
            0.147650957108,  0.174496650696,  0.201342344284,  0.228188037872,
            0.255033493042,  0.281879186630,  0.308724880219,  0.335570573807,
            0.362416028976,  0.389261722565,  0.416107416153,  0.442953109741,
            0.469798564911,  0.496644258499,  0.523489952087,  0.550335645676,
            0.577181339264,  0.604026794434,  0.630872488022,  0.657718181610,
            0.684563875198,  0.711409330368,  0.738255023956,  0.765100717545,
            0.791946411133,  0.818791866302,  0.845637559891,  0.872483253479,
            0.899328947067,  0.926174402237,  0.953020095825,  0.979865789413,
            1.006711483002,  1.033557176590,  1.060402631760,  1.087248325348,
            1.114094018936,  1.140939712524,  1.167785167694,  1.194630861282,
            1.221476554871,  1.248322248459,  1.275167703629,  1.302013397217,
            1.328859090805,  1.355704784393,  1.382550239563,  1.409395933151,
            1.436241626740,  1.463087320328,  1.489933013916,  1.516778469086,
            1.543624162674,  1.570469856262,  1.597315549850,  1.624161005020,
            1.651006698608,  1.677852392197,  1.704698085785,  1.731543540955,
            1.758389234543,  1.785234928131,  1.812080621719,  1.838926076889,
            1.865771770477,  1.892617464066,  1.919463157654,  1.946308851242,
            1.973154306412,  2.000000000000
    };

    const std::vector<double> output = {
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
            0.000000000000, 0.000000000000, 0.000000000000, 0.013422727585,
            0.040268421173, 0.067114114761, 0.093959808350, 0.120805263519,
            0.147650957108, 0.174496650696, 0.201342344284, 0.228188037872,
            0.255033493042, 0.281879186630, 0.308724880219, 0.335570573807,
            0.362416028976, 0.389261722565, 0.416107416153, 0.442953109741,
            0.469798564911, 0.496644258499, 0.523489952087, 0.550335645676,
            0.577181339264, 0.604026794434, 0.630872488022, 0.657718181610,
            0.684563875198, 0.711409330368, 0.738255023956, 0.765100717545,
            0.791946411133, 0.818791866302, 0.845637559891, 0.872483253479,
            0.899328947067, 0.926174402237, 0.953020095825, 0.979865789413,
            1.006711483002, 1.033557176590, 1.060402631760, 1.087248325348,
            1.114094018936, 1.140939712524, 1.167785167694, 1.194630861282,
            1.221476554871, 1.248322248459, 1.275167703629, 1.302013397217,
            1.328859090805, 1.355704784393, 1.382550239563, 1.409395933151,
            1.436241626740, 1.463087320328, 1.489933013916, 1.516778469086,
            1.543624162674, 1.570469856262, 1.597315549850, 1.624161005020,
            1.651006698608, 1.677852392197, 1.704698085785, 1.731543540955,
            1.758389234543, 1.785234928131, 1.812080621719, 1.838926076889,
            1.865771770477, 1.892617464066, 1.919463157654, 1.946308851242,
            1.973154306412, 2.000000000000
    };

    auto functor = rnn::activation::relu<double>{};
    for (auto i = 0ul; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], functor(input[i]), 1e-6);
    }
}


TEST(TestingRectifier, TestingLeakyReLU) {

    const std::vector<double> input = {
            -2.000000000000, -1.973154306412, -1.946308732033, -1.919463038445,
            -1.892617464066, -1.865771770477, -1.838926196098, -1.812080502510,
            -1.785234928131, -1.758389234543, -1.731543660164, -1.704697966576,
            -1.677852392197, -1.651006698608, -1.624161005020, -1.597315430641,
            -1.570469856262, -1.543624162674, -1.516778469086, -1.489932894707,
            -1.463087320328, -1.436241626740, -1.409395933151, -1.382550358772,
            -1.355704665184, -1.328859090805, -1.302013397217, -1.275167703629,
            -1.248322129250, -1.221476554871, -1.194630861282, -1.167785167694,
            -1.140939593315, -1.114094018936, -1.087248325348, -1.060402631760,
            -1.033557057381, -1.006711483002, -0.979865789413, -0.953020095825,
            -0.926174521446, -0.899328827858, -0.872483253479, -0.845637559891,
            -0.818791985512, -0.791946291924, -0.765100717545, -0.738255023956,
            -0.711409330368, -0.684563755989, -0.657718062401, -0.630872488022,
            -0.604026794434, -0.577181220055, -0.550335526466, -0.523489952087,
            -0.496644258499, -0.469798684120, -0.442952990532, -0.416107416153,
            -0.389261722565, -0.362416148186, -0.335570454597, -0.308724880219,
            -0.281879186630, -0.255033493042, -0.228187918663, -0.201342225075,
            -0.174496650696, -0.147650957108, -0.120805382729, -0.093959689140,
            -0.067114114761, -0.040268421173, -0.013422846794,  0.013422727585,
            0.040268421173,  0.067114114761,  0.093959808350,  0.120805263519,
            0.147650957108,  0.174496650696,  0.201342344284,  0.228188037872,
            0.255033493042,  0.281879186630,  0.308724880219,  0.335570573807,
            0.362416028976,  0.389261722565,  0.416107416153,  0.442953109741,
            0.469798564911,  0.496644258499,  0.523489952087,  0.550335645676,
            0.577181339264,  0.604026794434,  0.630872488022,  0.657718181610,
            0.684563875198,  0.711409330368,  0.738255023956,  0.765100717545,
            0.791946411133,  0.818791866302,  0.845637559891,  0.872483253479,
            0.899328947067,  0.926174402237,  0.953020095825,  0.979865789413,
            1.006711483002,  1.033557176590,  1.060402631760,  1.087248325348,
            1.114094018936,  1.140939712524,  1.167785167694,  1.194630861282,
            1.221476554871,  1.248322248459,  1.275167703629,  1.302013397217,
            1.328859090805,  1.355704784393,  1.382550239563,  1.409395933151,
            1.436241626740,  1.463087320328,  1.489933013916,  1.516778469086,
            1.543624162674,  1.570469856262,  1.597315549850,  1.624161005020,
            1.651006698608,  1.677852392197,  1.704698085785,  1.731543540955,
            1.758389234543,  1.785234928131,  1.812080621719,  1.838926076889,
            1.865771770477,  1.892617464066,  1.919463157654,  1.946308851242,
            1.973154306412,  2.000000000000
    };

    const std::vector<double> output = {
            -1.999999955297e-02, -1.973154209554e-02, -1.946308650076e-02,
            -1.919462904334e-02, -1.892617344856e-02, -1.865771785378e-02,
            -1.838926225901e-02, -1.812080480158e-02, -1.785234920681e-02,
            -1.758389174938e-02, -1.731543615460e-02, -1.704697869718e-02,
            -1.677852310240e-02, -1.651006750762e-02, -1.624161005020e-02,
            -1.597315445542e-02, -1.570469886065e-02, -1.543624140322e-02,
            -1.516778394580e-02, -1.489932835102e-02, -1.463087275624e-02,
            -1.436241623014e-02, -1.409395877272e-02, -1.382550317794e-02,
            -1.355704665184e-02, -1.328859105706e-02, -1.302013359964e-02,
            -1.275167707354e-02, -1.248322147876e-02, -1.221476495266e-02,
            -1.194630842656e-02, -1.167785096914e-02, -1.140939537436e-02,
            -1.114093977958e-02, -1.087248325348e-02, -1.060402579606e-02,
            -1.033557020128e-02, -1.006711460650e-02, -9.798658080399e-03,
            -9.530200622976e-03, -9.261745028198e-03, -8.993288502097e-03,
            -8.724831975996e-03, -8.456375449896e-03, -8.187919855118e-03,
            -7.919462397695e-03, -7.651006802917e-03, -7.382550276816e-03,
            -7.114093285054e-03, -6.845637224615e-03, -6.577180698514e-03,
            -6.308724638075e-03, -6.040267646313e-03, -5.771812051535e-03,
            -5.503355059773e-03, -5.234899464995e-03, -4.966442473233e-03,
            -4.697986878455e-03, -4.429529886693e-03, -4.161074291915e-03,
            -3.892617067322e-03, -3.624161472544e-03, -3.355704480782e-03,
            -3.087248653173e-03, -2.818791894242e-03, -2.550334902480e-03,
            -2.281879074872e-03, -2.013422315940e-03, -1.744966488332e-03,
            -1.476509496570e-03, -1.208053785376e-03, -9.395968518220e-04,
            -6.711411406286e-04, -4.026842070743e-04, -1.342284667771e-04,
            1.342272758484e-02,  4.026842117310e-02,  6.711411476135e-02,
            9.395980834961e-02,  1.208052635193e-01,  1.476509571075e-01,
            1.744966506958e-01,  2.013423442841e-01,  2.281880378723e-01,
            2.550334930420e-01,  2.818791866302e-01,  3.087248802185e-01,
            3.355705738068e-01,  3.624160289764e-01,  3.892617225647e-01,
            4.161074161530e-01,  4.429531097412e-01,  4.697985649109e-01,
            4.966442584991e-01,  5.234899520874e-01,  5.503356456757e-01,
            5.771813392639e-01,  6.040267944336e-01,  6.308724880219e-01,
            6.577181816101e-01,  6.845638751984e-01,  7.114093303680e-01,
            7.382550239563e-01,  7.651007175446e-01,  7.919464111328e-01,
            8.187918663025e-01,  8.456375598907e-01,  8.724832534790e-01,
            8.993289470673e-01,  9.261744022369e-01,  9.530200958252e-01,
            9.798657894135e-01,  1.006711483002e+00,  1.033557176590e+00,
            1.060402631760e+00,  1.087248325348e+00,  1.114094018936e+00,
            1.140939712524e+00,  1.167785167694e+00,  1.194630861282e+00,
            1.221476554871e+00,  1.248322248459e+00,  1.275167703629e+00,
            1.302013397217e+00,  1.328859090805e+00,  1.355704784393e+00,
            1.382550239563e+00,  1.409395933151e+00,  1.436241626740e+00,
            1.463087320328e+00,  1.489933013916e+00,  1.516778469086e+00,
            1.543624162674e+00,  1.570469856262e+00,  1.597315549850e+00,
            1.624161005020e+00,  1.651006698608e+00,  1.677852392197e+00,
            1.704698085785e+00,  1.731543540955e+00,  1.758389234543e+00,
            1.785234928131e+00,  1.812080621719e+00,  1.838926076889e+00,
            1.865771770477e+00,  1.892617464066e+00,  1.919463157654e+00,
            1.946308851242e+00,  1.973154306412e+00,  2.000000000000e+00
    };

    auto functor = rnn::activation::leaky_relu<double>{};
    for (auto i = 0ul; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], functor(input[i]), 1e-6);
    }
}


TEST(TestingRectifier, TestingELU) {

    const std::vector<double> input = {
            -2.000000000000, -1.973154306412, -1.946308732033, -1.919463038445,
            -1.892617464066, -1.865771770477, -1.838926196098, -1.812080502510,
            -1.785234928131, -1.758389234543, -1.731543660164, -1.704697966576,
            -1.677852392197, -1.651006698608, -1.624161005020, -1.597315430641,
            -1.570469856262, -1.543624162674, -1.516778469086, -1.489932894707,
            -1.463087320328, -1.436241626740, -1.409395933151, -1.382550358772,
            -1.355704665184, -1.328859090805, -1.302013397217, -1.275167703629,
            -1.248322129250, -1.221476554871, -1.194630861282, -1.167785167694,
            -1.140939593315, -1.114094018936, -1.087248325348, -1.060402631760,
            -1.033557057381, -1.006711483002, -0.979865789413, -0.953020095825,
            -0.926174521446, -0.899328827858, -0.872483253479, -0.845637559891,
            -0.818791985512, -0.791946291924, -0.765100717545, -0.738255023956,
            -0.711409330368, -0.684563755989, -0.657718062401, -0.630872488022,
            -0.604026794434, -0.577181220055, -0.550335526466, -0.523489952087,
            -0.496644258499, -0.469798684120, -0.442952990532, -0.416107416153,
            -0.389261722565, -0.362416148186, -0.335570454597, -0.308724880219,
            -0.281879186630, -0.255033493042, -0.228187918663, -0.201342225075,
            -0.174496650696, -0.147650957108, -0.120805382729, -0.093959689140,
            -0.067114114761, -0.040268421173, -0.013422846794,  0.013422727585,
            0.040268421173,  0.067114114761,  0.093959808350,  0.120805263519,
            0.147650957108,  0.174496650696,  0.201342344284,  0.228188037872,
            0.255033493042,  0.281879186630,  0.308724880219,  0.335570573807,
            0.362416028976,  0.389261722565,  0.416107416153,  0.442953109741,
            0.469798564911,  0.496644258499,  0.523489952087,  0.550335645676,
            0.577181339264,  0.604026794434,  0.630872488022,  0.657718181610,
            0.684563875198,  0.711409330368,  0.738255023956,  0.765100717545,
            0.791946411133,  0.818791866302,  0.845637559891,  0.872483253479,
            0.899328947067,  0.926174402237,  0.953020095825,  0.979865789413,
            1.006711483002,  1.033557176590,  1.060402631760,  1.087248325348,
            1.114094018936,  1.140939712524,  1.167785167694,  1.194630861282,
            1.221476554871,  1.248322248459,  1.275167703629,  1.302013397217,
            1.328859090805,  1.355704784393,  1.382550239563,  1.409395933151,
            1.436241626740,  1.463087320328,  1.489933013916,  1.516778469086,
            1.543624162674,  1.570469856262,  1.597315549850,  1.624161005020,
            1.651006698608,  1.677852392197,  1.704698085785,  1.731543540955,
            1.758389234543,  1.785234928131,  1.812080621719,  1.838926076889,
            1.865771770477,  1.892617464066,  1.919463157654,  1.946308851242,
            1.973154306412,  2.000000000000
    };

    const std::vector<double> output = {
            -0.864664733410, -0.860982358456, -0.857199788094, -0.853314280510,
            -0.849323093891, -0.845223307610, -0.841011941433, -0.836686015129,
            -0.832242369652, -0.827677786350, -0.822989046574, -0.818172693253,
            -0.813225328922, -0.808143317699, -0.802923023701, -0.797560751438,
            -0.792052567005, -0.786394417286, -0.780582368374, -0.774612247944,
            -0.768479585648, -0.762180089951, -0.755709171295, -0.749062240124,
            -0.742234408855, -0.735220849514, -0.728016376495, -0.720615863800,
            -0.713014066219, -0.705205440521, -0.697184264660, -0.688944876194,
            -0.680481314659, -0.671787500381, -0.662857055664, -0.653683662415,
            -0.644260704517, -0.634581327438, -0.624638497829, -0.614425182343,
            -0.603934049606, -0.593157351017, -0.582087516785, -0.570716440678,
            -0.559035956860, -0.547037661076, -0.534712910652, -0.522052824497,
            -0.509048223495, -0.495689809322, -0.481967896223, -0.467872679234,
            -0.453393876553, -0.438521176577, -0.423243731260, -0.407550692558,
            -0.391430556774, -0.374871909618, -0.357862621546, -0.340390592813,
            -0.322443097830, -0.304007321596, -0.285069853067, -0.265617221594,
            -0.245635181665, -0.225109457970, -0.204025328159, -0.182367429137,
            -0.160120338202, -0.137267813087, -0.113793589175, -0.089680545032,
            -0.064911514521, -0.039468422532, -0.013333162293,  0.013422727585,
            0.040268421173,  0.067114114761,  0.093959808350,  0.120805263519,
            0.147650957108,  0.174496650696,  0.201342344284,  0.228188037872,
            0.255033493042,  0.281879186630,  0.308724880219,  0.335570573807,
            0.362416028976,  0.389261722565,  0.416107416153,  0.442953109741,
            0.469798564911,  0.496644258499,  0.523489952087,  0.550335645676,
            0.577181339264,  0.604026794434,  0.630872488022,  0.657718181610,
            0.684563875198,  0.711409330368,  0.738255023956,  0.765100717545,
            0.791946411133,  0.818791866302,  0.845637559891,  0.872483253479,
            0.899328947067,  0.926174402237,  0.953020095825,  0.979865789413,
            1.006711483002,  1.033557176590,  1.060402631760,  1.087248325348,
            1.114094018936,  1.140939712524,  1.167785167694,  1.194630861282,
            1.221476554871,  1.248322248459,  1.275167703629,  1.302013397217,
            1.328859090805,  1.355704784393,  1.382550239563,  1.409395933151,
            1.436241626740,  1.463087320328,  1.489933013916,  1.516778469086,
            1.543624162674,  1.570469856262,  1.597315549850,  1.624161005020,
            1.651006698608,  1.677852392197,  1.704698085785,  1.731543540955,
            1.758389234543,  1.785234928131,  1.812080621719,  1.838926076889,
            1.865771770477,  1.892617464066,  1.919463157654,  1.946308851242,
            1.973154306412,  2.000000000000
    };

    auto functor = rnn::activation::elu<double>{};
    for (auto i = 0ul; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], functor(input[i]), 1e-6);
    }
}

TEST(TestingRectifier, TestingSELU) {

    const std::vector<double> input = {
            -2.000000000000, -1.973154306412, -1.946308732033, -1.919463038445,
            -1.892617464066, -1.865771770477, -1.838926196098, -1.812080502510,
            -1.785234928131, -1.758389234543, -1.731543660164, -1.704697966576,
            -1.677852392197, -1.651006698608, -1.624161005020, -1.597315430641,
            -1.570469856262, -1.543624162674, -1.516778469086, -1.489932894707,
            -1.463087320328, -1.436241626740, -1.409395933151, -1.382550358772,
            -1.355704665184, -1.328859090805, -1.302013397217, -1.275167703629,
            -1.248322129250, -1.221476554871, -1.194630861282, -1.167785167694,
            -1.140939593315, -1.114094018936, -1.087248325348, -1.060402631760,
            -1.033557057381, -1.006711483002, -0.979865789413, -0.953020095825,
            -0.926174521446, -0.899328827858, -0.872483253479, -0.845637559891,
            -0.818791985512, -0.791946291924, -0.765100717545, -0.738255023956,
            -0.711409330368, -0.684563755989, -0.657718062401, -0.630872488022,
            -0.604026794434, -0.577181220055, -0.550335526466, -0.523489952087,
            -0.496644258499, -0.469798684120, -0.442952990532, -0.416107416153,
            -0.389261722565, -0.362416148186, -0.335570454597, -0.308724880219,
            -0.281879186630, -0.255033493042, -0.228187918663, -0.201342225075,
            -0.174496650696, -0.147650957108, -0.120805382729, -0.093959689140,
            -0.067114114761, -0.040268421173, -0.013422846794,  0.013422727585,
            0.040268421173,  0.067114114761,  0.093959808350,  0.120805263519,
            0.147650957108,  0.174496650696,  0.201342344284,  0.228188037872,
            0.255033493042,  0.281879186630,  0.308724880219,  0.335570573807,
            0.362416028976,  0.389261722565,  0.416107416153,  0.442953109741,
            0.469798564911,  0.496644258499,  0.523489952087,  0.550335645676,
            0.577181339264,  0.604026794434,  0.630872488022,  0.657718181610,
            0.684563875198,  0.711409330368,  0.738255023956,  0.765100717545,
            0.791946411133,  0.818791866302,  0.845637559891,  0.872483253479,
            0.899328947067,  0.926174402237,  0.953020095825,  0.979865789413,
            1.006711483002,  1.033557176590,  1.060402631760,  1.087248325348,
            1.114094018936,  1.140939712524,  1.167785167694,  1.194630861282,
            1.221476554871,  1.248322248459,  1.275167703629,  1.302013397217,
            1.328859090805,  1.355704784393,  1.382550239563,  1.409395933151,
            1.436241626740,  1.463087320328,  1.489933013916,  1.516778469086,
            1.543624162674,  1.570469856262,  1.597315549850,  1.624161005020,
            1.651006698608,  1.677852392197,  1.704698085785,  1.731543540955,
            1.758389234543,  1.785234928131,  1.812080621719,  1.838926076889,
            1.865771770477,  1.892617464066,  1.919463157654,  1.946308851242,
            1.973154306412,  2.000000000000
    };

    const std::vector<double> output = {
            -1.520166397095, -1.513692498207, -1.507042407990, -1.500211238861,
            -1.493194341660, -1.485986471176, -1.478582501411, -1.470977067947,
            -1.463164687157, -1.455139756203, -1.446896433830, -1.438428878784,
            -1.429730892181, -1.420796275139, -1.411618471146, -1.402191042900,
            -1.392507076263, -1.382559537888, -1.372341394424, -1.361845254898,
            -1.351063489914, -1.339988350868, -1.328611850739, -1.316925764084,
            -1.304921865463, -1.292591214180, -1.279925107956, -1.266914248466,
            -1.253549575806, -1.239821195602, -1.225719213486, -1.211233496666,
            -1.196353793144, -1.181069135666, -1.165368556976, -1.149240732193,
            -1.132674217224, -1.115656971931, -1.098176598549, -1.080220580101,
            -1.061776041985, -1.042829513550, -1.023367643356, -1.003376126289,
            -0.982840776443, -0.961746513844, -0.940078437328, -0.917820692062,
            -0.894957304001, -0.871471941471, -0.847347438335, -0.822566628456,
            -0.797111451626, -0.770963788033, -0.744104504585, -0.716514587402,
            -0.688173830509, -0.659062027931, -0.629158020020, -0.598440468311,
            -0.566886961460, -0.534475088120, -0.501181125641, -0.466981440783,
            -0.431851059198, -0.395764768124, -0.358696788549, -0.320620059967,
            -0.281507462263, -0.241330444813, -0.200060427189, -0.157667294145,
            -0.114120885730, -0.069389410317, -0.023441024125,  0.014103273861,
            0.042310070246,  0.070516869426,  0.098723664880,  0.126930207014,
            0.155137017369,  0.183343812823,  0.211550608277,  0.239757403731,
            0.267963945866,  0.296170741320,  0.324377536774,  0.352584332228,
            0.380790889263,  0.408997684717,  0.437204480171,  0.465411275625,
            0.493617832661,  0.521824657917,  0.550031423569,  0.578238248825,
            0.606445014477,  0.634651541710,  0.662858366966,  0.691065192223,
            0.719271957874,  0.747478485107,  0.775685310364,  0.803892135620,
            0.832098901272,  0.860305428505,  0.888512253761,  0.916719019413,
            0.944925844669,  0.973132371902,  1.001339197159,  1.029546022415,
            1.057752728462,  1.085959553719,  1.114166140556,  1.142372965813,
            1.170579671860,  1.198786497116,  1.226993083954,  1.255199909210,
            1.283406615257,  1.311613440514,  1.339820027351,  1.368026852608,
            1.396233558655,  1.424440383911,  1.452646970749,  1.480853796005,
            1.509060502052,  1.537267327309,  1.565474152565,  1.593680739403,
            1.621887445450,  1.650094270706,  1.678301095963,  1.706507682800,
            1.734714388847,  1.762921214104,  1.791128039360,  1.819334626198,
            1.847541332245,  1.875748157501,  1.903954982758,  1.932161450386,
            1.960368275642,  1.988575100899,  2.016781806946,  2.044988632202,
            2.073195219040,  2.101402044296
    };

    auto functor = rnn::activation::selu<double>{};
    for (auto i = 0ul; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], functor(input[i]), 1e-5);
    }
}


