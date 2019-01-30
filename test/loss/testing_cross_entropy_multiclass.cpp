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
* Filename: testing_cross_entropy_multiclass.hpp
* Author: Mohammed Boujemaoui
* Date: 29/01/19
*/
#include <rnnlite/loss/cross_entropy_multiclass.hpp>
#include <gtest/gtest.h>
#include <vector>

TEST(TestingCrossEntropyMultiClass, SmallArrayWithBigNumbers) {
    const std::vector<double> expected = {
        0.,           6.7114094,    13.42281879,  20.13422819,  26.84563758,  33.55704698,  40.26845638,  46.97986577,
        53.69127517,  60.40268456,  67.11409396,  73.82550336,  80.53691275,  87.24832215,  93.95973154,  100.67114094,
        107.38255034, 114.09395973, 120.80536913, 127.51677852, 134.22818792, 140.93959732, 147.65100671, 154.36241611,
        161.0738255,  167.7852349,  174.4966443,  181.20805369, 187.91946309, 194.63087248, 201.34228188, 208.05369128,
        214.76510067, 221.47651007, 228.18791946, 234.89932886, 241.61073826, 248.32214765, 255.03355705, 261.74496644,
        268.45637584, 275.16778523, 281.87919463, 288.59060403, 295.30201342, 302.01342282, 308.72483221, 315.43624161,
        322.14765101, 328.8590604,  335.5704698,  342.28187919, 348.99328859, 355.70469799, 362.41610738, 369.12751678,
        375.83892617, 382.55033557, 389.26174497, 395.97315436, 402.68456376, 409.39597315, 416.10738255, 422.81879195,
        429.53020134, 436.24161074, 442.95302013, 449.66442953, 456.37583893, 463.08724832, 469.79865772, 476.51006711,
        483.22147651, 489.93288591, 496.6442953,  503.3557047,  510.06711409, 516.77852349, 523.48993289, 530.20134228,
        536.91275168, 543.62416107, 550.33557047, 557.04697987, 563.75838926, 570.46979866, 577.18120805, 583.89261745,
        590.60402685, 597.31543624, 604.02684564, 610.73825503, 617.44966443, 624.16107383, 630.87248322, 637.58389262,
        644.29530201, 651.00671141, 657.71812081, 664.4295302,  671.1409396,  677.85234899, 684.56375839, 691.27516779,
        697.98657718, 704.69798658, 711.40939597, 718.12080537, 724.83221477, 731.54362416, 738.25503356, 744.96644295,
        751.67785235, 758.38926174, 765.10067114, 771.81208054, 778.52348993, 785.23489933, 791.94630872, 798.65771812,
        805.36912752, 812.08053691, 818.79194631, 825.5033557,  832.2147651,  838.9261745,  845.63758389, 852.34899329,
        859.06040268, 865.77181208, 872.48322148, 879.19463087, 885.90604027, 892.61744966, 899.32885906, 906.04026846,
        912.75167785, 919.46308725, 926.17449664, 932.88590604, 939.59731544, 946.30872483, 953.02013423, 959.73154362,
        966.44295302, 973.15436242, 979.86577181, 986.57718121, 993.2885906,  1000.};

    const std::vector<double> predicted = {
        2.00000000e-02, 6.73140940e+00, 1.34428188e+01, 2.01542282e+01, 2.68656376e+01, 3.35770470e+01, 4.02884564e+01,
        4.69998658e+01, 5.37112752e+01, 6.04226846e+01, 6.71340940e+01, 7.38455034e+01, 8.05569128e+01, 8.72683221e+01,
        9.39797315e+01, 1.00691141e+02, 1.07402550e+02, 1.14113960e+02, 1.20825369e+02, 1.27536779e+02, 1.34248188e+02,
        1.40959597e+02, 1.47671007e+02, 1.54382416e+02, 1.61093826e+02, 1.67805235e+02, 1.74516644e+02, 1.81228054e+02,
        1.87939463e+02, 1.94650872e+02, 2.01362282e+02, 2.08073691e+02, 2.14785101e+02, 2.21496510e+02, 2.28207919e+02,
        2.34919329e+02, 2.41630738e+02, 2.48342148e+02, 2.55053557e+02, 2.61764966e+02, 2.68476376e+02, 2.75187785e+02,
        2.81899195e+02, 2.88610604e+02, 2.95322013e+02, 3.02033423e+02, 3.08744832e+02, 3.15456242e+02, 3.22167651e+02,
        3.28879060e+02, 3.35590470e+02, 3.42301879e+02, 3.49013289e+02, 3.55724698e+02, 3.62436107e+02, 3.69147517e+02,
        3.75858926e+02, 3.82570336e+02, 3.89281745e+02, 3.95993154e+02, 4.02704564e+02, 4.09415973e+02, 4.16127383e+02,
        4.22838792e+02, 4.29550201e+02, 4.36261611e+02, 4.42973020e+02, 4.49684430e+02, 4.56395839e+02, 4.63107248e+02,
        4.69818658e+02, 4.76530067e+02, 4.83241477e+02, 4.89952886e+02, 4.96664295e+02, 5.03375705e+02, 5.10087114e+02,
        5.16798523e+02, 5.23509933e+02, 5.30221342e+02, 5.36932752e+02, 5.43644161e+02, 5.50355570e+02, 5.57066980e+02,
        5.63778389e+02, 5.70489799e+02, 5.77201208e+02, 5.83912617e+02, 5.90624027e+02, 5.97335436e+02, 6.04046846e+02,
        6.10758255e+02, 6.17469664e+02, 6.24181074e+02, 6.30892483e+02, 6.37603893e+02, 6.44315302e+02, 6.51026711e+02,
        6.57738121e+02, 6.64449530e+02, 6.71160940e+02, 6.77872349e+02, 6.84583758e+02, 6.91295168e+02, 6.98006577e+02,
        7.04717987e+02, 7.11429396e+02, 7.18140805e+02, 7.24852215e+02, 7.31563624e+02, 7.38275034e+02, 7.44986443e+02,
        7.51697852e+02, 7.58409262e+02, 7.65120671e+02, 7.71832081e+02, 7.78543490e+02, 7.85254899e+02, 7.91966309e+02,
        7.98677718e+02, 8.05389128e+02, 8.12100537e+02, 8.18811946e+02, 8.25523356e+02, 8.32234765e+02, 8.38946174e+02,
        8.45657584e+02, 8.52368993e+02, 8.59080403e+02, 8.65791812e+02, 8.72503221e+02, 8.79214631e+02, 8.85926040e+02,
        8.92637450e+02, 8.99348859e+02, 9.06060268e+02, 9.12771678e+02, 9.19483087e+02, 9.26194497e+02, 9.32905906e+02,
        9.39617315e+02, 9.46328725e+02, 9.53040134e+02, 9.59751544e+02, 9.66462953e+02, 9.73174362e+02, 9.79885772e+02,
        9.86597181e+02, 9.93308591e+02, 1.00002000e+03};

    auto loss       = rnn::loss::cross_entropy_multiclass<double>{};
    const auto cost = loss(std::begin(predicted), std::end(predicted), std::begin(expected));
    EXPECT_NEAR(cost, 4.814095473337391, 1e-6);
}
