#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/pooling.h>
#include <torch/nn/modules/upsampling.h>

#include "lle/model.hpp"

namespace lle::model
{
	cnn::cnn()
	{
		relu_->options.inplace(false);
		val number_f = int32_t{32};

		val conv_options1 = torch::nn::Conv2dOptions(3, number_f, 3)
		                    .stride(1)
		                    .padding(1)
		                    .bias(true);
		conv1_ = torch::nn::Conv2d(conv_options1);

		val conv_options2 = torch::nn::Conv2dOptions(number_f, number_f, 3)
		                    .stride(1)
		                    .padding(1)
		                    .bias(true);
		conv2_ = torch::nn::Conv2d(conv_options2);

		val conv_options3 = torch::nn::Conv2dOptions(number_f, number_f, 3)
		                    .stride(1)
		                    .padding(1)
		                    .bias(true);
		conv3_ = torch::nn::Conv2d(conv_options3);

		val conv_options4 = torch::nn::Conv2dOptions(number_f, number_f, 3)
		                    .stride(1)
		                    .padding(1)
		                    .bias(true);
		conv4_ = torch::nn::Conv2d(conv_options4);

		val conv_options5 = torch::nn::Conv2dOptions(2 * number_f, number_f, 3)
		                    .stride(1)
		                    .padding(1)
		                    .bias(true);
		conv5_ = torch::nn::Conv2d(conv_options5);

		val conv_options6 = torch::nn::Conv2dOptions(2 * number_f, number_f, 3)
		                    .stride(1)
		                    .padding(1)
		                    .bias(true);
		conv6_ = torch::nn::Conv2d(conv_options6);

		val conv_options7 = torch::nn::Conv2dOptions(2 * number_f, 24, 3)
		                    .stride(1)
		                    .padding(1)
		                    .bias(true);
		conv7_ = torch::nn::Conv2d(conv_options7);

		val max_pool_option = torch::nn::MaxPool2dOptions(2)
		                      .stride(2)
		                      .ceil_mode(false);
		max_pool_ = torch::nn::MaxPool2d(max_pool_option);

		val up_sample_option = torch::nn::UpsampleOptions()
			.scale_factor(std::make_optional<std::vector<double>>({2}));
		up_sample_ = torch::nn::Upsample(up_sample_option);
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cnn::forward(torch::Tensor x)
	{
		val x1 = relu_(conv1_(x));
		val x2 = relu_(conv2_(x));
		val x3 = relu_(conv3_(x));
		val x4 = relu_(conv4_(x));

		val x5 = relu_(conv5_(torch::cat({x3, x4}, 1)));
		val x6 = relu_(conv6_(torch::cat({x2, x5}, 1)));

		val x_r = tanh(conv7_(torch::cat({x1, x6}, 1)));
		val rs = torch::split(x_r, 3, 1);
		val r1 = rs[0];
		val r2 = rs[1];
		val r3 = rs[2];
		val r4 = rs[3];
		val r5 = rs[4];
		val r6 = rs[5];
		val r7 = rs[6];
		val r8 = rs[7];

		x = x + r1 * (torch::pow(x, 2) - x);
		x = x + r2 * (torch::pow(x, 2) - x);
		x = x + r3 * (torch::pow(x, 2) - x);
		val enhance_image1 = x + r4 * (pow(x, 2) - x);
		x = enhance_image1 + r5 * (pow(enhance_image1, 2) - enhance_image1);
		x = x + r6 * (pow(x, 2) - x);
		x = x + r7 * (pow(x, 2) - x);
		val enhance_image = x + r8 * (pow(x, 2) - x);
		val r = cat(rs, 1);
		return std::make_tuple(enhance_image1, enhance_image, r);
	}
};
