#pragma once
#define val const auto
#define var auto

namespace lle::model
{
	class cnn final : public torch::nn::Module
	{
		torch::nn::ReLU relu_;
		torch::nn::Conv2d conv1_;
		torch::nn::Conv2d conv2_;
		torch::nn::Conv2d conv3_;
		torch::nn::Conv2d conv4_;
		torch::nn::Conv2d conv5_;
		torch::nn::Conv2d conv6_;
		torch::nn::Conv2d conv7_;
		torch::nn::MaxPool2d max_pool_;
		torch::nn::Upsample up_sample_;

	public:
		cnn();

		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);
	};
}
