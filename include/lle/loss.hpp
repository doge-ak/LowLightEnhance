#pragma once
#define val const auto
#define var auto

namespace lle::loss
{
	class loss_color final : public torch::nn::Module
	{
	public:
		torch::Tensor forward(torch::Tensor x);
	};

	class loss_spa final : public torch::nn::Module
	{
		torch::Tensor weight_left_;
		torch::Tensor weight_right_;
		torch::Tensor weight_up_;
		torch::Tensor weight_down_;
		torch::nn::AvgPool2d pool_;

	public:
		loss_spa();

		torch::Tensor forward(torch::Tensor org, torch::Tensor enhance);
	};

	class loss_exp final : torch::nn::Module
	{
		torch::nn::AvgPool2d pool_;
		int64_t mean_val_;

	public:
		loss_exp(const int64_t patch_size, const float mean_val);

		torch::Tensor forward(torch::Tensor x);
	};

	class loss_tv : torch::nn::Module
	{
		float loss_tv_weight_;

	public:
		loss_tv(const float loss_tv_weight_ = 1.0f);


		torch::Tensor forward(torch::Tensor x);
	};

	class loss_sa final : torch::nn::Module
	{
	public:
		torch::Tensor forward(torch::Tensor x);
	};
}
