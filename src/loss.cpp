#include <torch/nn/module.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/pooling.h>
#include "lle/util.hpp"
#include "lle/loss.hpp"

namespace lle::loss
{
	torch::Tensor loss_color::forward(torch::Tensor x)
	{
		val mean_rgb = torch::mean(x, {2, 3}, true);
		val mrgbs = torch::split(mean_rgb, 1, 1);
		val mr = mrgbs[0];
		val mg = mrgbs[1];
		val mb = mrgbs[2];
		val drg = torch::pow(mr - mg, 2);
		val drb = torch::pow(mr - mb, 2);
		val dgb = torch::pow(mb - mg, 2);
		val k = torch::pow(torch::pow(drg, 2) + torch::pow(drb, 2) + torch::pow(dgb, 2), 0.5);
		return k;
	}

	loss_spa::loss_spa()
	{
		val kernel_left = util::float32_tensor({{0, 0, 0}, {-1, 1, 0}, {0, 0, 0}})
		                  .cpu()
		                  .unsqueeze(0)
		                  .unsqueeze(0);
		val kernel_right = util::float32_tensor({{0, 0, 0}, {0, 1, -1}, {0, 0, 0}})
		                   .cpu()
		                   .unsqueeze(0)
		                   .unsqueeze(0);
		val kernel_up = util::float32_tensor({{0, -1, 0}, {0, 1, 0}, {0, 0, 0}})
		                .cpu()
		                .unsqueeze(0)
		                .unsqueeze(0);
		val kernel_down = util::float32_tensor({{0, 0, 0}, {0, 1, 0}, {0, -1, 0}})
		                  .cpu()
		                  .unsqueeze(0)
		                  .unsqueeze(0);
		weight_left_ = register_parameter("weight_left", kernel_left, false);
		weight_right_ = register_parameter("weight_right", kernel_right, false);
		weight_up_ = register_parameter("weight_up", weight_up_, false);
		weight_down_ = register_parameter("weight_down", weight_down_, false);
		pool_ = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(4));
	}

	torch::Tensor loss_spa::forward(torch::Tensor org, torch::Tensor enhance)
	{
		val org_mean = torch::mean(org, 1, true);
		val enhance_mean = torch::mean(enhance, 1, true);

		val org_pool = pool_(org_mean);
		val enhance_pool = pool_(enhance_mean);


		val weight_diff = torch::max(util::float32_tensor({1}).cpu() + 10000 * torch::min(
			                             org_pool - util::float32_tensor({0.3}).cpu(),
			                             util::float32_tensor({0}).cpu()),
		                             util::float32_tensor({0.5}).cpu());
		val e_1 = torch::mul(torch::sign(enhance_pool - util::float32_tensor({0.5}).cpu()),
		                     enhance_pool - org_pool);
		val d_org_left = torch::conv2d(org_pool, weight_left_, {}, 1, 1);
		val d_org_right = torch::conv2d(org_pool, weight_right_, {}, 1, 1);
		val d_org_up = torch::conv2d(org_pool, weight_up_, {}, 1, 1);
		val d_org_down = torch::conv2d(org_pool, weight_down_, {}, 1, 1);

		val d_enhance_left = conv2d(enhance_pool, weight_left_, {}, 1, 1);
		val d_enhance_right = conv2d(enhance_pool, weight_right_, {}, 1, 1);
		val d_enhance_up = conv2d(enhance_pool, weight_up_, {}, 1, 1);
		val d_enhance_down = conv2d(enhance_pool, weight_down_, {}, 1, 1);

		val d_left = torch::pow(d_org_left - d_enhance_left, 2);
		val d_right = torch::pow(d_org_right - d_enhance_right, 2);
		val d_up = torch::pow(d_org_up - d_enhance_up, 2);
		val d_down = torch::pow(d_org_down - d_enhance_down, 2);

		val e = d_left + d_right + d_up + d_down;
		return e;
	}
	;

	loss_exp::loss_exp(const int64_t patch_size, const float mean_val)
	{
		pool_ = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(patch_size));
		this->mean_val_ = mean_val;
	}

	torch::Tensor loss_exp::forward(torch::Tensor x)
	{
		x = mean(x, {1}, true);
		val mean = pool_(x);
		val d = at::mean(at::pow(mean - util::float32_tensor({mean_val_}).cpu(), 2));
		return d;
	}
	;

	loss_tv::loss_tv(const float loss_tv_weight_) : loss_tv_weight_(loss_tv_weight_)
	{
	}

	torch::Tensor loss_tv::forward(torch::Tensor x)
	{
		val sizes = x.sizes();
		val batch_size = int64_t{sizes[0]};
		val h_x = int64_t{sizes[2]};
		val w_x = int64_t{sizes[3]};

		val count_h = (h_x - 1) * w_x;
		val count_w = h_x * (w_x - 1);

		val h_tv = pow(
				x.index({
					torch::indexing::Slice(), torch::indexing::Slice(), at::indexing::Slice(1, at::indexing::None),
					at::indexing::Slice()
				}) -
				x.index({
					at::indexing::Slice(), at::indexing::Slice(), at::indexing::Slice(at::indexing::None, h_x - 1),
					at::indexing::Slice()
				}),
				2)
			.sum();

		val w_tv = pow(
				x.index({
					at::indexing::Slice(), at::indexing::Slice(), at::indexing::Slice(),
					at::indexing::Slice(1, at::indexing::None)
				}) -
				x.index({
					at::indexing::Slice(), at::indexing::Slice(), at::indexing::Slice(),
					at::indexing::Slice(at::indexing::None, w_x - 1)
				}),
				2)
			.sum();
		val loss = loss_tv_weight_ * 2.0f * (h_tv / count_h + w_tv / count_w) / batch_size;
		return loss;
	}
	;


	torch::Tensor loss_sa::forward(torch::Tensor x)
	{
		val channels = split(x, 1, 1);
		val r = channels[0];
		val g = channels[1];
		val b = channels[2];

		// Calculate mean values for each channel
		val mean_rgb = mean(x, {2, 3}, true);
		val mean_channels = split(mean_rgb, 1, 1);
		val mr = mean_channels[0];
		val mg = mean_channels[1];
		val mb = mean_channels[2];

		// Calculate differences
		val dr = r - mr;
		val dg = g - mg;
		val db = b - mb;

		// Calculate final loss
		var k = torch::pow(torch::pow(dr, 2) + torch::pow(db, 2) + torch::pow(dg, 2), 0.5);
		k = torch::mean(k);

		return k;
	}
};
