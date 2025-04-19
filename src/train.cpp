#include <c10/core/Device.h>
#include "lle/config.hpp"
#include "lle/load_data.hpp"
#include "lle/loss.hpp"
#include "lle/model.hpp"
#include "torch/torch.h"
#include "torch/data/dataloader.h"
#include "torch/optim/adam.h"

#define val const auto
#define var auto

namespace lle::train
{
	static void train(config::train_config& config)
	{
		val device = torch::Device(torch::kCPU);
		var model = model::cnn();
		model.to(device);

		val& data_set_path = config.data_set_path;
		val data_set = load_data::low_light_data_set(data_set_path)
			.map(torch::data::transforms::Stack());
		val train_loader = torch::data::make_data_loader(data_set, torch::data::DataLoaderOptions());

		var loss_color = loss::loss_color();
		var loss_exp = loss::loss_exp(16, 0.6);
		var loss_tv = loss::loss_tv();
		var loss_spa = loss::loss_spa();

		var optimizer = torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(0.001).weight_decay(0.001));
		model.train();
		val epoch_count = config.epoch_count;
		int display_iter = 10;
		int snapshot_iter = 10;

		std::string snapshot_folder = "snapshots/";

		for (var epoch = int32_t{0}; epoch < epoch_count; epoch++)
		{
			var iter = int32_t{0};
			for (val& batch : *train_loader)
			{
				val input = batch.data.to(device);
				val [enhanced1,enhanced,A] = model.forward(input);

				val loss_tv1 = loss_tv.forward(enhanced);
				val loss_spa1 = loss_spa.forward(enhanced, input).mean();
				val loss_col1 = 5 * loss_color.forward(enhanced).mean();
				val loss_exp1 = 10 * loss_exp.forward(enhanced).mean();

				val loss = loss_tv1 + loss_spa1 + loss_col1 + loss_exp1;

				optimizer.zero_grad();
				loss.backward();
				torch::nn::utils::clip_grad_norm_(model.parameters(), 0.1);
				optimizer.step();
				if ((iter + 1) % display_iter == 0)
				{
					std::cout << std::format("Epoch [{}] Iter [{}] Loss: {:.6f}", epoch, iter + 1,
					                         loss.item<float>()) << '\n';
				}

				if ((iter + 1) % snapshot_iter == 0)
				{
					std::string save_path = snapshot_folder + "Epoch" + std::to_string(epoch) + ".pt";
					torch::save(model, save_path);
				}
			}
		}
	}
} // namespace zero_dce::train
