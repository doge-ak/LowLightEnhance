#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <random>

#include "lle/load_data.hpp"


namespace lle::load_data
{
	low_light_data_set::low_light_data_set(const std::string& folder_path)
	{
		for (val& entry : std::filesystem::directory_iterator(folder_path))
		{  
			if (entry.path().extension() == ".jpg")
			{
				image_paths.push_back(entry.path());
			}
		}

		std::ranges::shuffle(image_paths, std::default_random_engine(1143));
	}

	torch::data::Example<> low_light_data_set::get(const size_t index) override
	{
		val img = cv::imread(image_paths[index].string(), cv::IMREAD_COLOR);
		if (img.empty())
		{
			throw std::runtime_error("Failed to load image: " + image_paths[index].string());
		}

		// Resize
		resize(img, img, cv::Size(size_, size_), 0, 0, cv::INTER_AREA);

		// Convert BGR to RGB
		cvtColor(img, img, cv::COLOR_BGR2RGB);

		// Convert to float and normalize to [0,1]
		img.convertTo(img, CV_32F, 1.0 / 255.0);

		// HWC to CHW
		var tensor = torch::from_blob(img.data, {size_, size_, 3}, torch::kFloat).clone();
		tensor = tensor.permute({2, 0, 1});

		return {tensor, tensor}; // 无标签任务，target 可以填 dummy
	}

	std::optional<size_t> low_light_data_set::size() const override
	{
		return image_paths.size();
	}
};
