#include <string>
#include <format>
#include <torch/script.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "lle/util.hpp"
#include "lle/enhancer.hpp"

namespace lle::enhancer
{
	image_enhancer::image_enhancer(const std::string& model_path, const torch::Device device): device_(device)
	{
		model_ = torch::jit::load(model_path);
		model_.to(device_);
	}

	torch::Tensor image_enhancer::load_image(const std::string& image_path)
	{
		// channel是BGR
		var img = cv::imread(image_path, cv::IMREAD_COLOR);

		if (img.empty())
		{
			std::cerr << std::format("failed to load image: {}\n", image_path);
			util::exit(1);
		}

		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		// 数据分布是 H(height), W(width), C(channel(RGB))
		var tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kU8).clone().to(device_);
		// 改变维度顺序 (C, H, W)
		tensor = tensor.permute({2, 0, 1});
		// 转换为 float 类型并归一化
		tensor = tensor.to(torch::kF32).div(255.0);
		// 增加维度 (1, C, H, W)
		tensor = tensor.unsqueeze(0);
		return tensor;
	}

	torch::Tensor image_enhancer::enhance_image(const torch::Tensor& input)
	{
		val output = model_.forward({input}).toTensor();
		return output;
	}

	void image_enhancer::save_image(const torch::Tensor& tensor, const std::string& save_path)
	{
		// 1 C H W 转换为 H W C, 归一化到 0-255, 数据裁剪, 转换为uint8, 移动到CPU
		val img = tensor.squeeze().permute({1, 2, 0}).mul(255).clamp(0, 255).to(torch::kU8).cpu();
		var output_img = cv::Mat(img.size(0), img.size(1), CV_8UC3, img.data_ptr());
		cv::cvtColor(output_img, output_img, cv::COLOR_RGB2BGR);
		cv::imwrite(save_path, output_img);
	}

	void image_enhancer::process_image(const std::string& image_path, const std::string& result_path)
	{
		val input_tensor = load_image(image_path);

		val start_time = std::chrono::high_resolution_clock::now();

		val enhanced_image = enhance_image(input_tensor);

		val end_time = std::chrono::high_resolution_clock::now();
		val duration = end_time - start_time;

		std::cout << std::format("Inference time: {} seconds.\n", duration.count());

		save_image(enhanced_image, result_path);
	}
};
