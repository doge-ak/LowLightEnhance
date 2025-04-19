#pragma once
#define val const auto
#define var auto

namespace lle::enhancer
{
	/**
	 * 图像增强器
	 */
	class image_enhancer
	{
		/**
		 * 模型的设备
		 */
		torch::Device device_;

		/**
		 * 模型
		 */
		torch::jit::Module model_;

	public:
		/**
		 * 构造函数
		 * @param model_path 模型路径
		 * @param device 模型加载设备
		 */
		explicit image_enhancer(const std::string& model_path, const torch::Device device = torch::kCPU);

		/**
		 * 加载图像
		 * @param image_path 图像路径
		 * @return 图像张量
		 */
		torch::Tensor load_image(const std::string& image_path);

		/**
		 * 增强图像
		 * @param input 增强前的图像张量
		 * @return 增强后的图像张量
		 */
		torch::Tensor enhance_image(const torch::Tensor& input);

		/**
		 * 保存图像
		 * @param tensor 增强后的图像张量
		 * @param save_path 保存路径
		 */
		void save_image(const torch::Tensor& tensor, const std::string& save_path);

		/**
		 * 读取、增强、保存图像
		 * @param image_path 图像输入路径
		 * @param result_path 图像输出路径
		 */
		void process_image(const std::string& image_path, const std::string& result_path);
	};
}
