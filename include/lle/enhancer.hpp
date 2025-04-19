#pragma once
#define val const auto
#define var auto

namespace lle::enhancer
{
	/**
	 * ͼ����ǿ��
	 */
	class image_enhancer
	{
		/**
		 * ģ�͵��豸
		 */
		torch::Device device_;

		/**
		 * ģ��
		 */
		torch::jit::Module model_;

	public:
		/**
		 * ���캯��
		 * @param model_path ģ��·��
		 * @param device ģ�ͼ����豸
		 */
		explicit image_enhancer(const std::string& model_path, const torch::Device device = torch::kCPU);

		/**
		 * ����ͼ��
		 * @param image_path ͼ��·��
		 * @return ͼ������
		 */
		torch::Tensor load_image(const std::string& image_path);

		/**
		 * ��ǿͼ��
		 * @param input ��ǿǰ��ͼ������
		 * @return ��ǿ���ͼ������
		 */
		torch::Tensor enhance_image(const torch::Tensor& input);

		/**
		 * ����ͼ��
		 * @param tensor ��ǿ���ͼ������
		 * @param save_path ����·��
		 */
		void save_image(const torch::Tensor& tensor, const std::string& save_path);

		/**
		 * ��ȡ����ǿ������ͼ��
		 * @param image_path ͼ������·��
		 * @param result_path ͼ�����·��
		 */
		void process_image(const std::string& image_path, const std::string& result_path);
	};
}
