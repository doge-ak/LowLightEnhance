#pragma once
#define val const auto
#define var auto

namespace lle::load_data
{
	class low_light_data_set : public torch::data::Dataset<low_light_data_set>
	{
		std::vector<std::filesystem::path> image_paths;
		int32_t size_ = 256;

	public:
		low_light_data_set(const std::string& folder_path);

		torch::data::Example<> get(const size_t index) override;

		std::optional<size_t> size() const override;
	};
}
