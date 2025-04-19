#pragma once

namespace lle::config {
	class train_config
	{
	public:
		int64_t batch_size = 16;
		int64_t epoch_count = 200;
		int64_t display_iter = 10;
		int64_t snapshot_iter = 10;
		std::string snapshot_folder = "snapshots/";
		std::string data_set_path = "path/to/your/dataset";
	};
}