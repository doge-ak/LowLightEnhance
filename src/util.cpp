#include "lle/util.hpp"

#define val const auto
#define var auto

namespace zero_dce::util
{
	inline torch::Tensor float32_tensor(torch::detail::TensorDataContainer tensor_data_container)
	{
		return torch::tensor(tensor_data_container, torch::TensorOptions().dtype(make_optional(torch::kFloat32)));
	}

	static var mutex = std::mutex();

	inline void exit(const int32_t code)
	{
		mutex.lock();
		std::exit(code);
	}
}
