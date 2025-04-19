#pragma once
#include <torch/types.h>

namespace lle::util
{
	inline torch::Tensor float32_tensor(torch::detail::TensorDataContainer tensor_data_container);
	inline void exit(int32_t code);
}
