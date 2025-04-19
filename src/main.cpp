#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
	setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
	cv::Mat image = cv::imread("C:/Users/msav5/Downloads/snowy-peace.png");
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	image.convertTo(image, CV_32FC3, 1.0 / 255.0);
	torch::Tensor a = torch::from_blob(image.ptr(), {1, image.rows, image.cols, image.channels()})
		.clone();
	return 0;
}
