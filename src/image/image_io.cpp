#include "plate/image/image_io.hpp"

#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>

namespace plate::image {
namespace {

int cv_type_for_channels(const int channels) {
  switch (channels) {
    case 1:
      return CV_8UC1;
    case 3:
      return CV_8UC3;
    default:
      throw std::invalid_argument(
          "Only 1-channel and 3-channel 8-bit images are supported.");
  }
}

}  // namespace

ImageBuffer load_image(const std::filesystem::path &path,
                       const ImageReadMode mode) {
  const int flags =
      mode == ImageReadMode::kColor ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
  const cv::Mat decoded = cv::imread(path.string(), flags);
  if (decoded.empty()) {
    throw std::runtime_error("Failed to decode image: " + path.string());
  }

  ImageBuffer buffer;
  buffer.width = decoded.cols;
  buffer.height = decoded.rows;
  buffer.channels = decoded.channels();
  buffer.row_stride = buffer.width * buffer.channels;
  buffer.pixels.resize(static_cast<std::size_t>(buffer.row_stride) *
                       buffer.height);

  for (int row = 0; row < buffer.height; ++row) {
    const auto *source = decoded.ptr<std::uint8_t>(row);
    auto *destination = buffer.pixels.data() +
                        static_cast<std::size_t>(row) * buffer.row_stride;
    std::copy_n(source, buffer.row_stride, destination);
  }

  return buffer;
}

void save_image(const std::filesystem::path &path, const ImageBuffer &image) {
  if (image.empty()) {
    throw std::invalid_argument("Cannot save an empty image.");
  }

  const cv::Mat view(
      image.height, image.width, cv_type_for_channels(image.channels),
      const_cast<std::uint8_t *>(image.pixels.data()), image.row_stride);

  if (!cv::imwrite(path.string(), view)) {
    throw std::runtime_error("Failed to save image: " + path.string());
  }
}

std::filesystem::path make_output_path(
    const std::filesystem::path &input_path) {
  const auto parent = input_path.parent_path();
  const auto stem = input_path.stem().string();
  auto extension = input_path.extension().string();
  if (extension.empty()) {
    extension = ".png";
  }

  return parent / (stem + "-out" + extension);
}

}  // namespace plate::image
