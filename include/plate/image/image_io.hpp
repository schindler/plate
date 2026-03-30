#pragma once

#include <filesystem>

#include "plate/image/image_buffer.hpp"

namespace plate::image {

ImageBuffer load_image(const std::filesystem::path &path, ImageReadMode mode);

void save_image(const std::filesystem::path &path, const ImageBuffer &image);

std::filesystem::path make_output_path(const std::filesystem::path &input_path);

}  // namespace plate::image
