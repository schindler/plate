#pragma once

#include <filesystem>
#include <string_view>

#include "plate/image/image_buffer.hpp"

namespace plate::image {

ImageBuffer load_image(const std::filesystem::path &path, ImageReadMode mode);

void save_image(const std::filesystem::path &path, const ImageBuffer &image);

std::filesystem::path make_output_path(const std::filesystem::path &input_path);

std::filesystem::path make_stage_output_path(
    const std::filesystem::path &input_path,
    const std::filesystem::path &output_directory, std::string_view stage_name);

}  // namespace plate::image
