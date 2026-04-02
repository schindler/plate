#pragma once

#include <functional>
#include <optional>
#include <string_view>

#include "plate/image/image_buffer.hpp"

namespace plate::filter {

using FilterStageCallback = std::function<void(
    std::string_view stage_name, const image::ImageBuffer &stage_image)>;

struct PlateMaskConfig {
  std::optional<float> scale;
  int target_max_width{1080};
  int target_max_height{600};
  int gradient_threshold{120};
  int closing_radius_x{8};
  int closing_radius_y{2};
};

image::ImageBuffer build_plate_candidate_mask(
    const image::ImageBuffer &source_image, const PlateMaskConfig &config = {},
    const FilterStageCallback &stage_callback = {});

}  // namespace plate::filter
