#pragma once

#include "plate/image/image_buffer.hpp"

namespace plate::filter {

struct PlateMaskConfig {
  int gradient_threshold{120};
  int closing_radius_x{8};
  int closing_radius_y{2};
};

image::ImageBuffer build_plate_candidate_mask(
    const image::ImageBuffer &grayscale, const PlateMaskConfig &config = {});

}  // namespace plate::filter
