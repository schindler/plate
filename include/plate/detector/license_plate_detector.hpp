#pragma once

#include <optional>
#include <vector>

#include "plate/core/bounding_box.hpp"
#include "plate/filter/npp_filters.hpp"
#include "plate/image/image_buffer.hpp"

namespace plate::detector {

struct DetectorConfig {
  std::optional<float> scale;
  float expected_plate_ratio{4.0F};
  int min_area{2500};
  int max_area{20000};
  float min_density{0.5F};
};

std::vector<core::BoundingBox> detect_license_plate(
    const image::ImageBuffer &source_image, const DetectorConfig &config = {},
    const filter::FilterStageCallback &stage_callback = {});

std::vector<core::BoundingBox> find_candidates(
    const image::ImageBuffer &img, float scale,
    const DetectorConfig &config = {});

}  // namespace plate::detector
