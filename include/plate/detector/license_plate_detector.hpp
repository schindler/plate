#pragma once

#include "plate/core/bounding_box.hpp"
#include "plate/filter/npp_filters.hpp"
#include "plate/image/image_buffer.hpp"

namespace plate::detector {

struct DetectionResult {
  core::BoundingBox box{};
  double score{};

  [[nodiscard]] bool found() const noexcept { return !box.empty(); }
};

DetectionResult detect_license_plate(
    const image::ImageBuffer &source_image,
    const filter::FilterStageCallback &stage_callback = {});

}  // namespace plate::detector
