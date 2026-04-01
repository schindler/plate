#include "plate/detector/license_plate_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "plate/core/bounding_box.hpp"
#include "plate/filter/npp_filters.hpp"

namespace plate::detector {
 

std::vector<core::BoundingBox> detect_license_plate(
    const image::ImageBuffer &source_image,
    const filter::FilterStageCallback &stage_callback) {
  if (source_image.channels != 1 && source_image.channels != 3) {
    throw std::invalid_argument(
        "License plate detection expects a 1-channel or 3-channel image.");
  }

  const auto mask =
      filter::build_plate_candidate_mask(source_image, {}, stage_callback);
  const float scale = std::min((float)mask.width / source_image.width,
                               (float)mask.height / source_image.height);

  auto rects = find_candidates(mask, scale);

  return rects;
}

}  // namespace plate::detector
