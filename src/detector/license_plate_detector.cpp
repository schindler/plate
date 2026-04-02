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
    const image::ImageBuffer &source_image, const DetectorConfig &config,
    const filter::FilterStageCallback &stage_callback) {
  if (source_image.channels != 1 && source_image.channels != 3) {
    throw std::invalid_argument(
        "License plate detection expects a 1-channel or 3-channel image.");
  }

  if (config.scale.has_value() && *config.scale <= 0.0F) {
    throw std::invalid_argument("Scale must be a positive number.");
  }
  if (config.expected_plate_ratio <= 0.0F) {
    throw std::invalid_argument(
        "Expected plate ratio must be a positive number.");
  }
  if (config.min_area <= 0 || config.max_area <= 0) {
    throw std::invalid_argument("Area thresholds must be positive integers.");
  }
  if (config.min_area > config.max_area) {
    throw std::invalid_argument(
        "Minimum area cannot be larger than maximum area.");
  }
  if (config.min_density <= 0.0F || config.min_density > 1.0F) {
    throw std::invalid_argument(
        "Minimum density must be in the (0, 1] interval.");
  }

  filter::PlateMaskConfig mask_config;
  mask_config.scale = config.scale;
  const auto mask = filter::build_plate_candidate_mask(
      source_image, mask_config, stage_callback);
  const float scale = std::min(
      static_cast<float>(mask.width) / static_cast<float>(source_image.width),
      static_cast<float>(mask.height) /
          static_cast<float>(source_image.height));

  auto rects = find_candidates(mask, scale, config);

  return rects;
}

}  // namespace plate::detector
