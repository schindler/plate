#pragma once

#include "plate/core/bounding_box.hpp"
#include "plate/filter/npp_filters.hpp"
#include "plate/image/image_buffer.hpp"

namespace plate::detector {

std::vector<core::BoundingBox> detect_license_plate(
    const image::ImageBuffer &source_image,
    const filter::FilterStageCallback &stage_callback = {});

std::vector<core::BoundingBox> find_candidates(const image::ImageBuffer& img, float scale);

}  // namespace plate::detector
