#pragma once

#include <cstdint>

#include "plate/core/bounding_box.hpp"
#include "plate/image/image_buffer.hpp"

namespace plate::draw {

struct BgrColor {
  std::uint8_t blue{};
  std::uint8_t green{};
  std::uint8_t red{};
};

void draw_rectangle(plate::image::ImageBuffer &image,
                    const plate::core::BoundingBox &box,
                    BgrColor color = {0, 0, 255}, int thickness = 3);

}  // namespace plate::draw
