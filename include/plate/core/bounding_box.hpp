#pragma once

#include <algorithm>

namespace plate::core {

struct BoundingBox {
  int x{};
  int y{};
  int width{};
  int height{};
  float score;

  [[nodiscard]] bool empty() const noexcept {
    return width <= 0 || height <= 0;
  }
};

[[nodiscard]] inline BoundingBox clamp_to_image(
    const BoundingBox box, const int image_width,
    const int image_height) noexcept {
  if (image_width <= 0 || image_height <= 0) {
    return {};
  }

  const int x0 = std::clamp(box.x, 0, image_width - 1);
  const int y0 = std::clamp(box.y, 0, image_height - 1);
  const int x1 = std::clamp(box.x + box.width, 0, image_width);
  const int y1 = std::clamp(box.y + box.height, 0, image_height);

  return {x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)};
}

}  // namespace plate::core
