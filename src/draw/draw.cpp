#include "plate/draw/draw.hpp"

#include <algorithm>
#include <stdexcept>

#include "plate/core/bounding_box.hpp"

namespace plate::draw {
namespace {

void paint_pixel(image::ImageBuffer &image, const int x, const int y,
                 const BgrColor color) {
  auto *pixel = &image.pixels[static_cast<std::size_t>(y) * image.row_stride +
                              static_cast<std::size_t>(x) * 3];
  pixel[0] = color.blue;
  pixel[1] = color.green;
  pixel[2] = color.red;
}

void draw_horizontal_line(image::ImageBuffer &image, const int y, const int x0,
                          const int x1, const BgrColor color) {
  if (y < 0 || y >= image.height) {
    return;
  }

  const int start = std::max(0, x0);
  const int end = std::min(image.width - 1, x1);
  for (int x = start; x <= end; ++x) {
    for (int off{-5}; off <=5; ++off ) {
       if (y+off>=0 && y+off<image.height)
       paint_pixel(image, x, y+off, color);
    }
  }
}

void draw_vertical_line(image::ImageBuffer &image, const int x, const int y0,
                        const int y1, const BgrColor color) {
  if (x < 0 || x >= image.width) {
    return;
  }

  const int start = std::max(0, y0);
  const int end = std::min(image.height - 1, y1);
  for (int y = start; y <= end; ++y) {
    paint_pixel(image, x, y, color);
  }
}

}  // namespace

void draw_rectangle(image::ImageBuffer &image, const core::BoundingBox &box,
                    const BgrColor color, const int thickness) {
  if (image.channels != 3) {
    throw std::invalid_argument(
        "Rectangle drawing expects a 3-channel color image.");
  }

  const auto clamped = core::clamp_to_image(box, image.width, image.height);
  if (clamped.empty()) {
    return;
  }

  const int line_width = std::max(1, thickness);
  for (int offset = 0; offset < line_width; ++offset) {
    draw_horizontal_line(image, clamped.y + offset, clamped.x,
                         clamped.x + clamped.width - 1, color);
    draw_horizontal_line(image, clamped.y + clamped.height - 1 - offset,
                         clamped.x, clamped.x + clamped.width - 1, color);
    draw_vertical_line(image, clamped.x + offset, clamped.y,
                       clamped.y + clamped.height - 1, color);
    draw_vertical_line(image, clamped.x + clamped.width - 1 - offset, clamped.y,
                       clamped.y + clamped.height - 1, color);
  }
}

}  // namespace plate::draw
