#include <exception>
#include <filesystem>
#include <iostream>

#include "plate/detector/license_plate_detector.hpp"
#include "plate/draw/draw.hpp"
#include "plate/image/image_io.hpp"

namespace {

void print_usage(const char *program_name) {
  std::cerr << "Usage: " << program_name << " <input.(png|jpg|jpeg)>\n";
}

}  // namespace

int main(const int argc, char **argv) {
  if (argc != 2) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    const std::filesystem::path input_path{argv[1]};
    auto color = plate::image::load_image(input_path,
                                          plate::image::ImageReadMode::kColor);
    const auto grayscale = plate::image::load_image(
        input_path, plate::image::ImageReadMode::kGrayscale);

    if (const auto detection = plate::detector::detect_license_plate(grayscale);
        detection.found()) {
      plate::draw::draw_rectangle(color, detection.box);
      std::cout << "Detected candidate: x=" << detection.box.x
                << " y=" << detection.box.y << " w=" << detection.box.width
                << " h=" << detection.box.height << " score=" << detection.score
                << '\n';
    } else {
      std::cout << "No plausible plate candidate found.\n";
    }

    const auto output_path = plate::image::make_output_path(input_path);
    plate::image::save_image(output_path, color);
    std::cout << "Wrote " << output_path << '\n';
  } catch (const std::exception &error) {
    std::cerr << "plate: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
