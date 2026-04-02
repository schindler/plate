
#include <exception>
#include <filesystem>
#include <iostream>
#include <string_view>

#include "plate/cli/command_line_options.hpp"
#include "plate/detector/license_plate_detector.hpp"
#include "plate/draw/draw.hpp"
#include "plate/image/image_io.hpp"

namespace {

void process_image(const std::filesystem::path &input_path,
                   const plate::cli::CommandLineOptions &options) {
  auto color =
      plate::image::load_image(input_path, plate::image::ImageReadMode::kColor);

  plate::filter::FilterStageCallback stage_callback;
  if (options.debug_enabled) {
    const auto debug_directory = input_path.parent_path() / "debug";
    std::filesystem::create_directories(debug_directory);
    stage_callback = [input_path, debug_directory](
                         const std::string_view stage_name,
                         const plate::image::ImageBuffer &stage_image) {
      const auto stage_output_path = plate::image::make_stage_output_path(
          input_path, debug_directory, stage_name);
      plate::image::save_image(stage_output_path, stage_image);
      std::cout << "Wrote debug stage [" << stage_image.width << ", "
                << stage_image.height << "] " << stage_output_path << std::endl;
    };
  }

  for (const auto &box : plate::detector::detect_license_plate(
           color, options.detector_config, stage_callback)) {
    plate::draw::draw_rectangle(color, box);
    std::cout << input_path << ": detected candidate: x=" << box.x
              << " y=" << box.y << " w=" << box.width << " h=" << box.height
              << '\n';
  }

  const auto output_path = plate::image::make_output_path(input_path);
  plate::image::save_image(output_path, color);
  std::cout << "Wrote " << output_path << '\n';
}

}  // namespace

int main(const int argc, char **argv) {
  try {
    const auto options = plate::cli::parse_command_line(argc, argv);
    int exit_code = 0;

    for (const auto &input_path : options.input_paths) {
      try {
        process_image(input_path, options);
      } catch (const std::exception &error) {
        std::cerr << "plate: " << input_path << ": " << error.what() << '\n';
        exit_code = 1;
      }
    }

    return exit_code;
  } catch (const std::invalid_argument &error) {
    plate::cli::print_usage(argv[0], std::cerr);
    std::cerr << "plate: " << error.what() << '\n';
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "plate: " << error.what() << '\n';
    return 1;
  }
}
