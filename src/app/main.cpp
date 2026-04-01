#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string_view>

#include "plate/detector/license_plate_detector.hpp"
#include "plate/draw/draw.hpp"
#include "plate/image/image_io.hpp"

namespace {

void print_usage(const char *program_name) {
  std::cerr << "Usage: " << program_name
            << " [--debug|-d] <input.(png|jpg|jpeg)>\n";
}

struct CommandLineOptions {
  bool debug_enabled{false};
  std::filesystem::path input_path;
};

CommandLineOptions parse_command_line(const int argc, char **argv) {
  CommandLineOptions options;
  std::optional<std::filesystem::path> input_path;

  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    const std::string_view argument{argv[argument_index]};
    if (argument == "--debug" || argument == "-d") {
      options.debug_enabled = true;
      continue;
    }

    if (input_path.has_value()) {
      throw std::invalid_argument("Only one input image path can be provided.");
    }

    input_path = std::filesystem::path(argv[argument_index]);
  }

  if (!input_path.has_value()) {
    throw std::invalid_argument("Missing input image path.");
  }

  options.input_path = *input_path;
  return options;
}

}  // namespace

int main(const int argc, char **argv) {
  try {
    const auto options = parse_command_line(argc, argv);
    const auto &input_path = options.input_path;
    auto color = plate::image::load_image(input_path,
                                          plate::image::ImageReadMode::kColor);

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
        std::cout << "Wrote debug stage [" << stage_image.width << ", " << stage_image.height << "] " << stage_output_path << std::endl;
      };
    }

    for (const auto& box: plate::detector::detect_license_plate(color, stage_callback)) {
      plate::draw::draw_rectangle(color, box);
      std::cout << "Detected candidate: x=" << box.x
                << " y=" << box.y << " w=" << box.width
                << " h=" << box.height 
                << '\n';
    } 

    const auto output_path = plate::image::make_output_path(input_path);
    plate::image::save_image(output_path, color);
    std::cout << "Wrote " << output_path << '\n';
  } catch (const std::invalid_argument &error) {
    print_usage(argv[0]);
    std::cerr << "plate: " << error.what() << '\n';
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "plate: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
