#include "plate/cli/command_line_options.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace plate::cli {
namespace {

float parse_positive_float(std::string_view option_name,
                           std::string_view raw_value) {
  const std::string value_string(raw_value);
  char *parse_end = nullptr;
  const float value = std::strtof(value_string.c_str(), &parse_end);
  if (parse_end != value_string.c_str() + value_string.size() ||
      value <= 0.0F) {
    throw std::invalid_argument("Invalid value for " +
                                std::string(option_name) + ": " + value_string);
  }

  return value;
}

int parse_positive_int(std::string_view option_name,
                       std::string_view raw_value) {
  const std::string value_string(raw_value);
  char *parse_end = nullptr;
  const long value = std::strtol(value_string.c_str(), &parse_end, 10);
  if (parse_end != value_string.c_str() + value_string.size() || value <= 0L ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("Invalid value for " +
                                std::string(option_name) + ": " + value_string);
  }

  return static_cast<int>(value);
}

std::string_view read_option_value(const int argc, char **argv,
                                   int *argument_index,
                                   std::string_view option_name) {
  if (*argument_index + 1 >= argc) {
    throw std::invalid_argument("Missing value for " +
                                std::string(option_name) + ".");
  }

  ++(*argument_index);
  return argv[*argument_index];
}

}  // namespace

void print_usage(const char *program_name, std::ostream &output) {
  output << "Usage: " << program_name
         << " [--debug|-d] [--scale FACTOR] [--plate-ratio RATIO]\n"
         << "       [--min-area PIXELS] [--max-area PIXELS]\n"
         << "       <input.(png|jpg|jpeg)> [more-inputs...]\n";
}

CommandLineOptions parse_command_line(const int argc, char **argv) {
  CommandLineOptions options;

  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    const std::string_view argument{argv[argument_index]};
    if (argument == "--debug" || argument == "-d") {
      options.debug_enabled = true;
      continue;
    }

    if (argument == "--scale") {
      options.detector_config.scale = parse_positive_float(
          "--scale", read_option_value(argc, argv, &argument_index, "--scale"));
      continue;
    }

    if (argument == "--plate-ratio" || argument == "--expected-plate-ratio") {
      options.detector_config.expected_plate_ratio = parse_positive_float(
          argument, read_option_value(argc, argv, &argument_index, argument));
      continue;
    }

    if (argument == "--min-area") {
      options.detector_config.min_area = parse_positive_int(
          "--min-area",
          read_option_value(argc, argv, &argument_index, "--min-area"));
      continue;
    }

    if (argument == "--max-area") {
      options.detector_config.max_area = parse_positive_int(
          "--max-area",
          read_option_value(argc, argv, &argument_index, "--max-area"));
      continue;
    }

    if (!argument.empty() && argument.front() == '-') {
      throw std::invalid_argument("Unknown option: " + std::string(argument));
    }

    options.input_paths.emplace_back(argv[argument_index]);
  }

  if (options.input_paths.empty()) {
    throw std::invalid_argument("Missing input image path.");
  }

  if (options.detector_config.min_area > options.detector_config.max_area) {
    throw std::invalid_argument(
        "Minimum area cannot be larger than maximum area.");
  }

  return options;
}

}  // namespace plate::cli
