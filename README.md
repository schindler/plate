# Overview

`plate` is a small CUDA/C++ sample project that uses NVIDIA NPP for GPU-side image filtering and a simple classical heuristic to locate a likely license plate region in one or more `.png`, `.jpg`, or `.jpeg` images.

The pipeline is intentionally simple:

1. Load the image on the CPU.
2. Resize the image on the `GPU` to an auto-selected working size around `1080x600`, or use a user-provided `--scale`.
3. Convert to grayscale on the `GPU`.
4. Run Gaussian smoothing and Sobel edge extraction on the `GPU` with NPP.
5. Build the binary candidate mask on the `GPU`.
6. Score connected components on the `GPU` with label propagation and candidate heuristics.
7. Draw a rectangle on the original image and save `<input>-out.<ext>`.

Detailed installation instructions are in `INSTALL`.

## Samples

**Many thanks**

- Car[05,06] <a href="https://unsplash.com/pt-br/@arteum?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Arteum.ro</a> at <a href="https://unsplash.com/pt-br/fotografias/carro-bmw-preto-estacionado-na-estrada-oaMFDtSqHXY?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

- Car[04] <a href="https://unsplash.com/pt-br/@flixxi?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Felix Janßen</a> at <a href="https://unsplash.com/pt-br/fotografias/porsche-911-preto-estacionado-no-chao-branco-2eMJDJM9vxk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
      
      
## Code organization

```text
.
├── bin
├── data
│   └── images
│       └── samples
├── Dockerfile
├── INSTALL
├── include/plate
│   ├── cli
│   ├── core
│   ├── detector
│   ├── draw
│   ├── filter
│   └── image
├── src
│   ├── app
│   ├── cli
│   ├── detector
│   ├── draw
│   ├── filter
│   └── image
└── CMakeLists.txt
```

- `bin/` holds generated executables. After a successful build, the main CLI is written here as `bin/plate`.
- `data/` holds lightweight example data for the repository. Right now it contains sample input images under `data/images/samples/`, including `audi.png`.
- `include/plate/cli/` and `src/cli/` hold command-line parsing and usage handling for the executable.
- `include/plate/` holds public headers grouped by module.
- `src/` holds the implementation files for the application entrypoint and each module.

## Build Summary

The project is designed to compile in Docker on macOS and directly on Linux. The executable is written to `bin/` when you build it.

Quick Linux build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

On Apple Silicon, Docker defaults to `linux/arm64`. If your target runtime machine is a typical x86_64 Linux NVIDIA box, build with `--platform=linux/amd64`.

Quick Docker build:

```bash
docker build --platform=linux/amd64 -t plate-build-env .

docker run --rm \
  --platform=linux/amd64 \
  -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  plate-build-env \
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86

docker run --rm \
  --platform=linux/amd64 \
  -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" \
  -w /workspace \
  plate-build-env \
  cmake --build build --parallel
```

## Run

The sample image lives at `data/images/samples/audi.png`.

Run the program on a Linux machine with a working CUDA runtime:

```bash
./bin/plate data/images/samples/audi.png
```

To process multiple files in one invocation, pass more than one input path:

```bash
./bin/plate data/images/samples/audi.png data/images/samples/car05.jpg
```

Each input is processed independently and writes its own `<input>-out.<ext>` file.

To save the intermediate GPU stages under a `debug/` folder beside the input
image, use:

```bash
./bin/plate --debug data/images/samples/audi.png
```

This writes files such as `debug/audi-grayscale.png`,
`debug/audi-gauss-blur.png`, `debug/audi-sobel-edge.png`,
`debug/audi-binary-mask.png`, and `debug/audi-closed-mask.png`.

You can also tune the detector from the command line:

```bash
./bin/plate \
  --scale 0.50 \
  --plate-ratio 4.2 \
  --min-area 1800 \
  --max-area 24000 \
  data/images/samples/audi.png
```

- `--scale` forces the preprocessing resize factor. If omitted, the tool keeps the current automatic resize calculation.
- `--plate-ratio` and `--expected-plate-ratio` are aliases. They define the expected plate `width / height` ratio used by the CCL scoring stage.
- `--min-area` and `--max-area` control the connected-component area filter in pixels on the resized working image.
- `--debug` saves one image per GPU stage beside each input file inside a `debug/` folder.

## Notes and limitations

- The detector is heuristic-based and tuned for rectangular plates with strong edge contrast.
- The current implementation expects `png`, `jpg`, or `jpeg` inputs.
- Image decoding/encoding is handled by OpenCV inside the `image` module; the filtering path uses CUDA/NPP.
- It is a heuristic demo and is not intended for production-grade license plate recognition.

## Example

![Execution](data/images/about/execution.png)
