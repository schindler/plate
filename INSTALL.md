plate installation guide
========================

This project builds a native CUDA executable that detects a likely license plate in an image, draws a rectangle around it, and writes a new image beside the input file.

The source tree keeps lightweight example data in `data/` and writes built executables to `bin/`.

Runtime requirement
-------------------

The executable must run on a Linux environment with:

- an NVIDIA GPU
- a compatible NVIDIA driver
- a CUDA runtime compatible with the build

macOS can be used to build the project in Docker, but not to run the CUDA executable.

Linux native install
--------------------

Recommended when you already have a Linux workstation or server with an NVIDIA GPU.

1. Install these dependencies:
   - CUDA Toolkit with NPP
   - OpenCV development packages
   - CMake 3.22 or newer
   - Ninja
   - a C++17-capable compiler
2. Configure the project:

   ```bash
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   ```

3. Build the executable:

   ```bash
   cmake --build build --parallel
   ```

4. The executable will be available in:

   ```text
   bin/plate
   ```

5. Run it with the sample image:

   ```bash
   ./bin/plate data/images/samples/audi.png
   ```

6. To save the intermediate filter stages:

   ```bash
   ./bin/plate --debug data/images/samples/audi.png
   ```

Linux install with Docker
-------------------------

Recommended when you want a reproducible toolchain without installing CUDA and OpenCV directly on the host.

1. Build the toolchain image:

   ```bash
   docker build -t plate-build-env .
   ```

2. Configure the project:

   ```bash
   docker run --rm \
     -u "$(id -u):$(id -g)" \
     -v "$PWD:/workspace" \
     -w /workspace \
     plate-build-env \
     cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
   ```

3. Compile:

   ```bash
   docker run --rm \
     -u "$(id -u):$(id -g)" \
     -v "$PWD:/workspace" \
     -w /workspace \
     plate-build-env \
     cmake --build build --parallel
   ```

4. The executable will be written to:

   ```text
   bin/plate
   ```

5. To run the program, use a Linux environment with NVIDIA GPU access.

6. To save intermediate images for debugging:

   ```bash
   ./bin/plate --debug data/images/samples/audi.png
   ```

macOS install
-------------

Recommended for source-level work when your development machine is a Mac.

Important constraints:

- Docker Desktop on macOS does not provide an NVIDIA GPU to containers.
- You can build on macOS, but you cannot execute the CUDA binary there.
- Run the produced binary later on Linux with an NVIDIA GPU.

Intel Mac:

1. Build the Docker image:

   ```bash
   docker build -t plate-build-env .
   ```

2. Configure and build:

   ```bash
   docker run --rm \
     -u "$(id -u):$(id -g)" \
     -v "$PWD:/workspace" \
     -w /workspace \
     plate-build-env \
     cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86

   docker run --rm \
     -u "$(id -u):$(id -g)" \
     -v "$PWD:/workspace" \
     -w /workspace \
     plate-build-env \
     cmake --build build --parallel
   ```

Apple Silicon Mac:

1. If your target runtime is a standard x86_64 Linux NVIDIA machine, force the build platform:

   ```bash
   docker build --platform=linux/amd64 -t plate-build-env .
   ```

2. Configure and build with the same platform:

   ```bash
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

3. If your target runtime is Linux ARM64 with an NVIDIA GPU, use `linux/arm64` instead.

Windows install
---------------

Recommended path: use WSL2 with Ubuntu and follow the Linux instructions inside WSL.

Why this is the recommended route:

- CUDA on Linux has the most direct compatibility with this project structure.
- The current project was validated in a Linux CUDA container, not with a native Visual Studio workflow.
- WSL2 allows Windows users with supported NVIDIA hardware to keep a Linux-compatible build and run path.

Suggested approach:

1. Install WSL2 and Ubuntu 22.04.
2. Install an NVIDIA Windows driver with WSL CUDA support.
3. Inside Ubuntu, choose one of:
   - the Linux native install steps above
   - the Linux Docker install steps above
