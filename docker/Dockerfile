# Use NVIDIA's CUDA development image with Ubuntu 20.04 as base
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set non-interactive frontend to suppress prompts during installation.
ENV DEBIAN_FRONTEND=noninteractive
# Set a default time zone (e.g., UTC). Adjust as needed.
ENV TZ=UTC

# Install system dependencies: build tools, MPI, CMake, Python3 and pip
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    openmpi-bin \
    libopenmpi-dev \
    python3 \
    libegl1 \
    libgl1 \
    libgomp1 \
    python3-pip \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages
RUN pip3 install --upgrade pip && \
    pip3 install mpi4py numpy open3d

# Set the working directory in the container
WORKDIR /workspace

# Copy all project files into the container's workspace
COPY . /workspace

# Build the C++/CUDA project using CMake
RUN mkdir -p build && cd build && cmake .. && make

# Default command: start an interactive shell (or adjust to run a default script)
CMD ["bash"]
