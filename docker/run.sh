#!/bin/bash
# run_container.sh

# Get the absolute path of the current directory
HOST_DIR=$(pwd)

# Create .Xauthority file if it doesn't exist
touch ~/.Xauthority

# Allow connections from any host to X server (careful with this in production)
xhost +local:docker

# Run the container with X11 forwarding
docker run --gpus all \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -v "$HOST_DIR":/workspace \
    -w /workspace \
    --network=host \
    -it pointcloud_fusion_env
