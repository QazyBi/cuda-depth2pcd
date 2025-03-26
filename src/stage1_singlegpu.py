import numpy as np
import ctypes
import open3d as o3d
import matplotlib.pyplot as plt
import os
import json
import glob
import matplotlib
matplotlib.use('Agg')  # For saving to file without display
from utils import create_extrinsics_matrix, visualize_point_cloud, load_camera_data, visualize_depth_image, benchmark_depth_to_point_cloud

# Load the CUDA library
cuda_lib = ctypes.CDLL("./build/libpointcloud.so")  # Adjust path as needed

# Define the function signature
cuda_lib.depthToWorldPCD.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # depth image
    ctypes.c_void_p,                 # output point cloud
    ctypes.c_float,                  # fx
    ctypes.c_float,                  # fy
    ctypes.c_float,                  # cx
    ctypes.c_float,                  # cy
    ctypes.POINTER(ctypes.c_float),  # extrinsics
    ctypes.c_int,                    # width
    ctypes.c_int,                    # height
    ctypes.c_int                     # gpuID
]


def main():
    # Path to the captured images directory
    data_dir = "assets/captured_images"
    
    # Find all camera directories
    camera_dirs = glob.glob(os.path.join(data_dir, "*"))
    
    if not camera_dirs:
        print(f"No camera data found in {data_dir}")
        return

    # Process each camera
    all_points = []
    all_colors = []
    
    for camera_dir in camera_dirs:
        camera_data = load_camera_data(camera_dir)
        camera_name = camera_data["name"]
        
        print(f"Processing camera: {camera_name}")
        
        # Skip if no depth image
        if camera_data["depth"] is None:
            print(f"Skipping {camera_name} - no depth image")
            continue
        
        # Skip if no intrinsics
        if camera_data["intrinsics"] is None:
            print(f"Skipping {camera_name} - no intrinsics")
            continue
        
        # Extract camera parameters from intrinsics
        K = np.array(camera_data["intrinsics"]["K"]).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Create extrinsics matrix
        extrinsics = create_extrinsics_matrix(camera_data["extrinsics_json"])
        
        # Visualize the depth image
        print(f"Visualizing depth image for {camera_name}")
        visualize_depth_image(camera_data["depth"], title=f"Depth Image - {camera_name}")
        
        # Run benchmark
        print(f"Benchmarking depth to point cloud conversion for {camera_name}")
        benchmark_results = benchmark_depth_to_point_cloud(
            camera_data["depth"], fx, fy, cx, cy, extrinsics, gpu_id=0, num_iterations=100
        )
        
        if benchmark_results is None:
            print(f"Warning: Benchmark failed for {camera_name}")
            continue
        
        # Extract benchmark results
        points = benchmark_results["points"]
        valid_mask = benchmark_results["valid_mask"]
        
        # Print benchmark results
        print(f"Benchmark results for {camera_name}:")
        print(f"  Average latency: {benchmark_results['avg_latency_ms']:.3f} ms")
        print(f"  Throughput: {benchmark_results['throughput_points_per_sec']:.2f} points/sec")
        print(f"  Valid points: {benchmark_results['num_valid_points']} out of {len(valid_mask)}")

        # Prepare RGB colors if available
        colors = None
        if camera_data["rgb"] is not None:
            # Flatten RGB image and select only valid points
            rgb_flat = camera_data["rgb"].reshape(-1, 3)
            if len(rgb_flat) == len(valid_mask):
                colors = rgb_flat[valid_mask] / 255.0  # Normalize to [0,1] for Open3D
            else:
                print(f"Warning: RGB shape {camera_data['rgb'].shape} doesn't match depth shape {camera_data['depth'].shape}")

        visualize_point_cloud(points, colors)

        # Add to combined point cloud
        all_points.append(points)
        if colors is not None:
            all_colors.append(colors)
    
    # Combine all point clouds if we have more than one
    if len(all_points) > 1:
        combined_points = np.vstack(all_points)
        combined_colors = None
        if len(all_colors) == len(all_points):  # Make sure we have colors for all point clouds
            combined_colors = np.vstack(all_colors)
        
        print(f"Visualizing combined point cloud from {len(all_points)} cameras")
        visualize_point_cloud(combined_points, combined_colors)
    elif len(all_points) == 0:
        print("No valid point clouds generated")  
    else:
        print("Single point cloud generated")


if __name__ == "__main__":
    main()
