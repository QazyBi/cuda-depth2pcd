import numpy as np
import ctypes
import open3d as o3d
import matplotlib.pyplot as plt
import os
import json
import glob
import matplotlib
matplotlib.use('Agg')  # For saving to file without display

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

def load_camera_data(camera_dir):
    """Load camera data from the specified directory"""
    camera_name = os.path.basename(camera_dir)
    
    # Load RGB image
    rgb_path = os.path.join(camera_dir, f"{camera_name}_rgb.png")
    if os.path.exists(rgb_path):
        rgb_img = plt.imread(rgb_path)
        # Convert from 0-1 to 0-255 if needed
        if rgb_img.dtype == np.float32 and rgb_img.max() <= 1.0:
            rgb_img = (rgb_img * 255).astype(np.uint8)
    else:
        print(f"Warning: RGB image not found at {rgb_path}")
        rgb_img = None
    
    # Load depth image
    depth_path = os.path.join(camera_dir, f"{camera_name}_depth.npy")
    if os.path.exists(depth_path):
        depth_img = np.load(depth_path) / 1000.0
    else:
        print(f"Warning: Depth image not found at {depth_path}")
        depth_img = None
    
    # Load intrinsics
    intrinsics_path = os.path.join(camera_dir, f"{camera_name}_intrinsics.json")
    if os.path.exists(intrinsics_path):
        with open(intrinsics_path, 'r') as f:
            intrinsics = json.load(f)
    else:
        print(f"Warning: Intrinsics not found at {intrinsics_path}")
        intrinsics = None
    
    # Load extrinsics
    extrinsics_path = os.path.join(camera_dir, f"{camera_name}_extrinsics.json")
    if os.path.exists(extrinsics_path):
        with open(extrinsics_path, 'r') as f:
            extrinsics_json = json.load(f)
    else:
        print(f"Warning: Extrinsics not found at {extrinsics_path}")
        extrinsics_json = None
    
    return {
        "name": camera_name,
        "rgb": rgb_img,
        "depth": depth_img,
        "intrinsics": intrinsics,
        "extrinsics_json": extrinsics_json
    }

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    x, y, z, w = q['x'], q['y'], q['z'], q['w']
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def create_extrinsics_matrix(extrinsics_json):
    """Create 4x4 extrinsics matrix from JSON data"""
    if extrinsics_json is None:
        # Return identity matrix if no extrinsics data
        return np.eye(4, dtype=np.float32)
    
    # Extract translation
    tx = extrinsics_json['translation']['x']
    ty = extrinsics_json['translation']['y']
    tz = extrinsics_json['translation']['z']
    
    # Extract rotation (quaternion)
    rotation = extrinsics_json['rotation']
    R = quaternion_to_rotation_matrix(rotation)
    
    # Create 4x4 transformation matrix
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = [tx, ty, tz]
    
    return extrinsics

def depth_to_point_cloud(depth_img, fx, fy, cx, cy, extrinsics, gpu_id=0):
    """Convert depth image to point cloud using CUDA"""
    if depth_img is None:
        print("Error: No depth image provided")
        return np.array([])
    
    height, width = depth_img.shape
    
    # Flatten the depth image
    depth_flat = depth_img.flatten().astype(np.float32)
    
    # Prepare the output array
    point_cloud = np.zeros(width * height * 3, dtype=np.float32)
    
    # Convert numpy arrays to ctypes pointers
    depth_ptr = depth_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cloud_ptr = point_cloud.ctypes.data_as(ctypes.c_void_p)
    extrinsics_ptr = extrinsics.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the CUDA function
    cuda_lib.depthToWorldPCD(
        depth_ptr, cloud_ptr, 
        fx, fy, cx, cy, 
        extrinsics_ptr, 
        width, height, gpu_id
    )
    
    # Reshape the output to a point cloud
    points = point_cloud.reshape(-1, 3)
    
    # Filter out invalid points (zeros or very far points)
    valid_mask = ~np.any(np.isnan(points), axis=1)
    valid_mask &= ~np.any(np.isinf(points), axis=1)
    valid_mask &= np.linalg.norm(points, axis=1) < 10.0  # Filter points too far away
    
    # Return both points and the valid mask for color mapping
    return points[valid_mask], valid_mask

def visualize_point_cloud(points, rgb=None):
    """Visualize point cloud using Open3D"""
    if len(points) == 0:
        print("Error: Empty point cloud")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors if available
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    # Add a coordinate frame for reference
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, frame])

def visualize_depth_image(depth_img, title="Depth Image"):
    """Visualize depth image using matplotlib"""
    if depth_img is None:
        print("Error: No depth image provided")
        return
    
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_img, cmap='viridis')
    plt.colorbar(label='Depth (m)')
    plt.title(title)
    # Option 1: Save the figure to a file instead of showing it
    plt.savefig('output_figure.png')
    # plt.show()  # Comment out or remove this line

def benchmark_depth_to_point_cloud(depth_img, fx, fy, cx, cy, extrinsics, gpu_id=0, num_iterations=100):
    """Benchmark the depth to point cloud conversion using CUDA"""
    if depth_img is None:
        print("Error: No depth image provided")
        return None
    
    height, width = depth_img.shape
    
    # Flatten the depth image
    depth_flat = depth_img.flatten().astype(np.float32)
    
    # Prepare the output array
    point_cloud = np.zeros(width * height * 3, dtype=np.float32)
    
    # Convert numpy arrays to ctypes pointers
    depth_ptr = depth_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    cloud_ptr = point_cloud.ctypes.data_as(ctypes.c_void_p)
    extrinsics_ptr = extrinsics.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Create CUDA events for timing
    start_event = ctypes.c_void_p()
    stop_event = ctypes.c_void_p()
    
    # Define the function signatures for CUDA event operations
    cuda_lib.createEvent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda_lib.recordEvent.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cuda_lib.synchronizeEvent.argtypes = [ctypes.c_void_p]
    cuda_lib.getElapsedTime.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    cuda_lib.destroyEvent.argtypes = [ctypes.c_void_p]
    
    # Create events
    cuda_lib.createEvent(ctypes.byref(start_event))
    cuda_lib.createEvent(ctypes.byref(stop_event))
    
    # Warm-up run
    cuda_lib.depthToWorldPCD(
        depth_ptr, cloud_ptr, 
        fx, fy, cx, cy, 
        extrinsics_ptr, 
        width, height, gpu_id
    )
    
    # Benchmark runs
    elapsed_time = ctypes.c_float(0)
    total_time = 0.0
    
    for i in range(num_iterations):
        # Record start event
        cuda_lib.recordEvent(start_event, 0)
        
        # Call the CUDA function
        cuda_lib.depthToWorldPCD(
            depth_ptr, cloud_ptr, 
            fx, fy, cx, cy, 
            extrinsics_ptr, 
            width, height, gpu_id
        )
        
        # Record stop event
        cuda_lib.recordEvent(stop_event, 0)
        
        # Synchronize to ensure completion
        cuda_lib.synchronizeEvent(stop_event)
        
        # Calculate elapsed time
        cuda_lib.getElapsedTime(start_event, stop_event, ctypes.byref(elapsed_time))
        total_time += elapsed_time.value
    
    # Clean up events
    cuda_lib.destroyEvent(start_event)
    cuda_lib.destroyEvent(stop_event)
    
    # Calculate statistics
    avg_time = total_time / num_iterations
    throughput = (width * height) / (avg_time / 1000.0)  # points per second
    
    # Reshape the output to a point cloud for the last run
    points = point_cloud.reshape(-1, 3)
    
    # Filter out invalid points (zeros or very far points)
    valid_mask = ~np.any(np.isnan(points), axis=1)
    valid_mask &= ~np.any(np.isinf(points), axis=1)
    valid_mask &= np.linalg.norm(points, axis=1) < 10.0  # Filter points too far away
    
    valid_points = points[valid_mask]
    
    # Return benchmark results and the point cloud
    return {
        "avg_latency_ms": avg_time,
        "throughput_points_per_sec": throughput,
        "num_valid_points": len(valid_points),
        "points": valid_points,
        "valid_mask": valid_mask
    }

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
    
    # Create a directory for benchmark results
    benchmark_dir = "benchmark_results"
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Open a file to write benchmark results
    benchmark_file = os.path.join(benchmark_dir, "benchmark_results.csv")
    with open(benchmark_file, 'w') as f:
        f.write("camera_name,avg_latency_ms,throughput_points_per_sec,num_valid_points,image_width,image_height\n")
    
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
        
        # Write benchmark results to file
        with open(benchmark_file, 'a') as f:
            f.write(f"{camera_name},{benchmark_results['avg_latency_ms']:.3f},{benchmark_results['throughput_points_per_sec']:.2f},{benchmark_results['num_valid_points']},{camera_data['depth'].shape[1]},{camera_data['depth'].shape[0]}\n")
        
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
    
    # Generate benchmark plots
    print("Generating benchmark plots...")
    generate_benchmark_plots(benchmark_file)

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


def generate_benchmark_plots(benchmark_file):
    """Generate plots from benchmark results"""
    import pandas as pd
    
    # Read benchmark results
    df = pd.read_csv(benchmark_file)
    
    # Plot latency
    plt.figure(figsize=(10, 6))
    plt.bar(df['camera_name'], df['avg_latency_ms'])
    plt.xlabel('Camera')
    plt.ylabel('Average Latency (ms)')
    plt.title('Kernel Latency by Camera')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('benchmark_results/latency.png')
    
    # Plot throughput
    plt.figure(figsize=(10, 6))
    plt.bar(df['camera_name'], df['throughput_points_per_sec'] / 1e6)  # Convert to millions
    plt.xlabel('Camera')
    plt.ylabel('Throughput (million points/sec)')
    plt.title('Kernel Throughput by Camera')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('benchmark_results/throughput.png')
    
    # Plot valid points percentage
    plt.figure(figsize=(10, 6))
    df['total_points'] = df['image_width'] * df['image_height']
    df['valid_percentage'] = 100 * df['num_valid_points'] / df['total_points']
    plt.bar(df['camera_name'], df['valid_percentage'])
    plt.xlabel('Camera')
    plt.ylabel('Valid Points (%)')
    plt.title('Percentage of Valid Points by Camera')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('benchmark_results/valid_points.png')

if __name__ == "__main__":
    main()
