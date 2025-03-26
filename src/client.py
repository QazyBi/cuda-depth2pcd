import requests
import numpy as np
import os
import open3d as o3d
from utils import load_camera_data

def process_camera(camera_dir, server_url="http://localhost:8000"):
    """Process a single camera and visualize the point cloud"""
    # Load camera data
    camera_data = load_camera_data(camera_dir)
    camera_name = camera_data["name"]
    
    print(f"Processing camera: {camera_name}")
    
    # Skip if no depth image or intrinsics
    if camera_data["depth"] is None or camera_data["intrinsics"] is None:
        print(f"Skipping {camera_name} - missing depth or intrinsics")
        return
    
    # Convert numpy arrays to lists for JSON serialization
    request_data = {
        "name": camera_name,
        "depth": camera_data["depth"].tolist(),
        "intrinsics": camera_data["intrinsics"],
        "extrinsics_json": camera_data["extrinsics_json"]
    }
    
    # Add RGB if available
    if camera_data["rgb"] is not None:
        request_data["rgb"] = camera_data["rgb"].tolist()
    
    # Send the request
    print(f"Sending data to {server_url}/process_point_cloud")
    try:
        response = requests.post(
            f"{server_url}/process_point_cloud",
            json=request_data
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return
        
        # Parse the response
        result = response.json()
        
        # Convert lists back to numpy arrays
        points = np.array(result["points"], dtype=np.float32)
        colors = np.array(result.get("colors", []), dtype=np.float32) if "colors" in result else None
        
        print(f"Received point cloud with {len(points)} points")
        
        # Visualize the point cloud
        if len(points) > 0:
            visualize_point_cloud(points, colors, f"Point Cloud - {camera_name}")
        else:
            print("No valid points received")
            
    except Exception as e:
        print(f"Error: {str(e)}")

def visualize_point_cloud(points, colors=None, window_name="Point Cloud"):
    """Visualize a point cloud using Open3D"""
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors if available
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    
    # Set some visualization options
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Path to a single camera directory
    data_dir = "assets/captured_images"
    
    # Get the first camera directory
    camera_dirs = os.listdir(data_dir)
    if not camera_dirs:
        print(f"No camera data found in {data_dir}")
    else:
        # Process the first camera
        camera_dir = os.path.join(data_dir, camera_dirs[0])
        process_camera(camera_dir)
