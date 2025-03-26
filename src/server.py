import numpy as np
import ctypes
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from utils import create_extrinsics_matrix

# Load the CUDA library
cuda_lib = ctypes.CDLL("./build/libpointcloud.so")

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

app = FastAPI(title="Point Cloud Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class CameraData(BaseModel):
    name: str
    depth: List[List[float]]  # 2D array of depth values
    rgb: Optional[List[List[List[int]]]] = None  # 3D array of RGB values
    intrinsics: Dict[str, Any]
    extrinsics_json: Dict[str, Any]

# Define response model
class PointCloudResult(BaseModel):
    points: List[List[float]]  # Nx3 array of points
    colors: Optional[List[List[float]]] = None  # Nx3 array of colors

def depth_to_point_cloud(depth, fx, fy, cx, cy, extrinsics, gpu_id=0):
    """Convert depth image to point cloud using CUDA library"""
    height, width = depth.shape
    
    # Flatten depth image and ensure it's in float32 format
    depth_flat = depth.flatten().astype(np.float32)
    
    # Allocate memory for the output point cloud (3 coordinates per pixel)
    max_points = width * height
    points_flat = np.zeros(max_points * 3, dtype=np.float32)
    
    # Convert extrinsics to float32
    extrinsics_flat = extrinsics.flatten().astype(np.float32)
    
    # Call CUDA function
    cuda_lib.depthToWorldPCD(
        depth_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        points_flat.ctypes.data_as(ctypes.c_void_p),
        fx, fy, cx, cy,
        extrinsics_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        width, height, gpu_id
    )
    
    # Reshape points to Nx3 array
    points = points_flat.reshape(-1, 3)
    
    # Filter out invalid points (those with all zeros)
    valid_mask = ~np.all(points == 0, axis=1)
    valid_points = points[valid_mask]
    
    return valid_points, valid_mask

@app.post("/process_point_cloud", response_model=PointCloudResult)
async def process_point_cloud(camera_data: CameraData = Body(...)):
    """Process camera data and generate point cloud"""
    # Extract camera parameters from intrinsics
    K = np.array(camera_data.intrinsics["K"]).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Create extrinsics matrix
    extrinsics = create_extrinsics_matrix(camera_data.extrinsics_json)
    
    # Convert depth to numpy array
    depth = np.array(camera_data.depth, dtype=np.float32)
    
    # Generate point cloud
    try:
        points, valid_mask = depth_to_point_cloud(depth, fx, fy, cx, cy, extrinsics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Point cloud generation failed: {str(e)}")
    
    # Prepare RGB colors if available
    colors = None
    if camera_data.rgb is not None:
        # Convert RGB to numpy array
        rgb = np.array(camera_data.rgb, dtype=np.uint8)
        # Flatten RGB image and select only valid points
        rgb_flat = rgb.reshape(-1, 3)
        if len(rgb_flat) == len(valid_mask):
            colors = rgb_flat[valid_mask] / 255.0  # Normalize to [0,1]
    
    # Return points and colors as lists
    result = {"points": points.tolist()}
    if colors is not None:
        result["colors"] = colors.tolist()
    
    return result

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
