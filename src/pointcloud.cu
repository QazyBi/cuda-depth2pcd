/**
 * pointcloud.cu
 *
 * Implements CUDA kernel to convert depth images to world-coordinate point clouds.
 * Each thread handles one pixel. The camera intrinsics/extrinsics are assumed known.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include "pointcloud.h"

// Error checking macro
#define CUDA_CHECK(call)                                 \
{                                                        \
    cudaError_t err = call;                              \
    if(err != cudaSuccess) {                             \
        printf("CUDA Error: %s (err_num=%d)\\n",         \
                cudaGetErrorString(err), err);           \
    }                                                    \
}

// Optimized kernel where each thread processes multiple pixels
__global__
void depthToWorldPCDKernel(const float* depth, float* outPCD,
                           float fx, float fy, float cx, float cy,
                           const float* extrinsics,
                           int width, int height) 
{
    // Calculate thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate number of pixels per thread
    int totalPixels = width * height;
    int pixelsPerThread = (totalPixels + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    
    // Process multiple pixels per thread
    for (int p = 0; p < pixelsPerThread; p++) {
        int idx = tid * pixelsPerThread + p;
        
        // Skip if this thread doesn't need to process this pixel
        if (idx >= totalPixels) continue;
        
        int u = idx % width;
        int v = idx / width;
        
        float d = depth[idx];
        
        // Camera coordinates
        float Xc = (u - cx) * d / fx;
        float Yc = (v - cy) * d / fy;
        float Zc = d;
        
        // Apply extrinsics (4x4)
        // [ Xw Yw Zw 1 ]^T = extrinsics_4x4 * [Xc Yc Zc 1]^T
        float Xw = extrinsics[0] * Xc + extrinsics[1] * Yc + extrinsics[2] * Zc  + extrinsics[3];
        float Yw = extrinsics[4] * Xc + extrinsics[5] * Yc + extrinsics[6] * Zc  + extrinsics[7];
        float Zw = extrinsics[8] * Xc + extrinsics[9] * Yc + extrinsics[10]* Zc + extrinsics[11];
        float W  = extrinsics[12]* Xc + extrinsics[13]* Yc + extrinsics[14]* Zc + extrinsics[15];
        
        // Avoid dividing by zero
        if (W != 0.f) {
            Xw /= W;
            Yw /= W;
            Zw /= W;
        }
        
        // Write to output array
        // Each point = 3 floats, so index offset is idx * 3
        outPCD[idx * 3 + 0] = Xw;
        outPCD[idx * 3 + 1] = Yw;
        outPCD[idx * 3 + 2] = Zw;
    }
}

// Host function to be called from C++ code
extern "C"
void depthToWorldPCD(const float* inDepth, void* outPointCloud,
                     float fx, float fy, float cx, float cy,
                     const float* extrinsics,
                     int width, int height, int gpuID)
{
    cudaSetDevice(gpuID);

    size_t numPixels = width * height;

    // Allocate device memory
    float* d_depth = nullptr;
    float* d_outPCD = nullptr;
    CUDA_CHECK(cudaMalloc(&d_depth, numPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outPCD, numPixels * 3 * sizeof(float)));

    // Copy depth image to device
    CUDA_CHECK(cudaMemcpy(d_depth, inDepth, numPixels * sizeof(float), cudaMemcpyHostToDevice));

    // Copy extrinsics
    float* d_extrinsics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_extrinsics, 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_extrinsics, extrinsics, 16 * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel with fewer threads but each thread processes more pixels
    int blockSize = 256;
    int numThreads = min(65535 * blockSize, (int)numPixels); // Limit total threads
    int gridSize = (numThreads + blockSize - 1) / blockSize;
    
    depthToWorldPCDKernel<<<gridSize, blockSize>>>(
        d_depth, d_outPCD, fx, fy, cx, cy, d_extrinsics, width, height
    );
    cudaDeviceSynchronize();

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(outPointCloud, d_outPCD,
                          numPixels * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_outPCD));
    CUDA_CHECK(cudaFree(d_extrinsics));
}

extern "C" {
    // CUDA event functions for benchmarking
    void createEvent(cudaEvent_t* event) {
        cudaEventCreate(event);
    }
    
    void recordEvent(cudaEvent_t event, int stream) {
        cudaEventRecord(event, 0);  // Using default stream (0)
    }
    
    void synchronizeEvent(cudaEvent_t event) {
        cudaEventSynchronize(event);
    }
    
    void getElapsedTime(cudaEvent_t start, cudaEvent_t stop, float* milliseconds) {
        cudaEventElapsedTime(milliseconds, start, stop);
    }
    
    void destroyEvent(cudaEvent_t event) {
        cudaEventDestroy(event);
    }
}
