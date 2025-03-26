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
    // Cache extrinsics matrix in shared memory for faster access
    __shared__ float sharedExtrinsics[16];
    
    // Have first 16 threads in the block load the extrinsics matrix
    if (threadIdx.x < 16) {
        sharedExtrinsics[threadIdx.x] = extrinsics[threadIdx.x];
    }
    
    // Synchronize to ensure all threads can access the shared memory
    __syncthreads();
    
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
        
        // Apply extrinsics (4x4) using shared memory
        // [ Xw Yw Zw 1 ]^T = extrinsics_4x4 * [Xc Yc Zc 1]^T
        float Xw = sharedExtrinsics[0] * Xc + sharedExtrinsics[1] * Yc + sharedExtrinsics[2] * Zc  + sharedExtrinsics[3];
        float Yw = sharedExtrinsics[4] * Xc + sharedExtrinsics[5] * Yc + sharedExtrinsics[6] * Zc  + sharedExtrinsics[7];
        float Zw = sharedExtrinsics[8] * Xc + sharedExtrinsics[9] * Yc + sharedExtrinsics[10]* Zc + sharedExtrinsics[11];
        float W  = sharedExtrinsics[12]* Xc + sharedExtrinsics[13]* Yc + sharedExtrinsics[14]* Zc + sharedExtrinsics[15];
        
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
    
    // Create CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Use pinned memory or zero-copy for extrinsics (small buffer)
    float* d_extrinsics = nullptr;
    CUDA_CHECK(cudaMalloc(&d_extrinsics, 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_extrinsics, extrinsics, 16 * sizeof(float), 
                              cudaMemcpyHostToDevice, stream));
    
    // Allocate device memory
    float* d_depth = nullptr;
    float* d_outPCD = nullptr;
    CUDA_CHECK(cudaMalloc(&d_depth, numPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outPCD, numPixels * 3 * sizeof(float)));
    
    // Asynchronously copy depth image to device
    CUDA_CHECK(cudaMemcpyAsync(d_depth, inDepth, numPixels * sizeof(float), 
                              cudaMemcpyHostToDevice, stream));
    
    // Launch kernel in the stream
    int blockSize = 256;
    int numThreads = min(65535 * blockSize, (int)numPixels); // Limit total threads
    int gridSize = (numThreads + blockSize - 1) / blockSize;
    
    depthToWorldPCDKernel<<<gridSize, blockSize, 0, stream>>>(
        d_depth, d_outPCD, fx, fy, cx, cy, d_extrinsics, width, height
    );
    
    // Asynchronously copy results back to host
    CUDA_CHECK(cudaMemcpyAsync(outPointCloud, d_outPCD,
                              numPixels * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    
    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_depth));
    CUDA_CHECK(cudaFree(d_outPCD));
    CUDA_CHECK(cudaFree(d_extrinsics));
    CUDA_CHECK(cudaStreamDestroy(stream));
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
