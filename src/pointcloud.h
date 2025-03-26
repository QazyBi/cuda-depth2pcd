// pointcloud.h
#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#ifdef __cplusplus
extern "C" {
#endif

struct Point3D {
    float x, y, z;
};

__global__ void depthToWorldPCDKernel(const float* depthImage, float* outPCD,
                                       float fx, float fy, float cx, float cy,
                                       const float* extrinsics,
                                       int width, int height);

// Host function declaration updated to use void* for output, matching the definition.
void depthToWorldPCD(const float* depthImage, void* outputCloud,
                     float fx, float fy, float cx, float cy,
                     const float* extrinsics, int width, int height, int gpuID);

#ifdef __cplusplus
}
#endif

#endif // POINTCLOUD_H
