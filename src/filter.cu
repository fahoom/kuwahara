#include "utils.h"

__global__ void kuwuhara_filtering(const float4* input, float4* output,
                                   u32 width, u32 height, i32 radius) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < radius || y < radius || x >= width - radius || y >= height - radius)
    return;

  float4 mean[4] = {};
  float4 variance[4] = {};
  int count[4] = {0, 0, 0, 0};

  for (i32 dy = -radius; dy <= radius; dy++) {
    for (i32 dx = -radius; dx <= radius; dx++) {
      int region = (dy <= 0 ? 0 : 2) + (dx > 0 ? 1 : 0);
      float4 col = input[(y + dy) * width + (x + dx)];

      mean[region].x += col.x;
      mean[region].y += col.y;
      mean[region].z += col.z;
      mean[region].w += col.w;

      variance[region].x += col.x * col.x;
      variance[region].y += col.y * col.y;
      variance[region].z += col.z * col.z;
      variance[region].w += col.w * col.w;

      count[region]++;
    }
  }

  float minSigma = INFINITY;
  float4 result = { 0.0, 0.0, 0.0, 1.0 };
  for (i32 i = 0; i < 4; i++) {
    mean[i].x /= count[i];
    mean[i].y /= count[i];
    mean[i].z /= count[i];
    mean[i].w /= count[i];

    variance[i].x = fabsf(variance[i].x / count[i] - mean[i].x * mean[i].x);
    variance[i].y = fabsf(variance[i].y / count[i] - mean[i].y * mean[i].y);
    variance[i].z = fabsf(variance[i].z / count[i] - mean[i].z * mean[i].z);
    variance[i].w = fabsf(variance[i].w / count[i] - mean[i].w * mean[i].w);

    float sigma = variance[i].x + variance[i].y + variance[i].z;
    if (sigma < minSigma) {
      minSigma = sigma;
      result = {mean[i].x, mean[i].y, mean[i].z, 1.0 };
    }
  }

  output[y * width + x] = result;
}