constexpr float kPI = 3.1415;
typedef uint32_t u32;
typedef int32_t i32;
typedef size_t usize;


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

__global__ void generalized_kuwuhara_filtering(const float4* input, float4* output, u32 width,
                                               u32 height, i32 radius, i32 N) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  // NOTE: MAXIMUM 16 SECTIONS
  float4 mean[16] = {}; 
  float4 variance[16] = {};
  int count[16] = {};

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx == 0 && dy == 0) continue;
      float angle = atan2f(dy, dx);
      if (angle < 0) angle += 2 * kPI;
      int region = int(N * angle / (2 * kPI)) % N;

      int px = min(max(x + dx, 0), width - 1);
      int py = min(max(y + dy, 0), height - 1);
      float4 col = input[py * width + px];
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
  float4 result = {0.0, 0.0, 0.0, 1.0};
  for (int i = 0; i < N; ++i) {
    if (count[i] == 0) continue;
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
      result = {mean[i].x, mean[i].y, mean[i].z, 1.0};
    }
  }

  output[y * width + x] = result;
}

__global__ void rgb_to_grayscale(const float4* input, float* gray, u32 width,
                                 u32 height) {
  u32 x = blockIdx.x * blockDim.x + threadIdx.x;
  u32 y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  u32 idx = y * width + x;
  float4 pix = input[idx];
  gray[idx] = 0.299f * pix.x + 0.587f * pix.y + 0.114f * pix.z;
}

__global__ void sobel_gradients(const float* gray, float2* gradients,
                                u32 width, u32 height) {
  u32 x = blockIdx.x * blockDim.x + threadIdx.x;
  u32 y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1) return;
  u32 idx = y * width + x;

  float gx = -gray[(y - 1) * width + (x - 1)] - 2 * gray[y * width + (x - 1)] -
             gray[(y + 1) * width + (x - 1)] + gray[(y - 1) * width + (x + 1)] +
             2 * gray[y * width + (x + 1)] + gray[(y + 1) * width + (x + 1)];

  float gy = -gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] -
             gray[(y - 1) * width + (x + 1)] + gray[(y + 1) * width + (x - 1)] +
             2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + (x + 1)];

  gradients[idx] = {gx, gy};
}

__global__ void compute_orientation(const float2* gradients,
                                    float* orientation, u32 width, u32 height) {
  u32 x = blockIdx.x * blockDim.x + threadIdx.x;
  u32 y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  u32 idx = y * width + x;

  float gx = gradients[idx].x;
  float gy = gradients[idx].y;
  orientation[idx] = 0.5f * atan2f(2.0f * gx * gy, gx * gx - gy * gy);
}

__global__ void anisotropic_kuwahara(const float4* input, float4* output,
                                     const float* orientation, u32 width,
                                     u32 height, i32 radius, u32 N, float a,
                                     float b) {
  u32 x = blockIdx.x * blockDim.x + threadIdx.x;
  u32 y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float theta = orientation[y * width + x];
  float cos_theta = cosf(theta), sin_theta = sinf(theta);

  float4 mean[16] = {};
  float4 variance[16] = {};
  u32 count[16] = {};

  for (i32 dy = -radius; dy <= radius; ++dy) {
    for (i32 dx = -radius; dx <= radius; ++dx) {
      // Rotate and scale
      float dxr = cos_theta * dx - sin_theta * dy;
      float dyr = sin_theta * dx + cos_theta * dy;
      float ellipse = (dxr * dxr) / (a * a) + (dyr * dyr) / (b * b);
      if (ellipse > 1.0f) continue;

      float angle = atan2f(dyr, dxr);
      if (angle < 0) angle += 2 * kPI;
      i32 region = static_cast<i32>(N * angle / (2 * kPI)) % N;

      i32 px = min(max(x + dx, 0), width - 1);
      i32 py = min(max(y + dy, 0), height - 1);
      float4 col = input[py * width + px];
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
  float4 result = {0.0, 0.0, 0.0, 1.0};
  for (int i = 0; i < N; ++i) {
    if (count[i] == 0) continue;
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
      result = {mean[i].x, mean[i].y, mean[i].z, 1.0};
    }
  }

  output[y * width + x] = result;
}