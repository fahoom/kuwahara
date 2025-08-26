#include <cuda_runtime.h>
#include <fmt/core.h>

#include <algorithm>
#include <optional>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef uint32_t u32;
typedef int32_t i32;
typedef size_t usize;


static std::optional<std::vector<float4>> loadImage(const char* path,
                                                    u32* width, u32* height);
static void writeImage(const char* path, const std::vector<float4>& data,
                       u32 width, u32 height);

__global__ void kuwuhara_filtering(const float4* input, float4* output,
                                   u32 width, u32 height, i32 radius);

__global__ void generalized_kuwuhara_filtering(const float4* input, float4* output,
                                   u32 width, u32 height, i32 radius, i32 N);

__global__ void rgb_to_grayscale(const float4* input, float* gray, u32 width,
                                 u32 height);

__global__ void sobel_gradients(const float* gray, float2* gradients, u32 width,
                                u32 height);

__global__ void compute_orientation(const float2* gradients, float* orientation,
                                    u32 width, u32 height);

__global__ void anisotropic_kuwahara(const float4* input, float4* output,
                                     const float* orientation, u32 width,
                                     u32 height, i32 radius, u32 N, float a,
                                     float b);
int main(int argc, char** argv) {
  u32 width, height;
  std::optional<std::vector<float4>> hImage =
      loadImage("input.png", &width, &height);

  if (!hImage) {
    fmt::println("Failed to load input.png");
    return -1;
  }

  // Load image data into device memory
  usize imageByteSize = hImage->size() * sizeof(float4);
  float4* dImage = nullptr;
  cudaMalloc(&dImage, imageByteSize);
  cudaMemcpy(dImage, hImage->data(), imageByteSize, cudaMemcpyHostToDevice);
  fmt::println("Copied {} bytes to device", imageByteSize);

  // Allocate the buffers the kernels will use
  float4* dOutput = nullptr;
  cudaMalloc(&dOutput, imageByteSize);

  float* dGray = nullptr;
  cudaMalloc(&dGray, width * height * sizeof(float));

  float2* dSobel = nullptr;
  cudaMalloc(&dSobel, width * height * sizeof(float2));

  float* dOrientation = nullptr;
  cudaMalloc(&dOrientation, width * height * sizeof(float));

  // Define launch parameters
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // Generate the greyscale image
  fmt::println("Launching greyscale kernel");
  rgb_to_grayscale<<<grid, block>>>(dImage, dGray, width, height);

  // Calculate sobel gradients
  fmt::println("Launching sobel kernel");
  sobel_gradients<<<grid, block>>>(dGray, dSobel, width, height);

  // Calculate orientation of pixels
  fmt::println("Launching orientation kernel");
  compute_orientation<<<grid, block>>>(dSobel, dOrientation, width, height);

  // Generate kuwahara filtered image
  fmt::println("Launching kuwahara kernel");
  anisotropic_kuwahara<<<grid, block>>>(dImage, dOutput, dOrientation, width, height, 10, 8, 10.0,
                       10.0);
  cudaDeviceSynchronize();
  fmt::println("Calculation finished!");

  // Read final image data from device memory, and write to file
  cudaMemcpy(hImage->data(), dOutput, imageByteSize, cudaMemcpyDeviceToHost);
  writeImage("output.png", *hImage, width, height);
  fmt::println("Wrote to output.png");

  cudaFree(dImage);
}

static std::optional<std::vector<float4>> loadImage(const char* path,
                                                    u32* width, u32* height) {
  i32 x, y, channels;
  unsigned char* img = stbi_load(path, &x, &y, &channels, 4);

  if (img == nullptr) {
    fmt::println("Failed to load {}", path);
    return std::nullopt;
  }

  std::vector<float4> pixels(x * y);
  for (int i = 0; i < x * y; i++) {
    i32 base = i * 4;
    pixels[i] = make_float4(img[base + 0] / 255.0f, img[base + 1] / 255.0f,
                            img[base + 2] / 255.0f, img[base + 3] / 255.0f);
  }

  stbi_image_free(img);

  *width = x;
  *height = y;

  return pixels;
}

static void writeImage(const char* path, const std::vector<float4>& data,
                       u32 width, u32 height) {
  std::vector<unsigned char> out(width * height * 4);
  for (int i = 0; i < width * height; i++) {
    out[i * 4 + 0] = static_cast<unsigned char>(
        std::clamp(data[i].x * 255.0f, 0.0f, 255.0f));
    out[i * 4 + 1] = static_cast<unsigned char>(
        std::clamp(data[i].y * 255.0f, 0.0f, 255.0f));
    out[i * 4 + 2] = static_cast<unsigned char>(
        std::clamp(data[i].z * 255.0f, 0.0f, 255.0f));
    out[i * 4 + 3] = static_cast<unsigned char>(
        std::clamp(data[i].w * 255.0f, 0.0f, 255.0f));
  }

  stbi_write_png(path, width, height, 4, out.data(), width * 4);
}