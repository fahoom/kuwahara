#include <cuda_runtime.h>
#include <fmt/core.h>

#include <algorithm>
#include <optional>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "utils.h"

static std::optional<std::vector<float4>> loadImage(const char* path,
                                                    u32* width, u32* height);
static void writeImage(const char* path, const std::vector<float4>& data,
                       u32 width, u32 height);

__global__ void kuwuhara_filtering(const float4* input, float4* output,
                                   u32 width, u32 height, i32 radius);

__global__ void generalized_kuwuhara_filtering(const float4* input, float4* output,
                                   u32 width, u32 height, i32 radius, i32 N);


int main() {
  u32 width, height;
  std::optional<std::vector<float4>> hImage =
      loadImage("input.png", &width, &height);

  if (!hImage) return -1;

  // Load image data into device memory
  usize imageByteSize = hImage->size() * sizeof(float4);
  float4* dImage = nullptr;
  cudaMalloc(&dImage, imageByteSize);
  cudaMemcpy(dImage, hImage->data(), imageByteSize, cudaMemcpyHostToDevice);
  fmt::println("Copied {} bytes to device", imageByteSize);

  // Allocate the buffers the kernels will use
  float4* dOutput = nullptr;
  cudaMalloc(&dOutput, imageByteSize);

  // Define launch parameters
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  fmt::println("Launching kuwahara kernel");
  generalized_kuwuhara_filtering<<<grid, block>>>(dImage, dOutput, width, height, 10, 8);
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