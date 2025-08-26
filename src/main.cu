#include <cuda_runtime.h>
#include <fmt/core.h>
#include <vector>
#include <algorithm>
#include <optional>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static std::optional<std::vector<float4>> loadImage(const char* path, uint32_t* width,
                                     uint32_t* height) {
  int x, y, channels;
  unsigned char* img = stbi_load(path, &x, &y, &channels, 4);

  if (img == nullptr) {
    fmt::println("Failed to load {}", path);
    return std::nullopt;
  }

  std::vector<float4> pixels(x * y);
  for (int i = 0; i < x * y; i++) {
    int base = i * 4;
    pixels[i] = make_float4(img[base + 0] / 255.0f, img[base + 1] / 255.0f,
                            img[base + 2] / 255.0f, img[base + 3] / 255.0f);
  }

  stbi_image_free(img);

  *width = x;
  *height = y;

  return pixels;
}

static void writeImage(const char* path, const std::vector<float4>& data,
                       uint32_t width, uint32_t height) {
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

int main() {
  uint32_t width, height;
  std::optional<std::vector<float4>> hImage = loadImage("input.png", &width, &height);

  if (!hImage) return -1;

  size_t imageByteSize = hImage->size() * sizeof(float4);
  float4* dImage = nullptr;
  cudaMalloc(&dImage, imageByteSize);
  cudaMemcpy(dImage, hImage->data(), imageByteSize, cudaMemcpyHostToDevice);
  fmt::println("Copied {} bytes to device", imageByteSize);

  // Do some processing on the GPU

  cudaMemcpy(hImage->data(), dImage, imageByteSize, cudaMemcpyDeviceToHost);
  writeImage("output.png", *hImage, width, height);
  fmt::println("Wrote to output.png");

  cudaFree(dImage);
}