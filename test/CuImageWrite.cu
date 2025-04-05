//
// Created by adamyuan on 4/2/25.
//

#include "CuImageWrite.hpp"

#include <array>
#include <stb_image_write.h>
#include <vector>

#define cudaCheckError() \
	{ \
		cudaError_t e = cudaGetLastError(); \
		if (e != cudaSuccess) { \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(0); \
		} \
	}

void CuImageWrite::Write(const std::filesystem::path &filename, const float *devicePixels, uint32_t width,
                         uint32_t height) {
	uint32_t pixelCount = width * height;
	std::vector<float> colors(3 * pixelCount);
	cudaMemcpy(colors.data(), devicePixels, colors.size() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaCheckError();

	const auto float2byte = [](float x) {
		x = x * 255.0f + 0.5f;
		auto u = (int)std::trunc(x);
		u = std::min(std::max(u, 0), 255);
		return (uint8_t)u;
	};

	std::vector<std::array<uint8_t, 3>> bytes(pixelCount);
	for (uint32_t i = 0; i < pixelCount; ++i) {
		bytes[i] = {
		    float2byte(colors[i]),
		    float2byte(colors[i + pixelCount]),
		    float2byte(colors[i + pixelCount * 2]),
		};
	}

	stbi_write_png(filename.string().c_str(), (int)width, (int)height, 3, bytes.data(), (int)width * 3);
}