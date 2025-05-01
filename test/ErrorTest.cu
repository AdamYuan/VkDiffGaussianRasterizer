#include "CuTileRasterizer.hpp"
#include "ErrorTest.hpp"
#include "GSModel.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
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

namespace cuperftest {
void WritePixelsPNG(const std::filesystem::path &filename, const float *devicePixels, uint32_t width, uint32_t height) {
	uint32_t pixelCount = width * height;
	std::vector<float> pixels(3 * pixelCount);
	cudaMemcpy(pixels.data(), devicePixels, pixels.size() * sizeof(float), cudaMemcpyDeviceToHost);
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
		    float2byte(pixels[i]),
		    float2byte(pixels[i + pixelCount]),
		    float2byte(pixels[i + pixelCount * 2]),
		};
	}

	stbi_write_png(filename.string().c_str(), (int)width, (int)height, 3, bytes.data(), (int)width * 3);
}
void RandomPixels(float *devicePixels, uint32_t width, uint32_t height) {
	uint32_t pixelCount = width * height;
	std::vector<float> pixels(3 * pixelCount);
	std::mt19937 randGen{0};
	for (float &pixel : pixels) {
		pixel = std::uniform_real_distribution<float>{}(randGen);
	}
	cudaMemcpy(devicePixels, pixels.data(), pixels.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaCheckError();
}
void ClearDL_DSplats(const CuTileRasterizer::SplatArgs &splats, uint32_t splatCount) {
	cudaMemset(splats.means, 0, splatCount * sizeof(GSModel::Mean));
	cudaMemset(splats.scales, 0, splatCount * sizeof(GSModel::Scale));
	cudaMemset(splats.rotates, 0, splatCount * sizeof(GSModel::Rotate));
	cudaMemset(splats.opacities, 0, splatCount * sizeof(GSModel::Opacity));
	cudaMemset(splats.shs, 0, splatCount * sizeof(GSModel::SH));
}
void WriteDL_DSplatsJSON(const std::filesystem::path &filename, const CuTileRasterizer::SplatArgs &splats,
                         uint32_t splatCount) {
	std::vector<GSModel::Mean> dL_dMeans(splatCount);
	std::vector<GSModel::Scale> dL_dScales(splatCount);
	std::vector<GSModel::Opacity> dL_dOpacities(splatCount);
	std::vector<GSModel::Rotate> dL_dRotates(splatCount);

	cudaMemcpy(dL_dMeans.data(), splats.means, splatCount * sizeof(GSModel::Mean), cudaMemcpyDeviceToHost);
	cudaMemcpy(dL_dScales.data(), splats.scales, splatCount * sizeof(GSModel::Scale), cudaMemcpyDeviceToHost);
	cudaMemcpy(dL_dOpacities.data(), splats.opacities, splatCount * sizeof(GSModel::Opacity), cudaMemcpyDeviceToHost);
	cudaMemcpy(dL_dRotates.data(), splats.rotates, splatCount * sizeof(GSModel::Rotate), cudaMemcpyDeviceToHost);

	nlohmann::json json;
	json["dL_dMeans"] = dL_dMeans;
	json["dL_dScales"] = dL_dScales;
	json["dL_dOpacities"] = dL_dOpacities;
	json["dL_dRotates"] = dL_dRotates;

	std::ofstream fout{filename};
	fout << json.dump(4, ' ') << std::endl;
}
} // namespace cuperftest

void GSGradient::Update(const CuTileRasterizer::SplatArgs &splats, uint32_t splatCount) {
	if (this->splatCount != splatCount) {
		this->splatCount = splatCount;

		means.resize(splatCount);
		scales.resize(splatCount);
		rotates.resize(splatCount);
		opacities.resize(splatCount);
		sh0s.resize(splatCount);
	}

	static std::vector<GSModel::SH> shs;
	shs.resize(splatCount);

	cudaMemcpy(means.data(), splats.means, splatCount * sizeof(Mean), cudaMemcpyDeviceToHost);
	cudaMemcpy(scales.data(), splats.scales, splatCount * sizeof(Scale), cudaMemcpyDeviceToHost);
	cudaMemcpy(opacities.data(), splats.opacities, splatCount * sizeof(Opacity), cudaMemcpyDeviceToHost);
	cudaMemcpy(rotates.data(), splats.rotates, splatCount * sizeof(Rotate), cudaMemcpyDeviceToHost);
	cudaMemcpy(shs.data(), splats.shs, splatCount * sizeof(GSModel::SH), cudaMemcpyDeviceToHost);
	for (uint32_t i = 0; i < splatCount; ++i)
		sh0s[i] = shs[i][0];
}
