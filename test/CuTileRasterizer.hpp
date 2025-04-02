//
// Created by adamyuan on 4/1/25.
//

#pragma once
#ifndef CUTILERASTERIZER_HPP
#define CUTILERASTERIZER_HPP

#include "../src/Camera.hpp"
#include "../src/Rasterizer.hpp"
#include <array>

struct CuTileRasterizer {
	struct Resource {
		struct ResizeableBuffer {
			std::size_t size{};
			char *data{};

			char *update(std::size_t updateSize);
		};
		ResizeableBuffer geometryBuffer{}, binningBuffer{}, imageBuffer{};
	};

	struct SplatArgs {
		uint32_t count{};
		const float *means{};
		const float *scales{};
		const float *rotates{};
		const float *opacities{};
		const float *shs{};
	};

	struct Camera {
		int width{}, height{};
		float tanFovX{}, tanFovY{};
		const float *viewMat{};
		const float *projMat{};
		const float *pos{};

		void Update(const VkGSRaster::Camera &vkCamera);
	};

	struct FwdROArgs {
		Camera camera;
		SplatArgs splats;
		const float *bgColor{};

		void Update(const VkGSRaster::Rasterizer::FwdROArgs &vkROArgs);
	};

	struct FwdRWArgs {
		float *outColor{};

		void Update(const VkGSRaster::Rasterizer::FwdRWArgs &vkRWArgs);
	};

	static void Forward(const FwdROArgs &roArgs, const FwdRWArgs &rwArgs, Resource &resource);
};

#endif
