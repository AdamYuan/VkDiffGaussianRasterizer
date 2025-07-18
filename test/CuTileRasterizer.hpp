//
// Created by adamyuan on 4/1/25.
//

#pragma once
#ifndef CUTILERASTERIZER_HPP
#define CUTILERASTERIZER_HPP

#include "../src/Rasterizer.hpp"
#include <array>
#include <rasterizer.h>

struct CuTileRasterizer {
	struct Resource {
		struct ResizeableBuffer {
			std::size_t size{};
			char *data{};

			char *Update(std::size_t updateSize);
		};
		ResizeableBuffer geometryBuffer{}, binningBuffer{}, imageBuffer{}, sampleBuffer{}, dLBuffer{};

		int numRendered{};
		int wtfIsThis{};
	};

	using PerfQuery = CudaRasterizer::Rasterizer::PerfQuery;
	using PerfMetrics = CudaRasterizer::Rasterizer::PerfMetrics;

	struct SplatArgs {
		float *means{};
		float *scales{};
		float *rotates{};
		float *opacities{};
		float *shs{};
	};

	struct CameraArgs {
		int width{}, height{};
		float tanFovX{}, tanFovY{};
		const float *viewMat{};
		const float *projMat{};
		const float *pos{};

		void Update(const vkgsraster::Rasterizer::CameraArgs &vkCamera);
	};

	struct FwdROArgs {
		CameraArgs camera;
		uint32_t splatCount{};
		SplatArgs splats;
		const float *bgColor{};

		void Update(const vkgsraster::Rasterizer::FwdROArgs &vkROArgs);
	};

	struct FwdRWArgs {
		float *outPixels{};

		void Update(const vkgsraster::Rasterizer::FwdRWArgs &vkRWArgs);
	};

	struct BwdROArgs {
		FwdROArgs fwd{};
		const float *dL_dPixels{};

		void Update(const FwdROArgs &fwdROArgs, const vkgsraster::Rasterizer::BwdROArgs &vkROArgs);
	};

	struct BwdRWArgs {
		SplatArgs dL_dSplats;

		void Update(const vkgsraster::Rasterizer::BwdRWArgs &vkRWArgs);
	};

	static void Forward(const FwdROArgs &roArgs, const FwdRWArgs &rwArgs, Resource &resource,
	                    const PerfQuery &perfQuery = PerfQuery{}, bool allocOnly = false);

	static void Backward(const BwdROArgs &roArgs, const BwdRWArgs &rwArgs, Resource &resource,
	                     const PerfQuery &perfQuery = PerfQuery{}, bool allocOnly = false);
};

#endif
