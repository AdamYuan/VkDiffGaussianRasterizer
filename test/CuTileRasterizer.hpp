//
// Created by adamyuan on 4/1/25.
//

#pragma once
#ifndef CUTILERASTERIZER_HPP
#define CUTILERASTERIZER_HPP

#include "../src/Rasterizer.hpp"
#include <array>

struct CuTileRasterizer {
	struct Resource {
		struct ResizeableBuffer {
			std::size_t size{};
			char *data{};

			char *Update(std::size_t updateSize);
		};
		ResizeableBuffer geometryBuffer{}, binningBuffer{}, imageBuffer{};
	};

	struct PerfMetrics {
		float forward;
	};

	struct PerfQuery {
		enum Event : uint32_t { kForwardStart, kForwardEnd, kEventCount };
		std::array<uintptr_t, kEventCount> events;

		static PerfQuery Create();
		void Record(Event event) const;
		PerfMetrics GetMetrics() const;
	};

	struct SplatArgs {
		const float *means{};
		const float *scales{};
		const float *rotates{};
		const float *opacities{};
		const float *shs{};
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
		float *outColor{};

		void Update(const vkgsraster::Rasterizer::FwdRWArgs &vkRWArgs);
	};

	static void Forward(const FwdROArgs &roArgs, const FwdRWArgs &rwArgs, Resource &resource,
	                    const PerfQuery &perfQuery = PerfQuery{});
};

#endif
