//
// Created by adamyuan on 3/17/25.
//

#pragma once

#include <array>
#include <myvk/BufferBase.hpp>
#include <myvk/ImageBase.hpp>

namespace VkGSRaster {

class Rasterizer {
public:
	struct ForwardROArgs {
		// Splats
		int splatCount;
		myvk::Ptr<myvk::BufferBase> pMeanBuffer;    // P * [float3]
		myvk::Ptr<myvk::BufferBase> pScaleBuffer;   // P * [float3]
		myvk::Ptr<myvk::BufferBase> pRotateBuffer;  // P * [float4]
		myvk::Ptr<myvk::BufferBase> pOpacityBuffer; // P * [float]
		myvk::Ptr<myvk::BufferBase> pSHBuffer;      // P * [M * float3]

		// Camera
		int camWidth, camHeight;
		float camTanFovX, camTanFovY;
		std::array<float, 3 * 3> camViewMatrix;
		std::array<float, 3> camPos;

		// Background
		std::array<float, 3> bgColor;
	};

	struct ForwardRWArgs {
		myvk::Ptr<myvk::BufferBase> pOutColorBuffer;
	};

	struct Resource {
		myvk::Ptr<myvk::BufferBase> pSortKeyBuffer, pSortPayloadBuffer; // P * [uint]
		myvk::Ptr<myvk::BufferBase> pColorMean2DXBuffer;                // P * [float4]
		myvk::Ptr<myvk::BufferBase> pConicMean2DYBuffer;                // P * [float4]
		myvk::Ptr<myvk::BufferBase> pQuadBuffer;                        // P * [float2x2]
		myvk::Ptr<myvk::BufferBase> pDrawArgBuffer;                     // uint4
		myvk::Ptr<myvk::BufferBase> pDispatchArgBuffer;                 // uint3
		myvk::Ptr<myvk::ImageBase> pColorImage;                         // W * H * [float4]
	};

private:
};

} // namespace VkGSRaster