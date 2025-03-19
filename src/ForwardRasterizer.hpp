//
// Created by adamyuan on 3/17/25.
//

#pragma once

#include <myvk/Buffer.hpp>

namespace VkGSRaster {

class ForwardRasterizer {
public:
	struct InputArgs {
		int P;                                         // Splat count
		int D;                                         // SH Degree {0: SH1, 1: SH4, 2: SH9, 3: SH16}
		int M;                                         // SH Buffer stride (M * sizeof(float3))
		myvk::Ptr<myvk::BufferBase> pBackgroundBuffer; // [float3]
		int width, height;
		myvk::Ptr<myvk::BufferBase> pMean3DBuffer;       // P * [float3]
		myvk::Ptr<myvk::BufferBase> pSHBuffer;           // P * [M * float3]
		myvk::Ptr<myvk::BufferBase> pColorPrecompBuffer; // Optional, P * [float3]
		myvk::Ptr<myvk::BufferBase> pOpacityBuffer;      // P * [float]
		myvk::Ptr<myvk::BufferBase> pScaleBuffer;        // P * [float3]
		float scaleModifier;
		myvk::Ptr<myvk::BufferBase> pRotationBuffer;     // P * [float4]
		myvk::Ptr<myvk::BufferBase> pCov3DPrecompBuffer; // P * [float2x3]
		myvk::Ptr<myvk::BufferBase> pViewMatrixBuffer;   // [float4x4]
		myvk::Ptr<myvk::BufferBase> pProjMatrixBuffer;   // [float4x4]
		myvk::Ptr<myvk::BufferBase> pCamPosBuffer;       // [float3]
		float tanFovX, tanFovY;
	};

	struct OutputArgs {
		myvk::Ptr<myvk::BufferBase> pSortedIDBuffer;    // P * [uint]
		myvk::Ptr<myvk::BufferBase> pQuadBuffer;        // P * [float2x2]
		myvk::Ptr<myvk::BufferBase> pDrawArgBuffer;     // uint4
		myvk::Ptr<myvk::BufferBase> pDispatchArgBuffer; // uint3
	};
};

}