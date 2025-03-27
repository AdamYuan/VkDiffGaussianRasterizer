//
// Created by adamyuan on 3/17/25.
//

#pragma once

#include <array>
#include <myvk/BufferBase.hpp>
#include <myvk/ComputePipeline.hpp>
#include <myvk/GraphicsPipeline.hpp>
#include <myvk/ImageBase.hpp>

#include "DeviceSorter.hpp"

namespace VkGSRaster {

class Rasterizer {
public:
	struct ForwardROArgs {
		// Camera
		uint32_t camWidth, camHeight;
		float camTanFovX, camTanFovY;
		std::array<float, 3 * 3> camViewMatrix;
		std::array<float, 3> camPos;

		// Background
		std::array<float, 3> bgColor;

		// Splats
		uint32_t splatCount;
		myvk::Ptr<myvk::BufferBase> pMeanBuffer;    // P * [float3]
		myvk::Ptr<myvk::BufferBase> pScaleBuffer;   // P * [float3]
		myvk::Ptr<myvk::BufferBase> pRotateBuffer;  // P * [float4]
		myvk::Ptr<myvk::BufferBase> pOpacityBuffer; // P * [float]
		myvk::Ptr<myvk::BufferBase> pSHBuffer;      // P * [M * float3]
	};

	struct ForwardRWArgs {
		myvk::Ptr<myvk::BufferBase> pOutColorBuffer;
	};

	struct Resource {
		DeviceSorter::Resource sorterResource;

		myvk::Ptr<myvk::BufferBase> pSortKeyBuffer, pSortPayloadBuffer; // P * [uint]
		myvk::Ptr<myvk::BufferBase> pColorMean2DXBuffer;                // P * [float4]
		myvk::Ptr<myvk::BufferBase> pConicMean2DYBuffer;                // P * [float4]
		myvk::Ptr<myvk::BufferBase> pQuadBuffer;                        // P * [float4]
		myvk::Ptr<myvk::BufferBase> pDrawArgBuffer;                     // uint4
		myvk::Ptr<myvk::BufferBase> pDispatchArgBuffer;                 // uint3

		myvk::Ptr<myvk::ImageBase> pColorImage; // W * H * [float4]
		myvk::Ptr<myvk::Framebuffer> pColorForwardFramebuffer;

		void updateBuffer(const myvk::Ptr<myvk::Device> &pDevice, uint32_t splatCount, double growFactor = 1.5);
		void updateImage(const myvk::Ptr<myvk::Device> &pDevice, uint32_t width, uint32_t height,
		                 const Rasterizer &rasterizer);
	};

private:
	DeviceSorter mSorter;

	myvk::Ptr<myvk::PipelineLayout> mpPipelineLayout;
	myvk::Ptr<myvk::ComputePipeline> mpForwardResetPipeline, mpForwardViewPipeline;
	myvk::Ptr<myvk::GraphicsPipeline> mpForwardDrawPipeline;
	myvk::Ptr<myvk::RenderPass> mpForwardRenderPass;

	static myvk::Ptr<myvk::ShaderModule> createDrawVertShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createForwardResetShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createForwardViewShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createForwardDrawGeomShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createForwardDrawFragShader(const myvk::Ptr<myvk::Device> &pDevice);

public:
	Rasterizer() = default;
	explicit Rasterizer(const myvk::Ptr<myvk::Device> &pDevice);

	void CmdForward(const myvk::Ptr<myvk::CommandBuffer> &pCommandBuffer, const ForwardROArgs &roArgs,
	                const ForwardRWArgs &rwArgs, const Resource &resource) const;
};

} // namespace VkGSRaster