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

namespace vkgsraster {

class Rasterizer {
public:
	struct Config {
		bool forwardOutputImage = false;
		bool performanceMetrics = false;
	};

	struct CameraArgs {
		uint32_t width, height;
		float focalX, focalY;
		std::array<float, 3 * 3> viewMat;
		std::array<float, 3> pos;

		static float GetFocalFromTanFov(float tanFov, uint32_t dim) { return float(dim) * 0.5f / tanFov; }
		static float GetTanFovFromFocal(float focal, uint32_t dim) { return float(dim) * 0.5f / focal; }
		// actually, tanFov = tan(fov / 2)
	};

	struct SplatArgs {
		myvk::Ptr<myvk::BufferBase> pMeanBuffer;    // P * [float3]
		myvk::Ptr<myvk::BufferBase> pScaleBuffer;   // P * [float3]
		myvk::Ptr<myvk::BufferBase> pRotateBuffer;  // P * [float4]
		myvk::Ptr<myvk::BufferBase> pOpacityBuffer; // P * [float]
		myvk::Ptr<myvk::BufferBase> pSHBuffer;      // P * [M * float3]
	};

	struct FwdROArgs {
		CameraArgs camera{};
		uint32_t splatCount{};
		SplatArgs splats{};
		std::array<float, 3> bgColor{};
	};
	struct FwdROArgsSyncState {
		myvk::BufferSyncState splatBuffers;
		myvk::BufferSyncState splatOpacityBuffer;
	};

	struct FwdRWArgs {
		myvk::Ptr<myvk::BufferBase> pOutColorBuffer;
		myvk::Ptr<myvk::ImageBase> pOutColorImage;
	};
	struct FwdRWArgsSyncState {
		myvk::BufferSyncState outColorBuffer;
		myvk::ImageSyncState outColorImage;
	};

	struct FwdArgsUsage {
		VkBufferUsageFlags splatBuffers;
		VkBufferUsageFlags outColorBuffer;
		VkImageUsageFlags outColorImage;
	};

	struct BwdROArgs {
		FwdROArgs fwd;
		myvk::Ptr<myvk::BufferBase> pdL_dColorBuffer;
	};

	struct BwdRWArgs {
		SplatArgs dL_dSplats{};
	};

	struct Resource {
		DeviceSorter::Resource sorterResource;

		myvk::Ptr<myvk::BufferBase> pSortKeyBuffer, pSortPayloadBuffer;
		myvk::Ptr<myvk::BufferBase> pSortSplatIndexBuffer; // P * [uint]

		myvk::Ptr<myvk::BufferBase> pColorMean2DXBuffer; // P * [float4]
		myvk::Ptr<myvk::BufferBase> pConicMean2DYBuffer; // P * [float4]
		myvk::Ptr<myvk::BufferBase> pViewOpacityBuffer;  // P * [float]

		myvk::Ptr<myvk::BufferBase> pDL_DColorMean2DXBuffer; // P * [float4]
		myvk::Ptr<myvk::BufferBase> pDL_DConicMean2DYBuffer; // P * [float4]
		myvk::Ptr<myvk::BufferBase> pDL_DViewOpacityBuffer;  // P * [float]

		myvk::Ptr<myvk::BufferBase> pQuadBuffer; // P * [float4]

		myvk::Ptr<myvk::BufferBase> pDrawArgBuffer;     // uint4
		myvk::Ptr<myvk::BufferBase> pDispatchArgBuffer; // uint3

		myvk::Ptr<myvk::ImageBase> pImage0, pImage1; // W * H * [float4]
		myvk::Ptr<myvk::ImageView> pImageView0, pImageView1;

		myvk::Ptr<myvk::Framebuffer> pForwardFramebuffer, pBackwardFramebuffer;

		void UpdateBuffer(const myvk::Ptr<myvk::Device> &pDevice, uint32_t splatCount, double growFactor = 1.5);
		void UpdateImage(const myvk::Ptr<myvk::Device> &pDevice, uint32_t width, uint32_t height,
		                 const Rasterizer &rasterizer);
	};

	struct PerfMetrics {
		double forward, forwardView, forwardSort, forwardDraw;
	};

	struct PerfQuery {
		enum Timestamp : uint32_t {
			kForward,
			kForwardView,
			kForwardSort,
			kForwardDraw,
			kTimestampCount,
		};
		myvk::Ptr<myvk::QueryPool> pQueryPool;
		// myvk::Ptr<myvk::BufferBase> pResultBuffer;
		// const uint64_t *pMappedResults;

		static PerfQuery Create(const myvk::Ptr<myvk::Device> &pDevice);
		void CmdWriteTimestamp(const myvk::Ptr<myvk::CommandBuffer> &pCommandBuffer, Timestamp timestamp) const;
		PerfMetrics GetMetrics() const;
	};

private:
	Config mConfig;

	DeviceSorter mSorter;

	myvk::Ptr<myvk::PipelineLayout> mpPipelineLayout;

	myvk::Ptr<myvk::ComputePipeline> mpForwardResetPipeline, mpForwardViewPipeline, mpForwardCopyPipeline;
	myvk::Ptr<myvk::GraphicsPipeline> mpForwardDrawPipeline;
	myvk::Ptr<myvk::RenderPass> mpForwardRenderPass;

	myvk::Ptr<myvk::ComputePipeline> mpBackwardResetPipeline, mpBackwardViewPipeline, mpBackwardCopyPipeline;
	myvk::Ptr<myvk::GraphicsPipeline> mpBackwardDrawPipeline;
	myvk::Ptr<myvk::RenderPass> mpBackwardRenderPass;

	static myvk::Ptr<myvk::ShaderModule> createDrawVertShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createForwardResetShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createForwardViewShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createForwardDrawGeomShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createForwardDrawFragShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createForwardCopyShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createBackwardResetShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createBackwardViewShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createBackwardDrawGeomShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createBackwardDrawFragShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createBackwardCopyShader(const myvk::Ptr<myvk::Device> &pDevice);

public:
	Rasterizer() = default;
	explicit Rasterizer(const myvk::Ptr<myvk::Device> &pDevice, const Config &config);

	const Config &GetConfig() const { return mConfig; }

	void CmdForward(const myvk::Ptr<myvk::CommandBuffer> &pCommandBuffer, const FwdROArgs &roArgs,
	                const FwdRWArgs &rwArgs, const Resource &resource, const PerfQuery &perfQuery = {}) const;

	static const FwdRWArgsSyncState &GetSrcFwdRWArgsSync();
	static const FwdRWArgsSyncState &GetDstFwdRWArgsSync();
	static const FwdROArgsSyncState &GetFwdROArgsSync();
	static const FwdArgsUsage &GetFwdArgsUsage();
};

} // namespace vkgsraster