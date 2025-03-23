//
// Created by adamyuan on 3/17/25.
//

#pragma once

#include <myvk/BufferBase.hpp>
#include <myvk/CommandBuffer.hpp>
#include <myvk/ComputePipeline.hpp>

namespace VkGSRaster {

class DeviceSorter {
public:
	// Read-Only
	struct ROArgs {
		myvk::Ptr<myvk::BufferBase> pCountBuffer;
	};
	// Read-Write
	struct RWArgs {
		myvk::Ptr<myvk::BufferBase> pKeyBuffer, pPayloadBuffer; // count * [uint]
	};

	struct ROArgsSyncState {
		myvk::BufferSyncState countBuffer;
	};
	struct RWArgsSyncState {
		myvk::BufferSyncState keyBuffer, payloadBuffer;
	};

	struct ArgsUsage {
		VkBufferUsageFlags countBuffer;
		VkBufferUsageFlags keyBuffer, payloadBuffer;
	};

	struct Resource {
		myvk::Ptr<myvk::BufferBase> pTempKeyBuffer, pTempPayloadBuffer; // count * [uint]
		myvk::Ptr<myvk::BufferBase> pGlobalHistBuffer;
		myvk::Ptr<myvk::BufferBase> pPassHistBuffer;
		myvk::Ptr<myvk::BufferBase> pIndexBuffer;
		myvk::Ptr<myvk::BufferBase> pDispatchArgBuffer;
		mutable bool isDispatchArgBufferInitialized = false;

		void update(const myvk::Ptr<myvk::Device> &pDevice, uint32_t count, double growFactor = 1.5);
	};

private:
	myvk::Ptr<myvk::PipelineLayout> mpPipelineLayout;
	myvk::Ptr<myvk::ComputePipeline> mpResetPipeline, mpGlobalHistPipeline, mpScanHistPipeline, mpOneSweepPipeline;

	static myvk::Ptr<myvk::ShaderModule> createResetShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createGlobalHistShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createScanHistShader(const myvk::Ptr<myvk::Device> &pDevice);
	static myvk::Ptr<myvk::ShaderModule> createOneSweepShader(const myvk::Ptr<myvk::Device> &pDevice);

public:
	DeviceSorter() = default;
	explicit DeviceSorter(const myvk::Ptr<myvk::Device> &pDevice);

	void CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &pCommandBuffer, const ROArgs &roArgs, const RWArgs &rwArgs,
	                const Resource &resource, bool keyAsPayload = false) const;

	// RWArgs must be visible to GetSrcRWArgsSync() before CmdExecute()
	static const RWArgsSyncState &GetSrcRWArgsSync();

	// RWArgs' later access must make GetDstRWArgsSync() available
	static const RWArgsSyncState &GetDstRWArgsSync();

	// ROArgs must be visible to GetROArgsSync() before CmdExecute()
	static const ROArgsSyncState &GetROArgsSync();

	static const ArgsUsage &GetArgsUsage();
};

} // namespace VkGSRaster