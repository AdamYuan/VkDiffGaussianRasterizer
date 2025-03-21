//
// Created by adamyuan on 3/17/25.
//

#include "DeviceSorter.hpp"

#include <array>
#include <myvk/Buffer.hpp>
#include <myvk/DescriptorSetLayout.hpp>
#include <shader/DeviceSorter/Size.hpp>

namespace VkGSRaster {

void DeviceSorter::Resource::update(const myvk::Ptr<myvk::Device> &pDevice, uint32_t count, double growFactor) {
	const auto growBuffer = [&pDevice, growFactor](myvk::Ptr<myvk::BufferBase> &pBuffer, uint32_t targetUintCount,
	                                               VkBufferUsageFlags bufferUsage) {
		if (pBuffer == nullptr || pBuffer->GetSize() < targetUintCount * sizeof(uint32_t)) {
			VkDeviceSize allocSize = pBuffer ? VkDeviceSize(double(pBuffer->GetSize()) * growFactor) : 0;
			allocSize = std::max(allocSize, targetUintCount * sizeof(uint32_t));
			pBuffer = myvk::Buffer::Create(pDevice, allocSize, 0, bufferUsage);
		}
	};
	const auto makeBuffer = [&pDevice](myvk::Ptr<myvk::BufferBase> &pBuffer, uint32_t uintCount,
	                                   VkBufferUsageFlags bufferUsage) {
		if (pBuffer == nullptr)
			pBuffer = myvk::Buffer::Create(pDevice, uintCount * sizeof(uint32_t), 0, bufferUsage);
	};
	growBuffer(pTempKeyBuffer, count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	growBuffer(pTempPayloadBuffer, count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	growBuffer(pPassHistBuffer, PASS_COUNT * RADIX * ((count + SORT_PART_SIZE - 1) / SORT_PART_SIZE),
	           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	makeBuffer(pGlobalHistBuffer, PASS_COUNT * RADIX, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	makeBuffer(pIndexBuffer, PASS_COUNT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	makeBuffer(pDispatchArgBuffer, 6, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
}

namespace {
struct OneSweepPushConstant {
	uint32_t passIdx, radixShift, writeKey;
};
} // namespace

DeviceSorter::DeviceSorter(const myvk::Ptr<myvk::Device> &pDevice) {
	auto pDescriptorSetLayout = myvk::DescriptorSetLayout::Create( //
	    pDevice,
	    std::vector<VkDescriptorSetLayoutBinding>{
	        {.binding = B_KEY_COUNT_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_DISPATCH_ARGS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_GLOBAL_HISTS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_PASS_HISTS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_INDICES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_SRC_KEYS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_SRC_PAYLOADS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_DST_KEYS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_DST_PAYLOADS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	    },
	    VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

	// Pipeline layouts
	mpPipelineLayout = myvk::PipelineLayout::Create(pDevice, {pDescriptorSetLayout}, {});

	mpOneSweepPipelineLayout = myvk::PipelineLayout::Create(
	    pDevice, {pDescriptorSetLayout},
	    {{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = (uint32_t)sizeof(OneSweepPushConstant)}});

	// Pipelines
	mpResetPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createResetShader(pDevice));
	mpGlobalHistPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createGlobalHistShader(pDevice));
	mpScanHistPipeline = myvk::ComputePipeline::Create(
	    mpPipelineLayout, createScanHistShader(pDevice)->GetPipelineShaderStageCreateInfo(
	                          VK_SHADER_STAGE_COMPUTE_BIT, VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT));
	mpOneSweepPipeline = myvk::ComputePipeline::Create(
	    mpOneSweepPipelineLayout,
	    createOneSweepShader(pDevice)->GetPipelineShaderStageCreateInfo(
	        VK_SHADER_STAGE_COMPUTE_BIT, VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT));
}
const DeviceSorter::RWArgsSyncState &DeviceSorter::GetSrcRWArgsSync() {
	static constexpr RWArgsSyncState kSync = {
	    .keyBuffer =
	        {
	            .stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	            .access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
	        },
	    .payloadBuffer =
	        {
	            .stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	            .access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
	        },
	};
	return kSync;
}
const DeviceSorter::RWArgsSyncState &DeviceSorter::GetDstRWArgsSync() {
	static constexpr RWArgsSyncState kSync = {
	    .keyBuffer =
	        {
	            .stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	            .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	        },
	    .payloadBuffer =
	        {
	            .stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	            .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	        },
	};
	return kSync;
}
const DeviceSorter::ROArgsSyncState &DeviceSorter::GetROArgsSync() {
	static constexpr ROArgsSyncState kSync = {.countBuffer = {
	                                              .stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                                              .access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
	                                          }};
	return kSync;
}

const DeviceSorter::ArgsUsage &DeviceSorter::GetArgsUsage() {
	static constexpr ArgsUsage kUsage = {
	    .countBuffer = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	    .keyBuffer = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	    .payloadBuffer = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	};
	return kUsage;
}

void DeviceSorter::CmdExecute(const myvk::Ptr<myvk::CommandBuffer> &pCommandBuffer, const ROArgs &roArgs,
                              const RWArgs &rwArgs, const Resource &resource, bool keyAsPayload) const {
	myvk::Ptr<myvk::BufferBase> pSrcKeyBuffer = rwArgs.pKeyBuffer, pSrcPayloadBuffer = rwArgs.pPayloadBuffer;
	myvk::Ptr<myvk::BufferBase> pDstKeyBuffer = resource.pTempKeyBuffer,
	                            pDstPayloadBuffer = resource.pTempPayloadBuffer;

	const std::vector kInitialDescriptorSetWrites = {
	    myvk::DescriptorSetWrite::WriteUniformBuffer(nullptr, roArgs.pCountBuffer, B_KEY_COUNT_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pDispatchArgBuffer, B_DISPATCH_ARGS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pGlobalHistBuffer, B_GLOBAL_HISTS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pPassHistBuffer, B_PASS_HISTS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pIndexBuffer, B_INDICES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pSrcKeyBuffer, B_SRC_KEYS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pSrcPayloadBuffer, B_SRC_PAYLOADS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pDstKeyBuffer, B_DST_KEYS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pDstPayloadBuffer, B_DST_PAYLOADS_BINDING),
	};

	// Since all pipeline layouts are compatible, we only need to bind the majority of buffers once
	pCommandBuffer->CmdPushDescriptorSet(mpPipelineLayout, VK_PIPELINE_BIND_POINT_COMPUTE, 0,
	                                     kInitialDescriptorSetWrites);

	// Reset Pass
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pDispatchArgBuffer->GetMemoryBarrier2(
	            // Prev-Access is READ
	            {.stage_mask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, .access_mask = 0},
	            {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pGlobalHistBuffer->GetMemoryBarrier2(
	            // Prev-Access is READ
	            {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .access_mask = 0}, // Last access is READ
	            {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pIndexBuffer->GetMemoryBarrier2(
	            // Prev-Access is WRITE
	            {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pPassHistBuffer->GetMemoryBarrier2(
	            // Prev-Access is WRITE
	            {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	    },
	    {});
	pCommandBuffer->CmdBindPipeline(mpResetPipeline);
	pCommandBuffer->CmdDispatch(RESET_GROUP_COUNT, 1, 1);

	// GlobalHist Pass
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pDispatchArgBuffer->GetMemoryBarrier2({.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                                                     .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	                                                    {.stage_mask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
	                                                     .access_mask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT}),
	        resource.pGlobalHistBuffer->GetMemoryBarrier2(
	            {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             .access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	    },
	    {});
	pCommandBuffer->CmdBindPipeline(mpGlobalHistPipeline);
	pCommandBuffer->CmdDispatchIndirect(resource.pDispatchArgBuffer, 0 * sizeof(uint32_t));

	// ScanHist Pass
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pGlobalHistBuffer->GetMemoryBarrier2({.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                                                       .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	                                                      {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                                                       .access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	        resource.pPassHistBuffer->GetMemoryBarrier2({.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                                                     .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	                                                    {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                                                     .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	    },
	    {});
	pCommandBuffer->CmdBindPipeline(mpScanHistPipeline);
	pCommandBuffer->CmdDispatch(PASS_COUNT, 1, 1);

	// OneSweep Passes
	pCommandBuffer->CmdBindPipeline(mpOneSweepPipeline);
	for (uint32_t passIdx = 0; passIdx < PASS_COUNT; ++passIdx) {
		std::vector bufferMemoryBarriers = {
		    resource.pPassHistBuffer->GetMemoryBarrier2(
		        {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		         .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
		        {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		         .access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
		    resource.pIndexBuffer->GetMemoryBarrier2(
		        {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		         .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
		        {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		         .access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
		    pDstKeyBuffer->GetMemoryBarrier2(
		        // Prev-Access is READ
		        {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .access_mask = 0},
		        {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		         .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
		    pDstPayloadBuffer->GetMemoryBarrier2(
		        // Prev-Access is READ
		        {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .access_mask = 0},
		        {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		         .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
		};

		if (passIdx > 0) {
			bufferMemoryBarriers.push_back(
			    pSrcKeyBuffer->GetMemoryBarrier2({.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			                                      .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
			                                     {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			                                      .access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT}));
			bufferMemoryBarriers.push_back(
			    pSrcPayloadBuffer->GetMemoryBarrier2({.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			                                          .access_mask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
			                                         {.stage_mask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			                                          .access_mask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT}));
		}

		pCommandBuffer->CmdPipelineBarrier2({}, bufferMemoryBarriers, {});

		if (passIdx == 0) {
			pCommandBuffer->CmdPushDescriptorSet(mpOneSweepPipelineLayout, VK_PIPELINE_BIND_POINT_COMPUTE, 0,
			                                     kInitialDescriptorSetWrites);
		} else {
			pCommandBuffer->CmdPushDescriptorSet(
			    mpOneSweepPipelineLayout, VK_PIPELINE_BIND_POINT_COMPUTE, 0,
			    {
			        myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pSrcKeyBuffer, B_SRC_KEYS_BINDING),
			        myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pSrcPayloadBuffer, B_SRC_PAYLOADS_BINDING),
			        myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pDstKeyBuffer, B_DST_KEYS_BINDING),
			        myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pDstPayloadBuffer, B_DST_PAYLOADS_BINDING),
			    });
		}

		OneSweepPushConstant pcData = {
		    .passIdx = passIdx,
		    .radixShift = passIdx * BITS_PER_PASS,
		    .writeKey = uint32_t(keyAsPayload || passIdx < PASS_COUNT - 1),
		};
		pCommandBuffer->CmdPushConstants(mpOneSweepPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
		                                 sizeof(OneSweepPushConstant), &pcData);
		pCommandBuffer->CmdDispatchIndirect(resource.pDispatchArgBuffer, 3 * sizeof(uint32_t));

		std::swap(pSrcKeyBuffer, pDstKeyBuffer);
		std::swap(pSrcPayloadBuffer, pDstPayloadBuffer);
	}
}

} // namespace VkGSRaster