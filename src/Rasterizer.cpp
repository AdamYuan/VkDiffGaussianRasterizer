//
// Created by adamyuan on 3/17/25.
//

#include "Rasterizer.hpp"

#include <array>
#include <shader/Rasterizer/Size.hpp>

#include "ResourceUtil.hpp"

namespace vkgsraster {

void Rasterizer::Resource::UpdateBuffer(const myvk::Ptr<myvk::Device> &pDevice, uint32_t splatCount,
                                        double growFactor) {
	sorterResource.Update(pDevice, splatCount, growFactor);

	// Sort Buffers
	GrowBuffer<sizeof(uint32_t)>(pDevice, pSortKeyBuffer,
	                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | DeviceSorter::GetArgsUsage().keyBuffer,
	                             splatCount, growFactor);
	GrowBuffer<sizeof(uint32_t)>(pDevice, pSortPayloadBuffer,
	                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | DeviceSorter::GetArgsUsage().payloadBuffer,
	                             splatCount, growFactor);
	GrowBuffer<sizeof(uint32_t)>(pDevice, pSortSplatIndexBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount,
	                             growFactor);

	// SplatViews
	GrowBuffer<sizeof(float) * 4>(pDevice, pColorMean2DXBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount,
	                              growFactor);
	GrowBuffer<sizeof(float) * 4>(pDevice, pConicMean2DYBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount,
	                              growFactor);
	GrowBuffer<sizeof(float)>(pDevice, pViewOpacityBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount, growFactor);

	// DL_DSplatViews
	GrowBuffer<sizeof(float) * 4>(pDevice, pDL_DColorMean2DXBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount,
	                              growFactor);
	GrowBuffer<sizeof(float) * 4>(pDevice, pDL_DConicMean2DYBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount,
	                              growFactor);
	GrowBuffer<sizeof(float)>(pDevice, pDL_DViewOpacityBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount,
	                          growFactor);

	// SplatQuads
	GrowBuffer<sizeof(float) * 4>(pDevice, pQuadBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount, growFactor);

	// Draw & Dispatch Args
	MakeBuffer<sizeof(VkDrawIndirectCommand)>(pDevice, pDrawArgBuffer,
	                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
	                                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
	                                              DeviceSorter::GetArgsUsage().countBuffer,
	                                          1);
	MakeBuffer<sizeof(VkDispatchIndirectCommand)>(
	    pDevice, pDispatchArgBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, 1);
}
void Rasterizer::Resource::UpdateImage(const myvk::Ptr<myvk::Device> &pDevice, uint32_t width, uint32_t height,
                                       const Rasterizer &rasterizer) {
	VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	if (rasterizer.GetConfig().forwardOutputImage)
		usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	else
		usage |= VK_IMAGE_USAGE_SAMPLED_BIT;

	if (ResizeImage<VK_FORMAT_R32G32B32A32_SFLOAT>(pDevice, pColorImage, usage, width, height))
		pColorImageView = myvk::ImageView::Create(pColorImage, VK_IMAGE_VIEW_TYPE_2D);

	ResizeFramebuffer(rasterizer.mpForwardRenderPass, {}, pForwardFramebuffer, width, height);
}

Rasterizer::PerfQuery Rasterizer::PerfQuery::Create(const myvk::Ptr<myvk::Device> &pDevice) {
	PerfQuery perfQuery{};
	perfQuery.pQueryPool = myvk::QueryPool::Create(pDevice, VK_QUERY_TYPE_TIMESTAMP, kTimestampCount);

	/* auto pResultBuffer =
	    myvk::Buffer::Create(pDevice, kQueryCount * sizeof(uint64_t),
	                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
	                         VK_BUFFER_USAGE_TRANSFER_DST_BIT);
	perfQuery.pResultBuffer = pResultBuffer;
	perfQuery.pMappedResults = reinterpret_cast<const uint64_t *>(pResultBuffer->GetMappedData()); */
	return perfQuery;
}
void Rasterizer::PerfQuery::CmdWriteTimestamp(const myvk::Ptr<myvk::CommandBuffer> &pCommandBuffer,
                                              Timestamp timestamp) const {
	if (!pQueryPool)
		return;
	vkCmdWriteTimestamp2(pCommandBuffer->GetHandle(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pQueryPool->GetHandle(),
	                     static_cast<uint32_t>(timestamp));
}
Rasterizer::PerfMetrics Rasterizer::PerfQuery::GetMetrics() const {
	std::array<uint64_t, kTimestampCount> results{};
	if (pQueryPool->GetResults64(0, kTimestampCount, results.data(), 0) != VK_SUCCESS)
		return {}; // Not ready
	pQueryPool->Reset(0, kTimestampCount);

	auto timestampPeriod =
	    (double)pQueryPool->GetDevicePtr()->GetPhysicalDevicePtr()->GetProperties().vk10.limits.timestampPeriod;

	const auto getMilliseconds = [&](Timestamp l, Timestamp r) {
		return double(results[r] - results[l]) * timestampPeriod / 1e6;
	};

	return PerfMetrics{
	    .forward = getMilliseconds(kForward, kForwardDraw),
	    .forwardView = getMilliseconds(kForward, kForwardView),
	    .forwardSort = getMilliseconds(kForwardView, kForwardSort),
	    .forwardDraw = getMilliseconds(kForwardSort, kForwardDraw),
	};
}

namespace {
struct PushConstantData {
	std::array<float, 3> bgColor;
	uint32_t splatCount;
	std::array<float, 2> camFocal;
	std::array<uint32_t, 2> camResolution;
	std::array<float, 3> camPos;
	std::array<float, 9> camViewMat; // Column-Major
};
} // namespace

Rasterizer::Rasterizer(const myvk::Ptr<myvk::Device> &pDevice, const Config &config)
    : mConfig{config}, mSorter{pDevice, {.useKeyAsPayload = false}} {
	auto pDescriptorSetLayout = myvk::DescriptorSetLayout::Create( //
	    pDevice,
	    std::vector<VkDescriptorSetLayoutBinding>{
	        // Splats
	        {.binding = SBUF_MEANS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_SCALES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_ROTATES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_OPACITIES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_SHS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},

	        // Sort Buffers
	        {.binding = SBUF_SORT_KEYS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_SORT_PAYLOADS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},
	        {.binding = SBUF_SORT_SPLAT_INDICES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},

	        // SplatViews
	        {.binding = SBUF_COLORS_MEAN2DXS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},
	        {.binding = SBUF_CONICS_MEAN2DYS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},
	        {.binding = SBUF_VIEW_OPACITIES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},

	        // SplatQuads
	        {.binding = SBUF_QUADS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},

	        // Draw Args
	        {.binding = SBUF_DRAW_ARGS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = UBUF_SORT_COUNT_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},

	        // Images
	        {.binding = SIMG_IMAGE0_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
	        {.binding = TEX_IMAGE0_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_PIXELS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},

	        // Dispatch Args
	        {.binding = SBUF_DISPATCH_ARGS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},

	        // DL_DSplats
	        {.binding = SBUF_DL_DMEANS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_DL_DSCALES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_DL_DROTATES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_DL_DOPACITIES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = SBUF_DL_DSHS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},

	        // DL_DSplatViews
	        {.binding = SBUF_DL_DCOLORS_MEAN2DXS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
	        {.binding = SBUF_DL_DCONICS_MEAN2DYS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
	        {.binding = SBUF_DL_DVIEW_OPACITIES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
	    },
	    VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

	mpPipelineLayout = myvk::PipelineLayout::Create(
	    pDevice, {pDescriptorSetLayout},
	    {VkPushConstantRange{
	        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	        .offset = 0u,
	        .size = sizeof(PushConstantData),
	    }});

	// Forward RenderPass
	mpForwardRenderPass = myvk::RenderPass::Create(pDevice, [&] {
		myvk::RenderPassState2 state;
		state.SetAttachmentCount(0).SetSubpassCount(1).SetSubpass(0, {}).SetDependencyCount(0);
		return state;
	}());

	// Common Shader Module
	auto pDrawVertShader = createDrawVertShader(pDevice);

	// Forward Pipelines
	mpForwardResetPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createForwardResetShader(pDevice));
	mpForwardViewPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createForwardViewShader(pDevice));
	mpForwardCopyPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createForwardCopyShader(pDevice));
	mpForwardDrawPipeline = myvk::GraphicsPipeline::Create(
	    mpPipelineLayout, mpForwardRenderPass,
	    myvk::GraphicsPipelineShaderModules{.vert = pDrawVertShader,
	                                        .geom = createForwardDrawGeomShader(pDevice),
	                                        .frag = createForwardDrawFragShader(pDevice)},
	    [] {
		    myvk::GraphicsPipelineState state{};
		    state.m_dynamic_state.Enable({VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});
		    state.m_color_blend_state.Enable(0, VK_FALSE);
		    state.m_viewport_state.Enable();
		    state.m_vertex_input_state.Enable();
		    state.m_input_assembly_state.Enable(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
		    state.m_rasterization_state.Initialize(VK_POLYGON_MODE_FILL, VK_FRONT_FACE_CLOCKWISE, VK_CULL_MODE_NONE);
		    state.m_multisample_state.Enable(VK_SAMPLE_COUNT_1_BIT);
		    return state;
	    }(),
	    0);
}

void Rasterizer::CmdForward(const myvk::Ptr<myvk::CommandBuffer> &pCommandBuffer, const FwdROArgs &roArgs,
                            const FwdRWArgs &rwArgs, const Resource &resource, const PerfQuery &perfQuery) const {
	std::vector descriptorSetWrites = {
	    // Splats
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.splats.pMeanBuffer, SBUF_MEANS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.splats.pScaleBuffer, SBUF_SCALES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.splats.pRotateBuffer, SBUF_ROTATES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.splats.pOpacityBuffer, SBUF_OPACITIES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.splats.pSHBuffer, SBUF_SHS_BINDING),

	    // Sort Buffers
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pSortKeyBuffer, SBUF_SORT_KEYS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pSortPayloadBuffer, SBUF_SORT_PAYLOADS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pSortSplatIndexBuffer,
	                                                 SBUF_SORT_SPLAT_INDICES_BINDING),

	    // SplatViews
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pColorMean2DXBuffer,
	                                                 SBUF_COLORS_MEAN2DXS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pConicMean2DYBuffer,
	                                                 SBUF_CONICS_MEAN2DYS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pViewOpacityBuffer, SBUF_VIEW_OPACITIES_BINDING),

	    // SplatQuads
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pQuadBuffer, SBUF_QUADS_BINDING),

	    // Draw Args
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pDrawArgBuffer, SBUF_DRAW_ARGS_BINDING),
	    // myvk::DescriptorSetWrite::WriteUniformBuffer(nullptr, resource.pDrawArgBuffer, SBUF_SORT_COUNT_BINDING),

	    // Images
	    myvk::DescriptorSetWrite::WriteStorageImage(nullptr, resource.pColorImageView, SIMG_IMAGE0_BINDING),
	};
	if (!mConfig.forwardOutputImage) {
		descriptorSetWrites.push_back(
		    myvk::DescriptorSetWrite::WriteSampledImage(nullptr, resource.pColorImageView, TEX_IMAGE0_BINDING));
		descriptorSetWrites.push_back(
		    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, rwArgs.pOutColorBuffer, SBUF_PIXELS_BINDING));
	}

	PushConstantData pcData = {
	    .bgColor = roArgs.bgColor,
	    .splatCount = roArgs.splatCount,
	    .camFocal = {roArgs.camera.focalX, roArgs.camera.focalY},
	    .camResolution = {roArgs.camera.width, roArgs.camera.height},
	    .camPos = roArgs.camera.pos,
	    .camViewMat = roArgs.camera.viewMat,
	};

	// Descriptors and Push Constants
	pCommandBuffer->CmdPushDescriptorSet(mpPipelineLayout, VK_PIPELINE_BIND_POINT_COMPUTE, 0, descriptorSetWrites);
	pCommandBuffer->CmdPushConstants(
	    mpPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, //
	    0, sizeof(PushConstantData), &pcData);

	// Reset
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pDrawArgBuffer->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT reads as DrawArg (ForwardDraw | BackwardDraw)
	            // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads as UniformBuffer (BackwardReset | BackwardView)
	            // DeviceSorter reads
	            {VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
	                 DeviceSorter::GetROArgsSync().countBuffer.stage_mask,
	             0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT}),
	    },
	    {});
	pCommandBuffer->CmdBindPipeline(mpForwardResetPipeline);
	pCommandBuffer->CmdDispatch(1, 1, 1);
	perfQuery.CmdWriteTimestamp(pCommandBuffer, PerfQuery::Timestamp::kForward);

	// View
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pDrawArgBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT}),
	        resource.pSortKeyBuffer->GetMemoryBarrier2(
	            DeviceSorter::GetDstRWArgsSync().keyBuffer.GetWrite(),
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT}),
	        resource.pSortPayloadBuffer->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT reads as StorageBuffer (ForwardDraw.geom | BackwardDraw.geom)
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT}),
	        resource.pSortSplatIndexBuffer->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads as StorageBuffer (BackwardView)
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT}),
	        // SplatView and SplatQuad Buffers
	        // VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT reads (ForwardDraw.geom | BackwardDraw.geom)
	        resource.pColorMean2DXBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pConicMean2DYBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pViewOpacityBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),

	        resource.pQuadBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	    },
	    {});
	pCommandBuffer->CmdBindPipeline(mpForwardViewPipeline);
	pCommandBuffer->CmdDispatch((roArgs.splatCount + FORWARD_VIEW_DIM - 1) / FORWARD_VIEW_DIM, 1, 1);
	perfQuery.CmdWriteTimestamp(pCommandBuffer, PerfQuery::Timestamp::kForwardView);

	// Prepare for Sort (+ Read-After-Write Barrier for pDrawArgBuffer)
	pCommandBuffer->CmdPipelineBarrier2( //
	    {},
	    {
	        resource.pSortKeyBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT},
	            DeviceSorter::GetSrcRWArgsSync().keyBuffer),
	        resource.pSortPayloadBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT},
	            DeviceSorter::GetSrcRWArgsSync().payloadBuffer),
	        resource.pDrawArgBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT},
	            DeviceSorter::GetROArgsSync().countBuffer |
	                // VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT reads as DrawArg (ForwardDraw | BackwardDraw)
	                // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads as UniformBuffer (BackwardReset | BackwardView)
	                myvk::BufferSyncState{
	                    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT,
	                }),
	    },
	    {});

	// Sort
	mSorter.CmdSort(pCommandBuffer, {.pCountBuffer = resource.pDrawArgBuffer},
	                {.pKeyBuffer = resource.pSortKeyBuffer, .pPayloadBuffer = resource.pSortPayloadBuffer},
	                resource.sorterResource);
	perfQuery.CmdWriteTimestamp(pCommandBuffer, PerfQuery::Timestamp::kForwardSort);

	// Rebind again after Sort
	pCommandBuffer->CmdPushDescriptorSet(mpPipelineLayout, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, descriptorSetWrites);
	if (!mConfig.forwardOutputImage)
		pCommandBuffer->CmdPushDescriptorSet(mpPipelineLayout, VK_PIPELINE_BIND_POINT_COMPUTE, 0, descriptorSetWrites);
	pCommandBuffer->CmdPushConstants(
	    mpPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, //
	    0, sizeof(PushConstantData), &pcData);

	// Read-After-Write Barriers for pSortPayloadBuffer, pSortSplatIndexBuffer, SplatView Buffers, SplatQuad Buffers
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pSortPayloadBuffer->GetMemoryBarrier2(
	            DeviceSorter::GetDstRWArgsSync().payloadBuffer.GetWrite(),
	            // VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT reads as StorageBuffer (ForwardDraw.geom | BackwardDraw.geom)
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	        resource.pSortSplatIndexBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads as StorageBuffer (BackwardView)
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	        // SplatView and SplatQuad Buffers
	        // VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT reads (ForwardDraw.geom | BackwardDraw.geom)
	        resource.pColorMean2DXBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	        resource.pConicMean2DYBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	        resource.pViewOpacityBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	        resource.pQuadBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	    },
	    {});

	// Clear Color Image
	pCommandBuffer->CmdPipelineBarrier2(
	    {}, {},
	    {
	        resource.pColorImage->GetMemoryBarrier2(
	            {
	                mConfig.forwardOutputImage ? VK_PIPELINE_STAGE_2_BLIT_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	            },
	            {VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}),
	    });
	pCommandBuffer->CmdClearColorImage(resource.pColorImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                                   {roArgs.bgColor[0], roArgs.bgColor[1], roArgs.bgColor[2], 1.0f});

	// Draw
	pCommandBuffer->CmdPipelineBarrier2(
	    {}, {},
	    {
	        resource.pColorImage->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
	            {
	                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	                VK_IMAGE_LAYOUT_GENERAL,
	            }),
	    });
	pCommandBuffer->CmdBeginRenderPass(mpForwardRenderPass, resource.pForwardFramebuffer, {});
	pCommandBuffer->CmdBindPipeline(mpForwardDrawPipeline);
	pCommandBuffer->CmdSetViewport({VkViewport{
	    .x = 0,
	    .y = 0,
	    .width = (float)roArgs.camera.width,
	    .height = (float)roArgs.camera.height,
	    .minDepth = 0,
	    .maxDepth = 0,
	}});
	pCommandBuffer->CmdSetScissor({VkRect2D{
	    .offset = {},
	    .extent = {.width = roArgs.camera.width, .height = roArgs.camera.height},
	}});
	pCommandBuffer->CmdDrawIndirect(resource.pDrawArgBuffer, 0, 1);
	perfQuery.CmdWriteTimestamp(pCommandBuffer, PerfQuery::Timestamp::kForwardDraw);
	pCommandBuffer->CmdEndRenderPass();

	if (mConfig.forwardOutputImage) {
		pCommandBuffer->CmdPipelineBarrier2({}, {},
		                                    {
		                                        resource.pColorImage->GetMemoryBarrier2(
		                                            {
		                                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		                                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                                                VK_IMAGE_LAYOUT_GENERAL,
		                                            },
		                                            {
		                                                VK_PIPELINE_STAGE_2_BLIT_BIT,
		                                                VK_ACCESS_2_TRANSFER_READ_BIT,
		                                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		                                            }),
		                                    });
		pCommandBuffer->CmdBlitImage(resource.pColorImage, rwArgs.pOutColorImage,
		                             {
		                                 .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
		                                 .srcOffsets = {{}, {(int)roArgs.camera.width, (int)roArgs.camera.height, 1u}},
		                                 .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
		                                 .dstOffsets = {{}, {(int)roArgs.camera.width, (int)roArgs.camera.height, 1u}},
		                             },
		                             VK_FILTER_NEAREST);
	} else {
		pCommandBuffer->CmdPipelineBarrier2({}, {},
		                                    {
		                                        resource.pColorImage->GetMemoryBarrier2(
		                                            {
		                                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		                                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                                                VK_IMAGE_LAYOUT_GENERAL,
		                                            },
		                                            {
		                                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		                                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		                                            }),
		                                    });
		pCommandBuffer->CmdBindPipeline(mpForwardCopyPipeline);
		pCommandBuffer->CmdDispatch((roArgs.camera.width + FORWARD_COPY_DIM_X - 1) / FORWARD_COPY_DIM_X,
		                            (roArgs.camera.height + FORWARD_COPY_DIM_Y - 1) / FORWARD_COPY_DIM_Y, 1u);
	}
}

const Rasterizer::FwdRWArgsSyncState &Rasterizer::GetSrcFwdRWArgsSync() {
	static constexpr FwdRWArgsSyncState kSync = {
	    .outColorBuffer = {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	    .outColorImage = {VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
	};
	return kSync;
}
const Rasterizer::FwdRWArgsSyncState &Rasterizer::GetDstFwdRWArgsSync() {
	// Identical to GetSrcFwdRWArgsSync
	static constexpr FwdRWArgsSyncState kSync = {
	    .outColorBuffer = {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	    .outColorImage = {VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
	};
	return kSync;
}
const Rasterizer::FwdROArgsSyncState &Rasterizer::GetFwdROArgsSync() {

	static constexpr FwdROArgsSyncState kSync = {
	    .splatBuffers = {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT},
	    // Opacity buffer is additionally read in Geometry Shader (as SplatView.opacity)
	    .splatOpacityBuffer = {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
	                           VK_ACCESS_2_SHADER_STORAGE_READ_BIT},
	};
	return kSync;
}
const Rasterizer::FwdArgsUsage &Rasterizer::GetFwdArgsUsage() {
	static constexpr FwdArgsUsage kUsage = {
	    .splatBuffers = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	    .outColorBuffer = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	    .outColorImage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	};
	return kUsage;
}

} // namespace vkgsraster

// Check DeviceSorter KeyCountBufferOffset
#undef SBUF_DISPATCH_ARGS_BINDING // Remove warning
#include <shader/DeviceSorter/Size.hpp>
static_assert(KEY_COUNT_BUFFER_OFFSET == offsetof(VkDrawIndirectCommand, instanceCount));
static_assert(KEY_COUNT_BUFFER_OFFSET == SORT_COUNT_BUFFER_OFFSET);
