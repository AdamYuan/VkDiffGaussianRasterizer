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
	if (ResizeImage<VK_FORMAT_D32_SFLOAT>(pDevice, pDepthImage, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, width,
	                                      height))
		pDepthImageView = myvk::ImageView::Create(pDepthImage, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT);

	VkImageUsageFlags usage0 =
	    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
	if (rasterizer.GetConfig().forwardOutputImage)
		usage0 |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	else
		usage0 |= VK_IMAGE_USAGE_SAMPLED_BIT;
	if (ResizeImage<VK_FORMAT_R32G32B32A32_SFLOAT>(pDevice, pImage0, usage0, width, height))
		pImageView0 = myvk::ImageView::Create(pImage0, VK_IMAGE_VIEW_TYPE_2D);
	ResizeFramebuffer(rasterizer.mpForwardRenderPass, {pDepthImageView}, pForwardFramebuffer, width, height);

	if (ResizeImage<VK_FORMAT_R32G32B32A32_SFLOAT>(
	        pDevice, pImage1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, width, height))
		pImageView1 = myvk::ImageView::Create(pImage1, VK_IMAGE_VIEW_TYPE_2D);
	ResizeFramebuffer(rasterizer.mpBackwardRenderPass, {pImageView0, pDepthImageView}, pBackwardFramebuffer, width,
	                  height);
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
void Rasterizer::PerfQuery::Reset() const { pQueryPool->Reset(0, kTimestampCount); }
Rasterizer::PerfMetrics Rasterizer::PerfQuery::GetMetrics() const {
	std::array<uint64_t, kTimestampCount> results{};
	pQueryPool->GetResults64(0, kTimestampCount, results.data(), 0);

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
	    .backward = getMilliseconds(kBackward, kBackwardView),
	    .backwardReset = getMilliseconds(kBackward, kBackwardReset),
	    .backwardDraw = getMilliseconds(kBackwardReset, kBackwardDraw),
	    .backwardView = getMilliseconds(kBackwardDraw, kBackwardView),
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
	         .stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_COMPUTE_BIT},

	        // Images
	        {.binding = SIMG_IMAGE0_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
	        {.binding = TEX_IMAGE0_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = IATT_IMAGE0_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
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
		state.SetAttachmentCount(1)
		    .SetAttachment(
		        0, VK_FORMAT_D32_SFLOAT, {.op = VK_ATTACHMENT_LOAD_OP_CLEAR},
		        {.op = VK_ATTACHMENT_STORE_OP_STORE, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL})
		    .SetSubpassCount(1)
		    .SetSubpass(0,
		                {
		                    .opt_depth_stencil_attachment_ref =
		                        myvk::RenderPassState2::SubpassInfo::AttachmentRef{
		                            .attachment = 0,
		                            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		                        },
		                })
		    .SetDependencyCount(1)
		    .SetSrcExternalDependency(
		        0,
		        myvk::GetAttachmentStoreOpSync(VK_IMAGE_ASPECT_DEPTH_BIT, VK_ATTACHMENT_STORE_OP_DONT_CARE) |
		            myvk::GetAttachmentStoreOpSync(VK_IMAGE_ASPECT_DEPTH_BIT, VK_ATTACHMENT_STORE_OP_STORE),
		        {.subpass = 0,
		         .sync = myvk::GetAttachmentLoadOpSync(VK_IMAGE_ASPECT_DEPTH_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR)});
		return state;
	}());

	// Backward RenderPass
	mpBackwardRenderPass = myvk::RenderPass::Create(pDevice, [&] {
		myvk::RenderPassState2 state;
		state.SetAttachmentCount(2)
		    .SetAttachment(0, VK_FORMAT_R32G32B32A32_SFLOAT,
		                   {.op = VK_ATTACHMENT_LOAD_OP_NONE_EXT, .layout = VK_IMAGE_LAYOUT_GENERAL},
		                   {.op = VK_ATTACHMENT_STORE_OP_NONE, .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL})
		    .SetAttachment(
		        1, VK_FORMAT_D32_SFLOAT,
		        {.op = VK_ATTACHMENT_LOAD_OP_LOAD, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
		        {.op = VK_ATTACHMENT_STORE_OP_DONT_CARE, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL})
		    .SetSubpassCount(1)
		    .SetSubpass(0,
		                {
		                    .input_attachment_refs =
		                        {
		                            {
		                                .attachment = 0,
		                                .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		                                .aspect_mask = VK_IMAGE_ASPECT_COLOR_BIT,
		                            },
		                        },
		                    .opt_depth_stencil_attachment_ref =
		                        myvk::RenderPassState2::SubpassInfo::AttachmentRef{
		                            .attachment = 1,
		                            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		                        },
		                })
		    .SetDependencyCount(2)
		    .SetSrcExternalDependency(0,
		                              {
		                                  // BackwardCopy
		                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		                              },
		                              {
		                                  .subpass = 0,
		                                  .sync =
		                                      {
		                                          // BackwardDraw.frag
		                                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		                                          VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
		                                      },
		                              })
		    .SetSrcExternalDependency(
		        1, myvk::GetAttachmentStoreOpSync(VK_IMAGE_ASPECT_DEPTH_BIT, VK_ATTACHMENT_STORE_OP_STORE),
		        {.subpass = 0,
		         .sync = myvk::GetAttachmentLoadOpSync(VK_IMAGE_ASPECT_DEPTH_BIT, VK_ATTACHMENT_LOAD_OP_LOAD)});
		;
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
		    state.m_depth_stencil_state.Enable(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS);
		    return state;
	    }(),
	    0);

	// Backward Pipelines
	mpBackwardResetPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createBackwardResetShader(pDevice));
	mpBackwardViewPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createBackwardViewShader(pDevice));
	mpBackwardCopyPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createBackwardCopyShader(pDevice));
	mpBackwardDrawPipeline = myvk::GraphicsPipeline::Create(
	    mpPipelineLayout, mpBackwardRenderPass,
	    myvk::GraphicsPipelineShaderModules{.vert = pDrawVertShader,
	                                        .geom = createBackwardDrawGeomShader(pDevice),
	                                        .frag = createBackwardDrawFragShader(pDevice)},
	    [] {
		    myvk::GraphicsPipelineState state{};
		    state.m_dynamic_state.Enable({VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});
		    state.m_color_blend_state.Enable(0, VK_FALSE);
		    state.m_viewport_state.Enable();
		    state.m_vertex_input_state.Enable();
		    state.m_input_assembly_state.Enable(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
		    state.m_rasterization_state.Initialize(VK_POLYGON_MODE_FILL, VK_FRONT_FACE_CLOCKWISE, VK_CULL_MODE_NONE);
		    state.m_multisample_state.Enable(VK_SAMPLE_COUNT_1_BIT);
		    state.m_depth_stencil_state.Enable(VK_TRUE, VK_FALSE, VK_COMPARE_OP_LESS);
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
	    myvk::DescriptorSetWrite::WriteUniformBuffer(nullptr, resource.pDrawArgBuffer, UBUF_SORT_COUNT_BINDING),

	    // Images
	    myvk::DescriptorSetWrite::WriteStorageImage(nullptr, resource.pImageView0, SIMG_IMAGE0_BINDING),
	};
	if (!mConfig.forwardOutputImage) {
		descriptorSetWrites.push_back(
		    myvk::DescriptorSetWrite::WriteSampledImage(nullptr, resource.pImageView0, TEX_IMAGE0_BINDING));
		descriptorSetWrites.push_back(
		    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, rwArgs.pOutPixelBuffer, SBUF_PIXELS_BINDING));
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

	// ForwardReset
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pDrawArgBuffer->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT reads as DrawArg (ForwardDraw | BackwardDraw)
	            // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads as UniformBuffer (BackwardReset | BackwardView)
	            // VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT reads as UniformBuffer (ForwardDraw.geom | BackwardDraw.geom)
	            // DeviceSorter reads
	            {VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
	                 VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | DeviceSorter::GetROArgsSync().countBuffer.stage_mask,
	             0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	    },
	    {});
	pCommandBuffer->CmdBindPipeline(mpForwardResetPipeline);
	pCommandBuffer->CmdDispatch(1, 1, 1);
	perfQuery.CmdWriteTimestamp(pCommandBuffer, PerfQuery::Timestamp::kForward);

	// ForwardView
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pDrawArgBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	        resource.pSortKeyBuffer->GetMemoryBarrier2(
	            DeviceSorter::GetDstRWArgsSync().keyBuffer.GetWrite(),
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pSortPayloadBuffer->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT reads as StorageBuffer (ForwardDraw.geom | BackwardDraw.geom)
	            {VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pSortSplatIndexBuffer->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads as StorageBuffer (BackwardView)
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
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
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            DeviceSorter::GetSrcRWArgsSync().keyBuffer),
	        resource.pSortPayloadBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            DeviceSorter::GetSrcRWArgsSync().payloadBuffer),
	        resource.pDrawArgBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            DeviceSorter::GetROArgsSync().countBuffer |
	                // VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT reads as DrawArg (ForwardDraw | BackwardDraw)
	                // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads as UniformBuffer (BackwardReset | BackwardView)
	                // VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT reads as UniformBuffer (ForwardDraw.geom |
	                // BackwardDraw.geom)
	                myvk::BufferSyncState{
	                    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
	                        VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT,
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

	// Clear Image0
	pCommandBuffer->CmdPipelineBarrier2(
	    {}, {},
	    {
	        resource.pImage0->GetMemoryBarrier2(
	            {
	                // Last Read in Forward (CmdBlit or ForwardCopy)
	                (mConfig.forwardOutputImage ? VK_PIPELINE_STAGE_2_BLIT_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
	                    // Last Read in Backward (BackwardDraw)
	                    | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	            },
	            {VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}),
	    });
	pCommandBuffer->CmdClearColorImage(resource.pImage0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                                   {0.0f, 0.0f, 0.0f, 1.0f});

	// ForwardDraw
	pCommandBuffer->CmdPipelineBarrier2(
	    {}, {},
	    {
	        resource.pImage0->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
	            {
	                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	                VK_IMAGE_LAYOUT_GENERAL,
	            }),
	    });
	pCommandBuffer->CmdBeginRenderPass(mpForwardRenderPass, resource.pForwardFramebuffer,
	                                   {VkClearValue{.depthStencil = {.depth = 1.0f}}});
	pCommandBuffer->CmdBindPipeline(mpForwardDrawPipeline);
	pCommandBuffer->CmdSetViewport({VkViewport{
	    .x = 0,
	    .y = 0,
	    .width = (float)roArgs.camera.width,
	    .height = (float)roArgs.camera.height,
	    .minDepth = 0.0f,
	    .maxDepth = 1.0f,
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
		                                        resource.pImage0->GetMemoryBarrier2(
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
		pCommandBuffer->CmdBlitImage(resource.pImage0, rwArgs.pOutPixelImage,
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
		                                        resource.pImage0->GetMemoryBarrier2(
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

void Rasterizer::CmdBackward(const myvk::Ptr<myvk::CommandBuffer> &pCommandBuffer, const BwdROArgs &roArgs,
                             const BwdRWArgs &rwArgs, const Resource &resource, const PerfQuery &perfQuery) const {
	std::vector descriptorSetWrites = {
	    // Splats
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.fwd.splats.pMeanBuffer, SBUF_MEANS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.fwd.splats.pScaleBuffer, SBUF_SCALES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.fwd.splats.pRotateBuffer, SBUF_ROTATES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.fwd.splats.pOpacityBuffer, SBUF_OPACITIES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.fwd.splats.pSHBuffer, SBUF_SHS_BINDING),

	    // Sort Buffers
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

	    // Draw & Dispatch Args
	    myvk::DescriptorSetWrite::WriteUniformBuffer(nullptr, resource.pDrawArgBuffer, UBUF_SORT_COUNT_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pDispatchArgBuffer, SBUF_DISPATCH_ARGS_BINDING),

	    // Images
	    myvk::DescriptorSetWrite::WriteStorageImage(nullptr, resource.pImageView0, SIMG_IMAGE0_BINDING),
	    myvk::DescriptorSetWrite::WriteInputAttachment(nullptr, resource.pImageView0, IATT_IMAGE0_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, roArgs.pdL_dPixelBuffer, SBUF_PIXELS_BINDING),

	    // DL_DSplats
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, rwArgs.dL_dSplats.pMeanBuffer, SBUF_DL_DMEANS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, rwArgs.dL_dSplats.pScaleBuffer, SBUF_DL_DSCALES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, rwArgs.dL_dSplats.pRotateBuffer,
	                                                 SBUF_DL_DROTATES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, rwArgs.dL_dSplats.pOpacityBuffer,
	                                                 SBUF_DL_DOPACITIES_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, rwArgs.dL_dSplats.pSHBuffer, SBUF_DL_DSHS_BINDING),

	    // DL_DSplatViews
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pDL_DColorMean2DXBuffer,
	                                                 SBUF_DL_DCOLORS_MEAN2DXS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pDL_DConicMean2DYBuffer,
	                                                 SBUF_DL_DCONICS_MEAN2DYS_BINDING),
	    myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, resource.pDL_DViewOpacityBuffer,
	                                                 SBUF_DL_DVIEW_OPACITIES_BINDING),
	};

	PushConstantData pcData = {
	    .bgColor = roArgs.fwd.bgColor,
	    .splatCount = roArgs.fwd.splatCount,
	    .camFocal = {roArgs.fwd.camera.focalX, roArgs.fwd.camera.focalY},
	    .camResolution = {roArgs.fwd.camera.width, roArgs.fwd.camera.height},
	    .camPos = roArgs.fwd.camera.pos,
	    .camViewMat = roArgs.fwd.camera.viewMat,
	};

	// Descriptors and Push Constants
	pCommandBuffer->CmdPushDescriptorSet(mpPipelineLayout, VK_PIPELINE_BIND_POINT_COMPUTE, 0, descriptorSetWrites);
	pCommandBuffer->CmdPushDescriptorSet(mpPipelineLayout, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, descriptorSetWrites);
	pCommandBuffer->CmdPushConstants(
	    mpPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, //
	    0, sizeof(PushConstantData), &pcData);

	// BackwardCopy
	pCommandBuffer->CmdPipelineBarrier2(
	    {}, {},
	    {
	        resource.pImage0->GetMemoryBarrier2(
	            {
	                // CmdBlit or ForwardCopy
	                mConfig.forwardOutputImage ? VK_PIPELINE_STAGE_2_BLIT_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                0,
	                mConfig.forwardOutputImage ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
	                                           : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	            },
	            {
	                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
	                VK_IMAGE_LAYOUT_GENERAL,
	            }),
	    });
	pCommandBuffer->CmdBindPipeline(mpBackwardCopyPipeline);
	pCommandBuffer->CmdDispatch((roArgs.fwd.camera.width + BACKWARD_COPY_DIM_X - 1) / BACKWARD_COPY_DIM_X,
	                            (roArgs.fwd.camera.height + BACKWARD_COPY_DIM_Y - 1) / BACKWARD_COPY_DIM_Y, 1u);
	// For BackwardDraw, switch storage image slot to image1
	pCommandBuffer->CmdPushDescriptorSet(
	    mpPipelineLayout, VK_PIPELINE_BIND_POINT_GRAPHICS, 0,
	    {
	        myvk::DescriptorSetWrite::WriteStorageImage(nullptr, resource.pImageView1, SIMG_IMAGE0_BINDING),
	    });

	perfQuery.CmdWriteTimestamp(pCommandBuffer, PerfQuery::Timestamp::kBackward);

	// BackwardReset
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pDispatchArgBuffer->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT read (BackwardView)
	            {VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),

	        // DL_DSplatViews
	        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT read (BackwardView)
	        resource.pDL_DColorMean2DXBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pDL_DConicMean2DYBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pDL_DViewOpacityBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	    },
	    {});
	pCommandBuffer->CmdBindPipeline(mpBackwardResetPipeline);
	pCommandBuffer->CmdDispatch(BACKWARD_RESET_GROUP_COUNT, 1, 1);
	perfQuery.CmdWriteTimestamp(pCommandBuffer, PerfQuery::Timestamp::kBackwardReset);

	// Clear Image1
	pCommandBuffer->CmdPipelineBarrier2(
	    {}, {},
	    {
	        resource.pImage1->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT read/write (BackwardDraw)
	            {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL}),
	    });
	pCommandBuffer->CmdClearColorImage(resource.pImage1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                                   {0.0f, 0.0f, 0.0f, 1.0f});

	// BackwardDraw
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        // DL_DSplatViews
	        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT write (BackwardReset)
	        resource.pDL_DColorMean2DXBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pDL_DConicMean2DYBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	        resource.pDL_DViewOpacityBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
	    },
	    {
	        // Barrier for image0 is in SubpassDependency0

	        resource.pImage1->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_CLEAR_BIT write
	            {VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
	            {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL}),
	    });
	pCommandBuffer->CmdBeginRenderPass(mpBackwardRenderPass, resource.pBackwardFramebuffer, {});
	pCommandBuffer->CmdBindPipeline(mpBackwardDrawPipeline);
	pCommandBuffer->CmdSetViewport({VkViewport{
	    .x = 0,
	    .y = 0,
	    .width = (float)roArgs.fwd.camera.width,
	    .height = (float)roArgs.fwd.camera.height,
	    .minDepth = 0.0f,
	    .maxDepth = 1.0f,
	}});
	pCommandBuffer->CmdSetScissor({VkRect2D{
	    .offset = {},
	    .extent = {.width = roArgs.fwd.camera.width, .height = roArgs.fwd.camera.height},
	}});
	pCommandBuffer->CmdDrawIndirect(resource.pDrawArgBuffer, 0, 1);
	perfQuery.CmdWriteTimestamp(pCommandBuffer, PerfQuery::Timestamp::kBackwardDraw);
	pCommandBuffer->CmdEndRenderPass();

	// BackwardView
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {
	        resource.pDispatchArgBuffer->GetMemoryBarrier2(
	            // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT write (BackwardReset)
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT}),

	        // DL_DSplatViews
	        // VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT write (BackwardDraw)
	        resource.pDL_DColorMean2DXBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	        resource.pDL_DConicMean2DYBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	        resource.pDL_DViewOpacityBuffer->GetMemoryBarrier2(
	            {VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
	    },
	    {});
	pCommandBuffer->CmdBindPipeline(mpBackwardViewPipeline);
	pCommandBuffer->CmdDispatchIndirect(resource.pDispatchArgBuffer);
	perfQuery.CmdWriteTimestamp(pCommandBuffer, PerfQuery::Timestamp::kBackwardView);
}

const Rasterizer::FwdRWArgsSyncState &Rasterizer::GetSrcFwdRWArgsSync() {
	static constexpr FwdRWArgsSyncState kSync = {
	    .outPixelBuffer = {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	    .outPixelImage = {VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
	                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL},
	};
	return kSync;
}
const Rasterizer::FwdRWArgsSyncState &Rasterizer::GetDstFwdRWArgsSync() {
	// Identical to GetSrcFwdRWArgsSync
	static constexpr FwdRWArgsSyncState kSync = {
	    .outPixelBuffer = {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	    .outPixelImage = {VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
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
	    .outPixelBuffer = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	    .outPixelImage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
	};
	return kUsage;
}

const Rasterizer::BwdRWArgsSyncState &Rasterizer::GetSrcBwdRWArgsSync() {
	static constexpr BwdRWArgsSyncState kSync = {
	    .dL_dSplatBuffers = {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	};
	return kSync;
}
const Rasterizer::BwdRWArgsSyncState &Rasterizer::GetDstBwdRWArgsSync() {
	static constexpr BwdRWArgsSyncState kSync = {
	    .dL_dSplatBuffers = {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
	};
	return kSync;
}
const Rasterizer::BwdROArgsSyncState &Rasterizer::GetBwdROArgsSync() {
	static BwdROArgsSyncState kSync = {
	    .fwd = GetFwdROArgsSync(),
	    .dL_dPixelBuffer = {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT},
	};
	return kSync;
}
const Rasterizer::BwdArgsUsage &Rasterizer::GetBwdArgsUsage() {
	static BwdArgsUsage kUsage = {
	    .fwd = GetFwdArgsUsage(),
	    .dL_dPixelBuffer = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	    .dL_dSplatBuffers = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	};
	return kUsage;
}

} // namespace vkgsraster

// Check DeviceSorter KeyCountBufferOffset
#undef SBUF_DISPATCH_ARGS_BINDING // Remove warning
#include <shader/DeviceSorter/Size.hpp>
static_assert(KEY_COUNT_BUFFER_OFFSET == offsetof(VkDrawIndirectCommand, instanceCount));
static_assert(KEY_COUNT_BUFFER_OFFSET == SORT_COUNT_BUFFER_OFFSET);
