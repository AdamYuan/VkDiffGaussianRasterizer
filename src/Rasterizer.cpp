//
// Created by adamyuan on 3/17/25.
//

#include "Rasterizer.hpp"

#include <array>
#include <shader/Rasterizer/Size.hpp>

#include "ResourceUtil.hpp"

namespace VkGSRaster {

void Rasterizer::Resource::update(const myvk::Ptr<myvk::Device> &pDevice, uint32_t width, uint32_t height,
                                  uint32_t splatCount, double growFactor) {
	sorterResource.update(pDevice, splatCount, growFactor);

	GrowBuffer<sizeof(uint32_t)>(pDevice, pSortKeyBuffer,
	                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | DeviceSorter::GetArgsUsage().keyBuffer,
	                             splatCount, growFactor);
	GrowBuffer<sizeof(uint32_t)>(pDevice, pSortPayloadBuffer,
	                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | DeviceSorter::GetArgsUsage().payloadBuffer,
	                             splatCount, growFactor);
	GrowBuffer<sizeof(float) * 4>(pDevice, pColorMean2DXBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount,
	                              growFactor);
	GrowBuffer<sizeof(float) * 4>(pDevice, pConicMean2DYBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount,
	                              growFactor);
	GrowBuffer<sizeof(float) * 4>(pDevice, pQuadBuffer, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, splatCount, growFactor);

	MakeBuffer<sizeof(VkDrawIndirectCommand)>(pDevice, pDrawArgBuffer,
	                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
	                                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
	                                              DeviceSorter::GetArgsUsage().countBuffer,
	                                          1);
	ResizeImage<VK_FORMAT_R32G32B32A32_SFLOAT>(pDevice, pColorImage, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, //
	                                           width, height);
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

Rasterizer::Rasterizer(const myvk::Ptr<myvk::Device> &pDevice) : mSorter{pDevice} {
	auto pDescriptorSetLayout = myvk::DescriptorSetLayout::Create( //
	    pDevice,
	    std::vector<VkDescriptorSetLayoutBinding>{
	        {.binding = B_MEANS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_SCALES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_ROTATES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_OPACITIES_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},
	        {.binding = B_SHS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},

	        {.binding = B_SORT_KEYS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_SORT_PAYLOADS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},

	        {.binding = B_COLORS_MEAN2DXS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},
	        {.binding = B_CONICS_MEAN2DYS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},
	        {.binding = B_QUADS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_GEOMETRY_BIT},

	        {.binding = B_DRAW_ARGS_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	        {.binding = B_SORT_COUNT_BINDING,
	         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	         .descriptorCount = 1u,
	         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
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
		    .SetAttachment(0, VK_FORMAT_R32G32B32A32_SFLOAT, {.op = VK_ATTACHMENT_LOAD_OP_CLEAR},
		                   {.op = VK_ATTACHMENT_STORE_OP_STORE, .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL})
		    .SetSubpassCount(1)
		    .SetSubpass(
		        0, {.color_attachment_refs = {{.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}}})
		    .SetDependencyCount(1)
		    .SetSrcExternalDependency(
		        0, {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0},
		        {.subpass = 0,
		         .sync = myvk::GetAttachmentLoadOpSync(VK_IMAGE_ASPECT_COLOR_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR)});
		return state;
	}());

	// Common Shader Module
	auto pDrawVertShader = createDrawVertShader(pDevice);

	// Forward Pipelines
	mpForwardResetPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createForwardResetShader(pDevice));
	mpForwardViewPipeline = myvk::ComputePipeline::Create(mpPipelineLayout, createForwardViewShader(pDevice));
	mpForwardDrawPipeline = myvk::GraphicsPipeline::Create(
	    mpPipelineLayout, mpForwardRenderPass,
	    myvk::GraphicsPipelineShaderModules{.vert = pDrawVertShader,
	                                        .geom = createForwardDrawGeomShader(pDevice),
	                                        .frag = createForwardDrawFragShader(pDevice)},
	    [] {
		    myvk::GraphicsPipelineState state{};
		    state.m_dynamic_state.Enable({VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});
		    state.m_color_blend_state.Enable({VkPipelineColorBlendAttachmentState{
		        .blendEnable = VK_TRUE,
		    }});
		    state.m_viewport_state.Enable();
		    state.m_vertex_input_state.Enable();
		    state.m_input_assembly_state.Enable(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
		    state.m_rasterization_state.Initialize(VK_POLYGON_MODE_FILL, VK_FRONT_FACE_COUNTER_CLOCKWISE,
		                                           VK_CULL_MODE_NONE);
		    state.m_multisample_state.Enable(VK_SAMPLE_COUNT_1_BIT);
		    return state;
	    }(),
	    0);
}

} // namespace VkGSRaster

// Check DeviceSorter KeyCountBufferOffset
#include <shader/DeviceSorter/Size.hpp>
static_assert(KEY_COUNT_BUFFER_OFFSET == offsetof(VkDrawIndirectCommand, instanceCount));
static_assert(KEY_COUNT_BUFFER_OFFSET == SORT_COUNT_BUFFER_OFFSET);
