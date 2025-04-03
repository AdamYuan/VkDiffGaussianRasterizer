#include "../src/DeviceSorter.hpp"

#include <random>

#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>
#include <myvk/ImGuiRenderer.hpp>
#include <myvk/Instance.hpp>
#include <myvk/Queue.hpp>
#include <myvk/QueueSelector.hpp>

#include <shader/DeviceSorter/Size.hpp>
#include <shader/DeviceSorterTest/Size.hpp>

constexpr uint32_t kFrameCount = 3;
constexpr uint32_t kMaxSortKeyCount = 1000000;

int main() {
	GLFWwindow *pWindow = myvk::GLFWCreateWindow("Test", 1280, 720, true);

	myvk::Ptr<myvk::Device> pDevice;
	myvk::Ptr<myvk::Queue> pGenericQueue;
	myvk::Ptr<myvk::PresentQueue> pPresentQueue;
	{
		auto pInstance = myvk::Instance::CreateWithGlfwExtensions();
		auto pSurface = myvk::Surface::Create(pInstance, pWindow);
		auto pPhysicalDevice = myvk::PhysicalDevice::Fetch(pInstance)[0];
		auto features = pPhysicalDevice->GetDefaultFeatures();
		features.vk13.synchronization2 = VK_TRUE;
		features.vk13.computeFullSubgroups = VK_TRUE;
		pDevice = myvk::Device::Create(
		    pPhysicalDevice, myvk::GenericPresentQueueSelector{&pGenericQueue, pSurface, &pPresentQueue}, features,
		    {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
	}

	auto pFrameManager = myvk::FrameManager::Create(pGenericQueue, pPresentQueue, false, kFrameCount);

	auto pRenderPass = myvk::RenderPass::Create(pDevice, [&] {
		myvk::RenderPassState2 state;
		state.SetAttachmentCount(1)
		    .SetAttachment(0, pFrameManager->GetSwapchain()->GetImageFormat(), {.op = VK_ATTACHMENT_LOAD_OP_CLEAR},
		                   {.op = VK_ATTACHMENT_STORE_OP_STORE, .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR})
		    .SetSubpassCount(1)
		    .SetSubpass(
		        0, {.color_attachment_refs = {{.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}}})
		    .SetDependencyCount(1)
		    .SetSrcExternalDependency(
		        0, {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0},
		        {.subpass = 0,
		         .sync = myvk::GetAttachmentLoadOpSync(VK_IMAGE_ASPECT_COLOR_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR)});
		return state;
	}());

	myvk::ImGuiInit(pWindow, myvk::CommandPool::Create(pGenericQueue));

	auto pImGuiRenderer = myvk::ImGuiRenderer::Create(pRenderPass, 0, kFrameCount);

	auto pFramebuffer = myvk::ImagelessFramebuffer::Create(pRenderPass, {pFrameManager->GetSwapchainImageViews()[0]});
	pFrameManager->SetResizeFunc([&](const VkExtent2D &) {
		pFramebuffer = myvk::ImagelessFramebuffer::Create(pRenderPass, {pFrameManager->GetSwapchainImageViews()[0]});
	});

	using vkgsraster::DeviceSorter;
	DeviceSorter sorter{pDevice, {.useKeyAsPayload = true}};
	DeviceSorter::Resource sorterResource;

	sorterResource.Update(pDevice, kMaxSortKeyCount);

	auto pKeyBuffer = myvk::Buffer::Create(pDevice, kMaxSortKeyCount * sizeof(uint32_t), 0,
	                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | DeviceSorter::GetArgsUsage().keyBuffer);
	auto pPayloadBuffer =
	    myvk::Buffer::Create(pDevice, kMaxSortKeyCount * sizeof(uint32_t), 0,
	                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | DeviceSorter::GetArgsUsage().payloadBuffer);
	std::array<myvk::Ptr<myvk::Buffer>, kFrameCount> pCountBuffers;
	for (auto &pCountBuffer : pCountBuffers)
		pCountBuffer = myvk::Buffer::Create(pDevice, KEY_COUNT_BUFFER_OFFSET + sizeof(uint32_t),
		                                    VMA_ALLOCATION_CREATE_MAPPED_BIT |
		                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                                    DeviceSorter::GetArgsUsage().countBuffer);

	std::array<uint32_t *, kFrameCount> pMappedCounts{};
	for (std::size_t i = 0; i < kFrameCount; ++i)
		pMappedCounts[i] = reinterpret_cast<uint32_t *>(pCountBuffers[i]->GetMappedData()) +
		                   (KEY_COUNT_BUFFER_OFFSET / sizeof(uint32_t));
	static_assert(KEY_COUNT_BUFFER_OFFSET % sizeof(uint32_t) == 0);

	std::array<myvk::Ptr<myvk::Buffer>, kFrameCount> pFailBuffers;
	for (auto &pFailBuffer : pFailBuffers)
		pFailBuffer = myvk::Buffer::Create(
		    pDevice, sizeof(uint32_t), VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
		    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	std::array<uint32_t *, kFrameCount> pMappedFails{};
	for (std::size_t i = 0; i < kFrameCount; ++i)
		pMappedFails[i] = reinterpret_cast<uint32_t *>(pFailBuffers[i]->GetMappedData());

	auto pGeneratePipeline = myvk::ComputePipeline::Create(
	    myvk::PipelineLayout::Create(
	        pDevice,
	        {myvk::DescriptorSetLayout::Create(pDevice,
	                                           {
	                                               {.binding = 0,
	                                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                                                .descriptorCount = 1u,
	                                                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	                                           },
	                                           VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR)},
	        {VkPushConstantRange{
	            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 2 * sizeof(uint32_t)}}),
	    [&] {
		    static constexpr uint32_t kCode[] = {
#include <shader/DeviceSorterTest/Generate.comp.inl>

		    };
		    return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
	    }());

	auto pValidatePipeline = myvk::ComputePipeline::Create(
	    myvk::PipelineLayout::Create(
	        pDevice,
	        {myvk::DescriptorSetLayout::Create(pDevice,
	                                           {
	                                               {.binding = 0,
	                                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                                                .descriptorCount = 1u,
	                                                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	                                               {.binding = 1,
	                                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                                                .descriptorCount = 1u,
	                                                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
	                                           },
	                                           VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR)},
	        {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(uint32_t)}}),
	    [&] {
		    static constexpr uint32_t kCode[] = {
#include <shader/DeviceSorterTest/Validate.comp.inl>

		    };
		    return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
	    }());

	std::mt19937 randGen{std::random_device{}()};

	uint32_t failCount = 0, keyCount = kMaxSortKeyCount;

	while (!glfwWindowShouldClose(pWindow)) {
		glfwPollEvents();

		{
			myvk::ImGuiNewFrame();

			ImGui::Begin("Info");
			ImGui::DragInt("Key Count", reinterpret_cast<int *>(&keyCount), 1, 0, kMaxSortKeyCount);
			keyCount = std::min(keyCount, kMaxSortKeyCount);
			ImGui::Text("Fail Count: %d", failCount);
			ImGui::End();

			ImGui::Render();
		}

		if (pFrameManager->NewFrame()) {
			uint32_t currentFrame = pFrameManager->GetCurrentFrame();
			const auto &pCommandBuffer = pFrameManager->GetCurrentCommandBuffer();

			uint32_t *pMappedCount = pMappedCounts[currentFrame];
			const auto &pCountBuffer = pCountBuffers[currentFrame];
			uint32_t *pMappedFail = pMappedFails[currentFrame];
			const auto &pFailBuffer = pFailBuffers[currentFrame];

			pMappedCount[0] = keyCount;
			failCount += pMappedFail[0];
			pMappedFail[0] = 0;

			pCommandBuffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

			pCommandBuffer->CmdPipelineBarrier2(
			    {},
			    {
			        pKeyBuffer->GetMemoryBarrier2(
			            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0},
			            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT}),
			        pPayloadBuffer->GetMemoryBarrier2(DeviceSorter::GetDstRWArgsSync().payloadBuffer,
			                                          DeviceSorter::GetSrcRWArgsSync().payloadBuffer),
			    },
			    {});

			uint32_t pcData[2] = {keyCount, std::uniform_int_distribution<uint32_t>{}(randGen)};

			pCommandBuffer->CmdBindPipeline(pGeneratePipeline);
			pCommandBuffer->CmdPushConstants(pGeneratePipeline->GetPipelineLayoutPtr(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
			                                 2 * sizeof(uint32_t), &pcData);
			pCommandBuffer->CmdPushDescriptorSet(
			    pGeneratePipeline->GetPipelineLayoutPtr(), VK_PIPELINE_BIND_POINT_COMPUTE, 0,
			    {
			        myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pKeyBuffer, 0),
			    });
			pCommandBuffer->CmdDispatch((keyCount + GENERATE_DIM - 1) / GENERATE_DIM, 1, 1);

			pCommandBuffer->CmdPipelineBarrier2(
			    {},
			    {
			        pKeyBuffer->GetMemoryBarrier2(
			            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT},
			            DeviceSorter::GetSrcRWArgsSync().keyBuffer),
			    },
			    {});

			sorter.CmdSort(pCommandBuffer,
			               {
			                   .pCountBuffer = pCountBuffer,
			               },
			               {
			                   .pKeyBuffer = pKeyBuffer,
			                   .pPayloadBuffer = pPayloadBuffer,
			               },
			               sorterResource);

			pCommandBuffer->CmdPipelineBarrier2(
			    {},
			    {
			        pKeyBuffer->GetMemoryBarrier2(
			            DeviceSorter::GetDstRWArgsSync().keyBuffer,
			            {VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT}),
			    },
			    {});

			pCommandBuffer->CmdBindPipeline(pValidatePipeline);
			pCommandBuffer->CmdPushConstants(pValidatePipeline->GetPipelineLayoutPtr(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
			                                 sizeof(uint32_t), &pcData);
			pCommandBuffer->CmdPushDescriptorSet(
			    pValidatePipeline->GetPipelineLayoutPtr(), VK_PIPELINE_BIND_POINT_COMPUTE, 0,
			    {
			        myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pKeyBuffer, 0),
			        myvk::DescriptorSetWrite::WriteStorageBuffer(nullptr, pFailBuffer, 1),
			    });
			pCommandBuffer->CmdDispatch((keyCount + VALIDATE_DIM - 1) / VALIDATE_DIM, 1, 1);

			pCommandBuffer->CmdPipelineBarrier2(
			    {}, {},
			    {pFrameManager->GetCurrentSwapchainImage()->GetMemoryBarrier2(
			        VK_IMAGE_ASPECT_COLOR_BIT,
			        myvk::ImageSyncState{.stage_mask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
			        myvk::ImageSyncState{.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR})});

			pCommandBuffer->CmdBeginRenderPass(pRenderPass, pFramebuffer,
			                                   {pFrameManager->GetCurrentSwapchainImageView()},
			                                   {{{0.5f, 0.5f, 0.5f, 1.0f}}});
			pImGuiRenderer->CmdDrawPipeline(pCommandBuffer, currentFrame);
			pCommandBuffer->CmdEndRenderPass();

			pCommandBuffer->End();

			pFrameManager->Render();
		}
	}

	pDevice->WaitIdle();
	glfwTerminate();
	return 0;
}
