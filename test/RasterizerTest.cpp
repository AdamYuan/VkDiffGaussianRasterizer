#include "../src/Rasterizer.hpp"
#include "GSModel.hpp"

#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>
#include <myvk/ImGuiRenderer.hpp>
#include <myvk/Instance.hpp>
#include <myvk/Queue.hpp>
#include <myvk/QueueSelector.hpp>
#include <tinyfiledialogs.h>

constexpr uint32_t kFrameCount = 3, kWidth = 1280, kHeight = 720;
constexpr uint32_t kMaxSortKeyCount = 1000000;

int main() {
	using VkGSRaster::Rasterizer;

	GLFWwindow *pWindow = myvk::GLFWCreateWindow("Test", kWidth, kHeight, false);

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
		    {
		        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		        VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME,
		    });
	}

	auto pFrameManager =
	    myvk::FrameManager::Create(pGenericQueue, pPresentQueue, false, kFrameCount,
	                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | Rasterizer::GetFwdArgsUsage().outColorImage);

	auto pRenderPass = myvk::RenderPass::Create(pDevice, [&] {
		myvk::RenderPassState2 state;
		state.SetAttachmentCount(1)
		    .SetAttachment(
		        0, pFrameManager->GetSwapchain()->GetImageFormat(),
		        {.op = VK_ATTACHMENT_LOAD_OP_LOAD, .layout = Rasterizer::GetDstFwdRWArgsSync().outColorImage.layout},
		        {.op = VK_ATTACHMENT_STORE_OP_STORE, .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR})
		    .SetSubpassCount(1)
		    .SetSubpass(
		        0, {.color_attachment_refs = {{.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}}})
		    .SetDependencyCount(1)
		    .SetSrcExternalDependency(
		        0, myvk::SyncStateCast<myvk::MemorySyncState>(Rasterizer::GetDstFwdRWArgsSync().outColorImage),
		        {.subpass = 0,
		         .sync = myvk::GetAttachmentLoadOpSync(VK_IMAGE_ASPECT_COLOR_BIT, VK_ATTACHMENT_LOAD_OP_LOAD)});
		return state;
	}());

	myvk::ImGuiInit(pWindow, myvk::CommandPool::Create(pGenericQueue));

	auto pImGuiRenderer = myvk::ImGuiRenderer::Create(pRenderPass, 0, kFrameCount);

	auto pFramebuffer = myvk::ImagelessFramebuffer::Create(pRenderPass, {pFrameManager->GetSwapchainImageViews()[0]});
	pFrameManager->SetResizeFunc([&](const VkExtent2D &) {
		pFramebuffer = myvk::ImagelessFramebuffer::Create(pRenderPass, {pFrameManager->GetSwapchainImageViews()[0]});
	});

	Rasterizer rasterizer{pDevice, {.forwardOutputImage = true}};
	Rasterizer::Resource rasterizerResource;

	rasterizerResource.updateImage(pDevice, kWidth, kHeight, rasterizer);

	VkGSModel vkGsModel{};

	while (!glfwWindowShouldClose(pWindow)) {
		glfwPollEvents();

		{
			myvk::ImGuiNewFrame();

			ImGui::Begin("Info");
			if (ImGui::Button("Load")) {

				static constexpr int kFilterCount = 1;
				static constexpr const char *kFilterPatterns[kFilterCount] = {"*.ply"};
				const char *filename =
				    tinyfd_openFileDialog("Open GS Model", "", kFilterCount, kFilterPatterns, nullptr, false);
				if (filename) {
					vkGsModel =
					    VkGSModel::Create(pGenericQueue, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, GSModel::Load(filename));
					if (!vkGsModel.IsEmpty())
						rasterizerResource.updateBuffer(pDevice, vkGsModel.splatCount);
				}
			}
			ImGui::Text("Splat Count: %u", vkGsModel.splatCount);
			ImGui::End();

			ImGui::Render();
		}

		if (pFrameManager->NewFrame()) {
			uint32_t currentFrame = pFrameManager->GetCurrentFrame();
			const auto &pCommandBuffer = pFrameManager->GetCurrentCommandBuffer();
			const auto &pSwapchainImage = pFrameManager->GetCurrentSwapchainImage();
			const auto &pSwapchainImageView = pFrameManager->GetCurrentSwapchainImageView();

			pCommandBuffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

			pCommandBuffer->CmdPipelineBarrier2(
			    {}, {},
			    {
			        pSwapchainImage->GetMemoryBarrier2({VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
			                                           Rasterizer::GetSrcFwdRWArgsSync().outColorImage),
			    });

			if (!vkGsModel.IsEmpty()) {
				rasterizer.CmdForward(pCommandBuffer,
				                      {
				                          .camera =
				                              {
				                                  .width = kWidth,
				                                  .height = kHeight,
				                                  .focalX = float(kHeight),
				                                  .focalY = float(kHeight),
				                                  .viewMat = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f},
				                                  .pos = {0.0f, 0.0f, 0.0f},
				                              },
				                          .splats =
				                              {
				                                  .count = vkGsModel.splatCount,
				                                  .pMeanBuffer = vkGsModel.pMeanBuffer,
				                                  .pScaleBuffer = vkGsModel.pScaleBuffer,
				                                  .pRotateBuffer = vkGsModel.pRotateBuffer,
				                                  .pOpacityBuffer = vkGsModel.pOpacityBuffer,
				                                  .pSHBuffer = vkGsModel.pSHBuffer,
				                              },
				                          .bgColor = {1.0f, 1.0f, 1.0f},
				                      },
				                      {
				                          .pOutColorImage = pSwapchainImage,
				                      },
				                      rasterizerResource);
			}

			pCommandBuffer->CmdBeginRenderPass(pRenderPass, pFramebuffer, {pSwapchainImageView}, {{{}}});
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
