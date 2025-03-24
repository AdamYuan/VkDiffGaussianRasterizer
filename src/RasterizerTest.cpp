#include "Rasterizer.hpp"

#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>
#include <myvk/ImGuiRenderer.hpp>
#include <myvk/Instance.hpp>
#include <myvk/Queue.hpp>
#include <myvk/QueueSelector.hpp>

constexpr uint32_t kFrameCount = 3, kWidth = 720, kHeight = 720;
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
		myvk::RenderPassState state{1, 1};
		state.RegisterAttachment(0, "color_attachment", pFrameManager->GetSwapchain()->GetImageFormat(),
		                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_SAMPLE_COUNT_1_BIT,
		                         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE);
		state.RegisterSubpass(0, "gui_pass").AddDefaultColorAttachment("color_attachment", nullptr);
		return state;
	}());

	myvk::ImGuiInit(pWindow, myvk::CommandPool::Create(pGenericQueue));

	auto pImGuiRenderer = myvk::ImGuiRenderer::Create(pRenderPass, 0, kFrameCount);

	auto pFramebuffer = myvk::ImagelessFramebuffer::Create(pRenderPass, {pFrameManager->GetSwapchainImageViews()[0]});
	pFrameManager->SetResizeFunc([&](const VkExtent2D &) {
		pFramebuffer = myvk::ImagelessFramebuffer::Create(pRenderPass, {pFrameManager->GetSwapchainImageViews()[0]});
	});

	using VkGSRaster::Rasterizer;
	Rasterizer rasterizer{pDevice};
	Rasterizer::Resource rasterizerResource;

	rasterizerResource.update(pDevice, kWidth, kHeight, kMaxSortKeyCount);

	while (!glfwWindowShouldClose(pWindow)) {
		glfwPollEvents();

		{
			myvk::ImGuiNewFrame();

			ImGui::Begin("Info");
			ImGui::End();

			ImGui::Render();
		}

		if (pFrameManager->NewFrame()) {
			uint32_t currentFrame = pFrameManager->GetCurrentFrame();
			const auto &pCommandBuffer = pFrameManager->GetCurrentCommandBuffer();

			pCommandBuffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

			pCommandBuffer->CmdBeginRenderPass(pRenderPass, {pFramebuffer},
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
