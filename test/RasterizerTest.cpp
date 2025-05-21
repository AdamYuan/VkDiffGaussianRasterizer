#include "../src/Rasterizer.hpp"
#include "GSModel.hpp"

#include <array>
#include <cmath>
#include <myvk/FrameManager.hpp>
#include <myvk/GLFWHelper.hpp>
#include <myvk/ImGuiHelper.hpp>
#include <myvk/ImGuiRenderer.hpp>
#include <myvk/Instance.hpp>
#include <myvk/Queue.hpp>
#include <myvk/QueueSelector.hpp>
#include <tinyfiledialogs.h>

#include <imgui_internal.h>

constexpr uint32_t kFrameCount = 3, kMinResolution = 512, kMaxResolution = 4096;

template <typename T> static T clamp(T x, T minVal, T maxVal) { return std::max(std::min(x, maxVal), minVal); }

struct Config {
	uint32_t width = 1280, height = 720;
	bool forwardOutputImage = true;
	bool backward = false;

	bool operator==(const Config &) const = default;
} config = {};

struct Camera {
	std::array<float, 3> position = {};
	float focal = 720.0f;
	float yaw = 0.0f, pitch = 0.0f;

	float speed = 1.f, sensitivity = 1.f;

	void Update(GLFWwindow *pWindow) {
		speed = std::max(speed, 0.0f);
		sensitivity = std::max(sensitivity, 0.0f);

		static double prevUpdateTime = glfwGetTime();
		double updateTime = glfwGetTime();
		auto delta = float(updateTime - prevUpdateTime);
		prevUpdateTime = updateTime;

		float deltaSpeed = speed * delta;
		float deltaSensitivity = sensitivity * delta;

		const auto moveForward = [&](const std::array<float, 3> &dir, float coef) {
			position[0] += dir[0] * coef * deltaSpeed;
			position[1] += dir[1] * coef * deltaSpeed;
			position[2] += dir[2] * coef * deltaSpeed;
		};

		static double prevCursorPosX = 0.0, prevCursorPosY = 0.0;
		double cursorPosX, cursorPosY;
		glfwGetCursorPos(pWindow, &cursorPosX, &cursorPosY);

		if (!ImGui::GetCurrentContext()->NavWindow ||
		    (ImGui::GetCurrentContext()->NavWindow->Flags & ImGuiWindowFlags_NoBringToFrontOnFocus)) {
			auto lookDir = GetLookDir(), sideDir = GetSideDir();
			if (glfwGetKey(pWindow, GLFW_KEY_W) == GLFW_PRESS)
				moveForward(lookDir, 1.0f);
			if (glfwGetKey(pWindow, GLFW_KEY_S) == GLFW_PRESS)
				moveForward(lookDir, -1.0f);
			if (glfwGetKey(pWindow, GLFW_KEY_D) == GLFW_PRESS)
				moveForward(sideDir, 1.0f);
			if (glfwGetKey(pWindow, GLFW_KEY_A) == GLFW_PRESS)
				moveForward(sideDir, -1.0f);

			if (glfwGetMouseButton(pWindow, GLFW_MOUSE_BUTTON_LEFT)) {
				float offsetX = float(cursorPosX - prevCursorPosX) * deltaSensitivity;
				float offsetY = float(cursorPosY - prevCursorPosY) * deltaSensitivity;
				yaw -= offsetX;
				pitch -= offsetY;
			}
		}

		prevCursorPosX = cursorPosX, prevCursorPosY = cursorPosY;

		pitch = clamp(pitch, -(float)M_PI_2, (float)M_PI_2);
	}

	std::array<float, 3> GetLookDir() const {
		float cosYaw = std::cos(yaw), sinYaw = std::sin(yaw);
		float cosPitch = std::cos(pitch), sinPitch = std::sin(pitch);
		return {-cosPitch * cosYaw, -sinPitch, -cosPitch * sinYaw};
	}
	std::array<float, 3> GetUpDir() const {
		float cosYaw = std::cos(yaw), sinYaw = std::sin(yaw);
		float cosPitch = std::cos(pitch), sinPitch = std::sin(pitch);
		return {-sinPitch * cosYaw, cosPitch, -sinPitch * sinYaw};
	}
	std::array<float, 3> GetSideDir() const {
		float cosYaw = std::cos(yaw), sinYaw = std::sin(yaw);
		return {-sinYaw, 0, cosYaw};
	}
	std::array<float, 9> GetViewMatrix() const {
		auto lookDir = GetLookDir(), sideDir = GetSideDir(), upDir = GetUpDir();
		return {
		    sideDir[0], upDir[0], lookDir[0], //
		    sideDir[1], upDir[1], lookDir[1], //
		    sideDir[2], upDir[2], lookDir[2], //
		};
	}
} camera = {};

int main() {
	using vkgsraster::Rasterizer;

	GLFWwindow *pWindow = myvk::GLFWCreateWindow("Test", config.width, config.height, false);

	myvk::Ptr<myvk::Device> pDevice;
	myvk::Ptr<myvk::Queue> pGenericQueue;
	myvk::Ptr<myvk::PresentQueue> pPresentQueue;
	{
		auto pInstance = myvk::Instance::CreateWithGlfwExtensions();
		auto pSurface = myvk::Surface::Create(pInstance, pWindow);
		auto pPhysicalDevice = [&] {
			auto pPhysicalDevices = myvk::PhysicalDevice::Fetch(pInstance);
			for (const auto &pPhyDev : pPhysicalDevices) {
				if (pPhyDev->GetProperties().vk10.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
					return pPhyDev;
			}
			return pPhysicalDevices[0];
		}();
		auto features = pPhysicalDevice->GetDefaultFeatures();
		VkPhysicalDeviceShaderAtomicFloatFeaturesEXT shaderAtomicFloatFeature{
		    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
		    .shaderBufferFloat32AtomicAdd = VK_TRUE,
		};
		VkPhysicalDeviceShaderQuadControlFeaturesKHR shaderQuadControlFeature{
		    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR,
		    .pNext = &shaderAtomicFloatFeature,
		    .shaderQuadControl = VK_TRUE,
		};
		VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragShaderInterlockFeature{
		    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
		    .pNext = &shaderQuadControlFeature,
		    .fragmentShaderPixelInterlock = VK_TRUE,
		};
		features.vk12.hostQueryReset = VK_TRUE;
		features.vk13.synchronization2 = VK_TRUE;
		features.vk13.computeFullSubgroups = VK_TRUE;
		features.SetPNext(&fragShaderInterlockFeature);
		pDevice = myvk::Device::Create(
		    pPhysicalDevice, myvk::GenericPresentQueueSelector{&pGenericQueue, pSurface, &pPresentQueue}, features,
		    {
		        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		        VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME,
		        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
		        VK_EXT_LOAD_STORE_OP_NONE_EXTENSION_NAME,
		        VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME,
		        VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME,
		    });
	}

	auto pFrameManager =
	    myvk::FrameManager::Create(pGenericQueue, pPresentQueue, false, kFrameCount,
	                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | Rasterizer::GetFwdArgsUsage().outPixelImage);

	auto pRenderPass = myvk::RenderPass::Create(pDevice, [&] {
		myvk::RenderPassState2 state;
		state.SetAttachmentCount(1)
		    .SetAttachment(
		        0, pFrameManager->GetSwapchain()->GetImageFormat(),
		        {.op = VK_ATTACHMENT_LOAD_OP_LOAD, .layout = Rasterizer::GetDstFwdRWArgsSync().outPixelImage.layout},
		        {.op = VK_ATTACHMENT_STORE_OP_STORE, .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR})
		    .SetSubpassCount(1)
		    .SetSubpass(
		        0, {.color_attachment_refs = {{.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}}})
		    .SetDependencyCount(1)
		    .SetSrcExternalDependency(
		        0, myvk::SyncStateCast<myvk::MemorySyncState>(Rasterizer::GetDstFwdRWArgsSync().outPixelImage),
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

	Rasterizer rasterizer;
	Rasterizer::Resource rasterizerResource;
	myvk::Ptr<myvk::Buffer> pPixelBuffer, pDL_DPixelBuffer;

	const auto updateConfig = [&] {
		static bool isFirstUpdate = true;
		static Config prevConfig = {};

		config.width = clamp(config.width, kMinResolution, kMaxResolution);
		config.height = clamp(config.height, kMinResolution, kMaxResolution);

		if (isFirstUpdate || config != prevConfig) {
			pDevice->WaitIdle();

			rasterizer = Rasterizer{pDevice, {.forwardOutputImage = config.forwardOutputImage}};
			rasterizerResource.UpdateImage(pDevice, config.width, config.height, rasterizer);

			if (config.width != prevConfig.width || config.height != prevConfig.height) {
				glfwSetWindowSize(pWindow, (int)config.width, (int)config.height);
				pFrameManager->Resize();
			}

			VkDeviceSize imageBufferSize = sizeof(float) * 3 * config.width * config.height;
			if (!pPixelBuffer || pPixelBuffer->GetSize() < imageBufferSize)
				pPixelBuffer =
				    myvk::Buffer::Create(pDevice, imageBufferSize, 0, Rasterizer::GetFwdArgsUsage().outPixelBuffer);
			if (!pDL_DPixelBuffer || pDL_DPixelBuffer->GetSize() < imageBufferSize)
				pDL_DPixelBuffer =
				    myvk::Buffer::Create(pDevice, imageBufferSize, 0, Rasterizer::GetBwdArgsUsage().dL_dPixelBuffer);
		}

		isFirstUpdate = false;
		prevConfig = config;
	};

	VkGSModel vkGsModel{};
	Rasterizer::SplatArgs pDL_DSplats;

	std::array<Rasterizer::PerfQuery, kFrameCount> rasterizerPerfQueries;
	for (auto &query : rasterizerPerfQueries) {
		query = Rasterizer::PerfQuery::Create(pDevice);
		query.Reset();
	}

	Rasterizer::PerfMetrics rasterizerPerfMetrics{};

	while (!glfwWindowShouldClose(pWindow)) {
		glfwPollEvents();

		{
			myvk::ImGuiNewFrame();

			ImGui::Begin("Controller");
			ImGui::Text("GPU: %s", pDevice->GetPhysicalDevicePtr()->GetProperties().vk10.deviceName);
			if (ImGui::Button("Load 3DGS")) {
				static constexpr int kFilterCount = 1;
				static constexpr const char *kFilterPatterns[kFilterCount] = {"*.ply"};
				const char *filename =
				    tinyfd_openFileDialog("Open 3DGS Model", "", kFilterCount, kFilterPatterns, nullptr, false);
				if (filename) {
					pGenericQueue->WaitIdle();
					auto gsModel = GSModel::Load(filename);
					if (!gsModel.IsEmpty()) {
						vkGsModel = {};
						vkGsModel = VkGSModel::Create(pGenericQueue, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, gsModel);
						rasterizerResource.UpdateBuffer(pDevice, vkGsModel.splatCount);
						pDL_DSplats = VkGSModel::Create(pDevice, Rasterizer::GetBwdArgsUsage().dL_dSplatBuffers,
						                                vkGsModel.splatCount)
						                  .GetSplatArgs();
					}
				}
			}
			ImGui::Text("3DGS Splat Count: %u", vkGsModel.splatCount);

			if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::InputFloat3("Position", camera.position.data());
				ImGui::InputFloat("Focal", &camera.focal);
				ImGui::InputFloat("Yaw", &camera.yaw);
				ImGui::InputFloat("Pitch", &camera.pitch);
				ImGui::InputFloat("Speed", &camera.speed);
				ImGui::InputFloat("Sensitivity", &camera.sensitivity);
			}
			if (ImGui::CollapsingHeader("Config", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::Checkbox("Backward", &config.backward);
				ImGui::Checkbox("Output Image", &config.forwardOutputImage);
				{
					static int sWidthHeight[] = {(int)config.width, (int)config.height};
					ImGui::InputInt2("Resolution", sWidthHeight);
					sWidthHeight[0] = std::max(sWidthHeight[0], (int)kMinResolution);
					sWidthHeight[1] = std::max(sWidthHeight[1], (int)kMinResolution);
					if (ImGui::Button("Resize")) {
						config.width = (uint32_t)sWidthHeight[0];
						config.height = (uint32_t)sWidthHeight[1];
					}
				}
			}
			if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::Text("Forward: %lf ms", rasterizerPerfMetrics.forward);
				ImGui::Text("Forward View: %lf ms", rasterizerPerfMetrics.forwardView);
				ImGui::Text("Forward Sort: %lf ms", rasterizerPerfMetrics.forwardSort);
				ImGui::Text("Forward Draw: %lf ms", rasterizerPerfMetrics.forwardDraw);
				ImGui::Text("Backward: %lf ms", rasterizerPerfMetrics.backward);
				ImGui::Text("Backward Reset: %lf ms", rasterizerPerfMetrics.backwardReset);
				ImGui::Text("Backward Draw: %lf ms", rasterizerPerfMetrics.backwardDraw);
				ImGui::Text("Backward View: %lf ms", rasterizerPerfMetrics.backwardView);
			}
			ImGui::End();

			ImGui::Render();
		}

		updateConfig();
		camera.Update(pWindow);

		if (pFrameManager->NewFrame()) {
			uint32_t currentFrame = pFrameManager->GetCurrentFrame();
			const auto &pCommandBuffer = pFrameManager->GetCurrentCommandBuffer();
			const auto &pSwapchainImage = pFrameManager->GetCurrentSwapchainImage();
			const auto &pSwapchainImageView = pFrameManager->GetCurrentSwapchainImageView();
			const auto &rasterizerPerfQuery = rasterizerPerfQueries[currentFrame];

			rasterizerPerfMetrics = rasterizerPerfQuery.GetMetrics();
			rasterizerPerfQuery.Reset();

			pCommandBuffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

			pCommandBuffer->CmdPipelineBarrier2(
			    {}, {},
			    {
			        pSwapchainImage->GetMemoryBarrier2({VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT},
			                                           Rasterizer::GetSrcFwdRWArgsSync().outPixelImage),
			    });

			if (!vkGsModel.IsEmpty()) {
				if (!config.forwardOutputImage)
					pCommandBuffer->CmdPipelineBarrier2(
					    {},
					    {pPixelBuffer->GetMemoryBarrier2(Rasterizer::GetDstFwdRWArgsSync().outPixelBuffer.GetWrite(),
					                                     Rasterizer::GetSrcFwdRWArgsSync().outPixelBuffer)},
					    {});

				Rasterizer::FwdROArgs fwdROArgs = {
				    .camera =
				        {
				            .width = pSwapchainImage->GetExtent().width,
				            .height = pSwapchainImage->GetExtent().height,
				            .focalX = camera.focal,
				            .focalY = camera.focal,
				            .viewMat = camera.GetViewMatrix(),
				            .pos = camera.position,
				        },
				    .splatCount = vkGsModel.splatCount,
				    .splats = vkGsModel.GetSplatArgs(),
				    .bgColor = {1.0f, 1.0f, 1.0f},
				};
				rasterizer.CmdForward(pCommandBuffer, fwdROArgs,
				                      {
				                          .pOutPixelBuffer = pPixelBuffer,
				                          .pOutPixelImage = pSwapchainImage,
				                      },
				                      rasterizerResource, rasterizerPerfQuery);

				if (config.backward) {
					pCommandBuffer->CmdPipelineBarrier2(
					    {},
					    {
					        pDL_DSplats.pMeanBuffer->GetMemoryBarrier2(
					            Rasterizer::GetDstBwdRWArgsSync().dL_dSplatBuffers.GetWrite(),
					            Rasterizer::GetSrcBwdRWArgsSync().dL_dSplatBuffers),
					        pDL_DSplats.pScaleBuffer->GetMemoryBarrier2(
					            Rasterizer::GetDstBwdRWArgsSync().dL_dSplatBuffers.GetWrite(),
					            Rasterizer::GetSrcBwdRWArgsSync().dL_dSplatBuffers),
					        pDL_DSplats.pRotateBuffer->GetMemoryBarrier2(
					            Rasterizer::GetDstBwdRWArgsSync().dL_dSplatBuffers.GetWrite(),
					            Rasterizer::GetSrcBwdRWArgsSync().dL_dSplatBuffers),
					        pDL_DSplats.pOpacityBuffer->GetMemoryBarrier2(
					            Rasterizer::GetDstBwdRWArgsSync().dL_dSplatBuffers.GetWrite(),
					            Rasterizer::GetSrcBwdRWArgsSync().dL_dSplatBuffers),
					        pDL_DSplats.pSHBuffer->GetMemoryBarrier2(
					            Rasterizer::GetDstBwdRWArgsSync().dL_dSplatBuffers.GetWrite(),
					            Rasterizer::GetSrcBwdRWArgsSync().dL_dSplatBuffers),
					    },
					    {});
					rasterizer.CmdBackward(pCommandBuffer,
					                       {
					                           .fwd = fwdROArgs,
					                           .pdL_dPixelBuffer = pDL_DPixelBuffer,
					                       },
					                       {
					                           .dL_dSplats = pDL_DSplats,
					                       },
					                       rasterizerResource, rasterizerPerfQuery);
				}
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
