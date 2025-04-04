#include "../src/Rasterizer.hpp"
#include "CuImageWrite.hpp"
#include "CuTileRasterizer.hpp"
#include "GSDataset.hpp"
#include "GSModel.hpp"
#include "VkCuBuffer.hpp"

#include <myvk/ExternalMemoryUtil.hpp>
#include <myvk/Instance.hpp>
#include <myvk/Queue.hpp>
#include <myvk/QueueSelector.hpp>

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 2) {
		printf("./PerformanceTest [3dgs.ply] [cameras.json]\n");
		return EXIT_FAILURE;
	}

	myvk::Ptr<myvk::Device> pDevice;
	myvk::Ptr<myvk::Queue> pGenericQueue;
	{
		auto pInstance = myvk::Instance::Create({});
		auto pPhysicalDevice = myvk::PhysicalDevice::Fetch(pInstance)[0];
		auto features = pPhysicalDevice->GetDefaultFeatures();
		VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT fragShaderInterlockFeature{
		    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
		    .fragmentShaderPixelInterlock = VK_TRUE,
		};
		features.vk13.synchronization2 = VK_TRUE;
		features.vk13.computeFullSubgroups = VK_TRUE;
		features.SetPNext(&fragShaderInterlockFeature);
		pDevice = myvk::Device::Create(pPhysicalDevice, myvk::GenericQueueSelector{&pGenericQueue}, features,
		                               {
		                                   VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		                                   VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME,
		                                   myvk::GetExternalMemoryExtensionName(),
		                               });
	}
	auto pCommandPool = myvk::CommandPool::Create(pGenericQueue, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	auto pCommandBuffer = myvk::CommandBuffer::Create(pCommandPool);
	auto pFence = myvk::Fence::Create(pDevice);

	const auto createVkCuBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage) {
		return VkCuBuffer::Create(pDevice, size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	};
	VkGSModel vkGsModel = VkGSModel::Create(pGenericQueue, vkgsraster::Rasterizer::GetFwdArgsUsage().splatBuffers,
	                                        GSModel::Load(argv[0]), createVkCuBuffer);
	GSDataset gsDataset = GSDataset::Load(argv[1]);

	if (vkGsModel.IsEmpty()) {
		printf("Invalid 3DGS Model %s\n", argv[0]);
		return EXIT_FAILURE;
	}
	if (gsDataset.IsEmpty()) {
		printf("Empty Dataset %s\n", argv[0]);
		return EXIT_FAILURE;
	}

	printf("splatCount = %d\n", vkGsModel.splatCount);

	vkgsraster::Rasterizer vkRasterizer{pDevice, {.forwardOutputImage = false}};
	vkgsraster::Rasterizer::Resource vkRasterResource = {};
	vkRasterResource.UpdateBuffer(pDevice, vkGsModel.splatCount);
	vkgsraster::Rasterizer::PerfQuery vkRasterPerfQuery = vkgsraster::Rasterizer::PerfQuery::Create(pDevice);
	vkgsraster::Rasterizer::FwdROArgs vkRasterFwdROArgs = {
	    .splatCount = vkGsModel.splatCount,
	    .splats = vkGsModel.GetSplatArgs(),
	    .bgColor = {1.0f, 1.0f, 1.0f},
	};
	vkgsraster::Rasterizer::FwdRWArgs vkRasterFwdRWArgs;
	vkgsraster::Rasterizer::BwdROArgs vkRasterBwdROArgs = {
	    .fwd = vkRasterFwdROArgs,
	};
	vkgsraster::Rasterizer::BwdRWArgs vkRasterBwdRWArgs = {
	    .dL_dSplats = VkGSModel::Create(pDevice, 0 /* TODO */, vkGsModel.splatCount, createVkCuBuffer).GetSplatArgs(),
	};

	CuTileRasterizer::Resource cuTileRasterResource{};
	CuTileRasterizer::FwdROArgs cuTileRasterFwdROArgs{};
	CuTileRasterizer::FwdRWArgs cuTileRasterFwdRWArgs{};
	CuTileRasterizer::BwdROArgs cuTileRasterBwdROArgs{};
	CuTileRasterizer::BwdRWArgs cuTileRasterBwdRWArgs{};
	CuTileRasterizer::PerfQuery cuTileRasterPerfQuery = CuTileRasterizer::PerfQuery::Create();

	for (const auto &entry : gsDataset.entries) {
		vkRasterResource.UpdateImage(pDevice, entry.camera.width, entry.camera.height, vkRasterizer);
		vkRasterFwdROArgs.camera = entry.camera;
		std::size_t colorBufferSize = 3 * entry.camera.width * entry.camera.height * sizeof(float);
		if (!vkRasterFwdRWArgs.pOutColorBuffer || vkRasterFwdRWArgs.pOutColorBuffer->GetSize() < colorBufferSize) {
			vkRasterFwdRWArgs.pOutColorBuffer =
			    VkCuBuffer::Create(pDevice, colorBufferSize, vkgsraster::Rasterizer::GetFwdArgsUsage().outColorBuffer,
			                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}
		if (!vkRasterBwdROArgs.pdL_dColorBuffer || vkRasterBwdROArgs.pdL_dColorBuffer->GetSize() < colorBufferSize) {
			vkRasterBwdROArgs.pdL_dColorBuffer =
			    VkCuBuffer::Create(pDevice, colorBufferSize, 0 /* TODO */, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}

		cuTileRasterFwdROArgs.Update(vkRasterFwdROArgs);
		cuTileRasterFwdRWArgs.Update(vkRasterFwdRWArgs);
		cuTileRasterBwdROArgs.Update(cuTileRasterFwdROArgs, vkRasterBwdROArgs);
		cuTileRasterBwdRWArgs.Update(vkRasterBwdRWArgs);
		float *cuOutColors = cuTileRasterFwdRWArgs.outColors;

		CuTileRasterizer::Forward(cuTileRasterFwdROArgs, cuTileRasterFwdRWArgs, cuTileRasterResource);
		CuTileRasterizer::Forward(cuTileRasterFwdROArgs, cuTileRasterFwdRWArgs, cuTileRasterResource,
		                          cuTileRasterPerfQuery);

		CuTileRasterizer::Backward(cuTileRasterBwdROArgs, cuTileRasterBwdRWArgs, cuTileRasterResource);
		CuTileRasterizer::Backward(cuTileRasterBwdROArgs, cuTileRasterBwdRWArgs, cuTileRasterResource,
		                           cuTileRasterPerfQuery);

		auto cuTileRasterPerfMetrics = cuTileRasterPerfQuery.GetMetrics();
		printf("cu_forward: %lf ms (numRendered = %d)\n", cuTileRasterPerfMetrics.forward,
		       cuTileRasterResource.numRendered);
		printf("cu_backward: %lf ms\n", cuTileRasterPerfMetrics.backward);
		CuImageWrite::Write(entry.imageName + "_cu.png", cuOutColors, entry.camera.width, entry.camera.height);

		pCommandBuffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
		vkRasterizer.CmdForward(pCommandBuffer, vkRasterFwdROArgs, vkRasterFwdRWArgs, vkRasterResource,
		                        vkRasterPerfQuery);
		pCommandBuffer->End();
		pCommandBuffer->Submit(pFence);
		pFence->Wait();
		pCommandPool->Reset();
		pFence->Reset();

		auto vkRasterPerfMetrics = vkRasterPerfQuery.GetMetrics();
		printf("vk_forward: %lf ms\n", vkRasterPerfMetrics.forward);
		CuImageWrite::Write(entry.imageName + "_vk.png", cuOutColors, entry.camera.width, entry.camera.height);

		break; // Only once entry
	}
	return 0;
}
