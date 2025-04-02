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

	VkGSModel vkGsModel =
	    VkGSModel::Create(pGenericQueue, VkGSRaster::Rasterizer::GetFwdArgsUsage().splatBuffers, GSModel::Load(argv[0]),
	                      [&](VkDeviceSize size, VkBufferUsageFlags usage) {
		                      return VkCuBuffer::Create(pDevice, size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	                      });
	GSDataset gsDataset = GSDataset::Load(argv[1]);

	if (vkGsModel.IsEmpty()) {
		printf("Invalid 3DGS Model %s\n", argv[0]);
		return EXIT_FAILURE;
	}
	if (gsDataset.IsEmpty()) {
		printf("Empty Dataset %s\n", argv[0]);
		return EXIT_FAILURE;
	}

	VkGSRaster::Rasterizer vkRasterizer{pDevice, {.forwardOutputImage = false}};
	VkGSRaster::Rasterizer::Resource vkRasterResource = {};
	vkRasterResource.updateBuffer(pDevice, vkGsModel.splatCount);
	VkGSRaster::Rasterizer::FwdROArgs vkRasterFwdROArgs = {
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
	};
	VkGSRaster::Rasterizer::FwdRWArgs vkRasterFwdRWArgs;

	CuTileRasterizer::Resource cuTileRasterResource{};
	CuTileRasterizer::FwdROArgs cuTileRasterFwdROArgs{};
	CuTileRasterizer::FwdRWArgs cuTileRasterFwdRWArgs{};

	for (const auto &entry : gsDataset.entries) {
		vkRasterResource.updateImage(pDevice, entry.camera.width, entry.camera.height, vkRasterizer);
		vkRasterFwdROArgs.camera = entry.camera;
		std::size_t outColorBufferSize = 3 * entry.camera.width * entry.camera.height * sizeof(float);
		if (!vkRasterFwdRWArgs.pOutColorBuffer || vkRasterFwdRWArgs.pOutColorBuffer->GetSize() < outColorBufferSize) {
			vkRasterFwdRWArgs.pOutColorBuffer = VkCuBuffer::Create(
			    pDevice, outColorBufferSize, VkGSRaster::Rasterizer::GetFwdArgsUsage().outColorBuffer,
			    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}

		cuTileRasterFwdROArgs.Update(vkRasterFwdROArgs);
		cuTileRasterFwdRWArgs.Update(vkRasterFwdRWArgs);
		float *cuOutColors = cuTileRasterFwdRWArgs.outColor;

		pCommandBuffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
		vkRasterizer.CmdForward(pCommandBuffer, vkRasterFwdROArgs, vkRasterFwdRWArgs, vkRasterResource);
		pCommandBuffer->End();
		pCommandBuffer->Submit(pFence);
		pFence->Wait();
		pCommandPool->Reset();
		pFence->Reset();

		CuImageWrite::Write(entry.imageName + "_vk.png", cuOutColors, entry.camera.width, entry.camera.height);

		CuTileRasterizer::Forward(cuTileRasterFwdROArgs, cuTileRasterFwdRWArgs, cuTileRasterResource);
		CuImageWrite::Write(entry.imageName + "_cu.png", cuOutColors, entry.camera.width, entry.camera.height);

		break; // Only once entry
	}
	return 0;
}
