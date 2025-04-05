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
		pDevice = myvk::Device::Create(pPhysicalDevice, myvk::GenericQueueSelector{&pGenericQueue}, features,
		                               {
		                                   VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
		                                   VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME,
		                                   VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
		                                   VK_EXT_LOAD_STORE_OP_NONE_EXTENSION_NAME,
		                                   VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME,
		                                   VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME,
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
	    .dL_dSplats = VkGSModel::Create(pDevice, vkgsraster::Rasterizer::GetBwdArgsUsage().dL_dSplatBuffers,
	                                    vkGsModel.splatCount, createVkCuBuffer)
	                      .GetSplatArgs(),
	};

	CuTileRasterizer::Resource cuTileRasterResource{};
	CuTileRasterizer::FwdROArgs cuTileRasterFwdROArgs{};
	CuTileRasterizer::FwdRWArgs cuTileRasterFwdRWArgs{};
	CuTileRasterizer::BwdROArgs cuTileRasterBwdROArgs{};
	CuTileRasterizer::BwdRWArgs cuTileRasterBwdRWArgs{};
	CuTileRasterizer::PerfQuery cuTileRasterPerfQuery = CuTileRasterizer::PerfQuery::Create();

	for (const auto &entry : gsDataset.entries) {
		vkRasterPerfQuery.Reset();
		vkRasterResource.UpdateImage(pDevice, entry.camera.width, entry.camera.height, vkRasterizer);
		vkRasterFwdROArgs.camera = entry.camera;
		vkRasterBwdROArgs.fwd.camera = entry.camera;
		std::size_t pixelBufferSize = 3 * entry.camera.width * entry.camera.height * sizeof(float);
		if (!vkRasterFwdRWArgs.pOutPixelBuffer || vkRasterFwdRWArgs.pOutPixelBuffer->GetSize() < pixelBufferSize) {
			vkRasterFwdRWArgs.pOutPixelBuffer =
			    VkCuBuffer::Create(pDevice, pixelBufferSize, vkgsraster::Rasterizer::GetFwdArgsUsage().outPixelBuffer,
			                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}
		if (!vkRasterBwdROArgs.pdL_dPixelBuffer || vkRasterBwdROArgs.pdL_dPixelBuffer->GetSize() < pixelBufferSize) {
			vkRasterBwdROArgs.pdL_dPixelBuffer =
			    VkCuBuffer::Create(pDevice, pixelBufferSize, vkgsraster::Rasterizer::GetBwdArgsUsage().dL_dPixelBuffer,
			                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}

		cuTileRasterFwdROArgs.Update(vkRasterFwdROArgs);
		cuTileRasterFwdRWArgs.Update(vkRasterFwdRWArgs);
		cuTileRasterBwdROArgs.Update(cuTileRasterFwdROArgs, vkRasterBwdROArgs);
		cuTileRasterBwdRWArgs.Update(vkRasterBwdRWArgs);
		float *cuOutPixels = cuTileRasterFwdRWArgs.outPixels;

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
		printf("cu: %lf ms\n", cuTileRasterPerfMetrics.forward + cuTileRasterPerfMetrics.backward);
		CuImageWrite::Write(entry.imageName + "_cu.png", cuOutPixels, entry.camera.width, entry.camera.height);

		const auto runVkCommand = [&](auto &&cmdRun) {
			pCommandBuffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			cmdRun();
			pCommandBuffer->End();
			pCommandBuffer->Submit(pFence);
			pFence->Wait();
			pCommandPool->Reset();
			pFence->Reset();
		};
		runVkCommand([&] {
			vkRasterizer.CmdForward(pCommandBuffer, vkRasterFwdROArgs, vkRasterFwdRWArgs, vkRasterResource,
			                        vkRasterPerfQuery);
		});
		runVkCommand([&] {
			vkRasterizer.CmdBackward(pCommandBuffer, vkRasterBwdROArgs, vkRasterBwdRWArgs, vkRasterResource,
			                         vkRasterPerfQuery);
		});

		auto vkRasterPerfMetrics = vkRasterPerfQuery.GetMetrics();
		printf("vk_forward: %lf ms\n", vkRasterPerfMetrics.forward);
		printf("vk_backward: %lf ms\n", vkRasterPerfMetrics.backward);
		printf("vk: %lf ms\n", vkRasterPerfMetrics.forward + vkRasterPerfMetrics.backward);

		printf("speedup_forward: %lf\n", cuTileRasterPerfMetrics.forward / vkRasterPerfMetrics.forward);
		printf("speedup_backward: %lf\n", cuTileRasterPerfMetrics.backward / vkRasterPerfMetrics.backward);
		printf("speedup: %lf\n", (cuTileRasterPerfMetrics.forward + cuTileRasterPerfMetrics.backward) /
		                             (vkRasterPerfMetrics.forward + vkRasterPerfMetrics.backward));
		CuImageWrite::Write(entry.imageName + "_vk.png", cuOutPixels, entry.camera.width, entry.camera.height);

		break; // Only once entry
	}
	return 0;
}
