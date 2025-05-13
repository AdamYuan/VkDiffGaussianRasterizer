#include "../src/Rasterizer.hpp"
#include "CuTileRasterizer.hpp"
#include "GSDataset.hpp"
#include "GSModel.hpp"
#include "VkCuBuffer.hpp"

#include <myvk/ExternalMemoryUtil.hpp>
#include <myvk/Instance.hpp>
#include <myvk/Queue.hpp>
#include <myvk/QueueSelector.hpp>
#include <random>

static constexpr uint32_t kMaxWriteSplatCount = 128;
static constexpr uint32_t kDefaultModelIteration = 7000;

namespace cuperftest {
void WritePixelsPNG(const std::filesystem::path &filename, const float *devicePixels, uint32_t width, uint32_t height);
void RandomPixels(float *devicePixels, uint32_t width, uint32_t height);
void ClearDL_DSplats(const CuTileRasterizer::SplatArgs &splats, uint32_t splatCount);
void WriteDL_DSplatsJSON(const std::filesystem::path &filename, const CuTileRasterizer::SplatArgs &splats,
                         uint32_t splatCount);
} // namespace cuperftest

struct MemStat {
	std::size_t sortMem, allMem;

	static MemStat From(const CuTileRasterizer::Resource &resource) {
		MemStat stat{};
		stat.sortMem += resource.binningBuffer.size;
		stat.allMem += stat.sortMem;
		stat.allMem += resource.imageBuffer.size;
		stat.allMem += resource.geometryBuffer.size;
		stat.allMem += resource.dLBuffer.size;
		return stat;
	}

	static MemStat From(const vkgsraster::Rasterizer::Resource &resource) {
		MemStat stat{};
		const auto getBufferSize = [](const myvk::Ptr<myvk::BufferBase> &pBuf) -> std::size_t {
			if (!pBuf)
				return 0;
			return pBuf->GetSize();
		};
		stat.sortMem += getBufferSize(resource.pSortKeyBuffer);
		stat.sortMem += getBufferSize(resource.pSortPayloadBuffer);
		stat.sortMem += getBufferSize(resource.sorterResource.pTempKeyBuffer);
		stat.sortMem += getBufferSize(resource.sorterResource.pTempPayloadBuffer);
		stat.sortMem += getBufferSize(resource.sorterResource.pGlobalHistBuffer);
		stat.sortMem += getBufferSize(resource.sorterResource.pPassHistBuffer);
		stat.sortMem += getBufferSize(resource.sorterResource.pIndexBuffer);
		stat.sortMem += getBufferSize(resource.sorterResource.pDispatchArgBuffer);
		stat.allMem += stat.sortMem;
		stat.allMem += getBufferSize(resource.pSortSplatIndexBuffer);
		stat.allMem += getBufferSize(resource.pColorMean2DXBuffer);
		stat.allMem += getBufferSize(resource.pConicMean2DYBuffer);
		stat.allMem += getBufferSize(resource.pViewOpacityBuffer);
		stat.allMem += getBufferSize(resource.pDL_DColorMean2DXBuffer);
		stat.allMem += getBufferSize(resource.pDL_DConicMean2DYBuffer);
		stat.allMem += getBufferSize(resource.pDL_DViewOpacityBuffer);
		stat.allMem += getBufferSize(resource.pQuadBuffer);
		stat.allMem += getBufferSize(resource.pDrawArgBuffer);
		stat.allMem += getBufferSize(resource.pDispatchArgBuffer);
		const auto getImageSize = [](const myvk::Ptr<myvk::ImageBase> &pImg) -> std::size_t {
			if (!pImg)
				return 0;
			std::size_t size = pImg->GetExtent().width * pImg->GetExtent().height;
			std::size_t formatSize = 4 * sizeof(float);
			if (pImg->GetFormat() == VK_FORMAT_R16G16B16A16_UNORM || pImg->GetFormat() == VK_FORMAT_R16G16B16A16_SFLOAT)
				formatSize = 4 * sizeof(uint16_t);
			else if (pImg->GetFormat() == VK_FORMAT_R8G8B8A8_UNORM)
				formatSize = 4 * sizeof(uint8_t);
			return size * formatSize;
		};
		stat.allMem += getImageSize(resource.pPixelTImage);
		stat.allMem += getImageSize(resource.pDL_DPixelImage);
		return stat;
	}

	void Print(const char *prefix) const {
		printf("%s sortMem: %lu = %lf KB = %lf MB = %lf GB\n", prefix, sortMem, (double)sortMem / 1000.0,
		       (double)sortMem / 1000000.0, (double)sortMem / 1000000000.0);
		printf("%s allMem: %lu = %lf KB = %lf MB = %lf GB\n", prefix, allMem, (double)allMem / 1000.0,
		       (double)allMem / 1000000.0, (double)allMem / 1000000000.0);
	}
};

int main(int argc, char **argv) {
	--argc, ++argv;
	static constexpr int kStaticArgCount = 1;
	static constexpr const char *kHelpString =
	    "./PerformanceTest [dataset] (-w=[width]) (-h=[height]) (-i=[model "
	    "iteration]) (-e=[entries per scene]) (-c=[crop ratio]) (-w: write result) (-s: single) (-nocu) (-novk)\n";
	if (argc < kStaticArgCount) {
		printf(kHelpString);
		return EXIT_FAILURE;
	}

	uint32_t modelIteration = kDefaultModelIteration;
	uint32_t resizeWidth = 0, resizeHeight = 0;
	uint32_t entriesPerScene = 0;
	float cropRatio = 0.0f;
	bool writeResult = false;
	bool single = false;
	bool noCu = false, noVk = false;
	for (int i = kStaticArgCount; i < argc; ++i) {
		auto arg = std::string{argv[i]};
		std::string val;

		const auto getValue = [](const std::string &arg, const std::string &prefix, std::string &value) -> bool {
			if (arg.length() <= prefix.length())
				return false;
			if (arg.substr(0, prefix.length()) != prefix)
				return false;
			value = arg.substr(prefix.length());
			return true;
		};

		if (arg == "-w")
			writeResult = true;
		else if (arg == "-s")
			single = true;
		else if (arg == "-nocu")
			noCu = true;
		else if (arg == "-novk")
			noVk = true;
		else if (getValue(arg, "-w=", val))
			resizeWidth = std::stoul(val);
		else if (getValue(arg, "-h=", val))
			resizeHeight = std::stoul(val);
		else if (getValue(arg, "-i=", val))
			modelIteration = std::stoul(val);
		else if (getValue(arg, "-e=", val))
			entriesPerScene = std::stoul(val);
		else if (getValue(arg, "-c=", val))
			cropRatio = std::stof(val);
		else {
			printf(kHelpString);
			return EXIT_FAILURE;
		}
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

	GSDataset gsDataset = GSDataset::Load(argv[0], modelIteration);
	if (gsDataset.IsEmpty()) {
		printf("Empty Dataset %s\n", argv[0]);
		return EXIT_FAILURE;
	}
	gsDataset.ResizeCamera(resizeWidth, resizeHeight);
	{
		std::mt19937 randGen{};
		if (entriesPerScene && cropRatio != 0.0f) {
			printf("-e= and -c= cannot be used together\n");
			return EXIT_FAILURE;
		}
		if (entriesPerScene)
			gsDataset.RandomCrop(randGen, entriesPerScene);
		if (cropRatio != 0.0f)
			gsDataset.RandomCrop(randGen, cropRatio);
	}
	if (single)
		gsDataset.SingleCrop();

	uint32_t gsDatasetMaxSplatCount = gsDataset.GetMaxSplatCount();
	uint32_t gsDatasetMaxPixelCount = gsDataset.GetMaxPixelCount();
	VkGSModel vkGsModel = VkGSModel::Create(pDevice, vkgsraster::Rasterizer::GetFwdArgsUsage().splatBuffers,
	                                        gsDatasetMaxSplatCount, createVkCuBuffer);

	vkgsraster::Rasterizer vkRasterizer{pDevice, {.forwardOutputImage = false}};
	vkgsraster::Rasterizer::Resource vkRasterResource = {};
	vkgsraster::Rasterizer::FwdROArgs vkRasterFwdROArgs = {};
	vkgsraster::Rasterizer::FwdRWArgs vkRasterFwdRWArgs = {};
	vkgsraster::Rasterizer::BwdROArgs vkRasterBwdROArgs = {};
	printf("gsDatasetMaxPixelCount: %d\n", gsDatasetMaxPixelCount);
	vkRasterFwdRWArgs.pOutPixelBuffer = VkCuBuffer::Create(pDevice, gsDatasetMaxPixelCount * 3 * sizeof(float),
	                                                       vkgsraster::Rasterizer::GetFwdArgsUsage().outPixelBuffer,
	                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	auto pdL_dPixelBuffer = VkCuBuffer::Create(pDevice, gsDatasetMaxPixelCount * 3 * sizeof(float),
	                                           vkgsraster::Rasterizer::GetBwdArgsUsage().dL_dPixelBuffer,
	                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vkRasterBwdROArgs.pdL_dPixelBuffer = pdL_dPixelBuffer;
	cuperftest::RandomPixels(pdL_dPixelBuffer->GetCudaMappedPtr<float>(), gsDatasetMaxPixelCount, 1);
	if (!noVk) {
		vkRasterResource.UpdateBuffer(pDevice, gsDatasetMaxSplatCount, 0.0);
	}
	vkgsraster::Rasterizer::PerfQuery vkRasterPerfQuery = vkgsraster::Rasterizer::PerfQuery::Create(pDevice);
	vkgsraster::Rasterizer::BwdRWArgs vkRasterBwdRWArgs = {
	    .dL_dSplats = VkGSModel::Create(pDevice, vkgsraster::Rasterizer::GetBwdArgsUsage().dL_dSplatBuffers,
	                                    gsDatasetMaxSplatCount, createVkCuBuffer)
	                      .GetSplatArgs(),
	};

	CuTileRasterizer::Resource cuTileRasterResource{};
	CuTileRasterizer::FwdROArgs cuTileRasterFwdROArgs{};
	CuTileRasterizer::FwdRWArgs cuTileRasterFwdRWArgs{};
	CuTileRasterizer::BwdROArgs cuTileRasterBwdROArgs{};
	CuTileRasterizer::BwdRWArgs cuTileRasterBwdRWArgs{};
	auto cuTileRasterPerfQuery = CuTileRasterizer::PerfQuery::Create();

	uint32_t sumCount = 0;
	double sumCuForward = 0, sumVkForward = 0, sumCuBackward = 0, sumVkBackward = 0;
	double sumCuForwardDraw = 0, sumVkForwardDraw = 0, sumCuBackwardDraw = 0, sumVkBackwardDraw = 0;

	for (auto &scene : gsDataset.scenes) {
		{
			auto gsModel = GSModel::Load(scene.modelFilename);
			if (gsModel.IsEmpty()) {
				printf("Invalid 3DGS Model %s\n", scene.modelFilename.c_str());
				return EXIT_FAILURE;
			}
			vkGsModel.CopyFrom(pGenericQueue, gsModel);
		}

		printf("model: %s\nsplatCount = %d\n", scene.modelFilename.c_str(), vkGsModel.splatCount);

		vkRasterFwdROArgs.splatCount = vkGsModel.splatCount;
		vkRasterFwdROArgs.splats = vkGsModel.GetSplatArgs();
		vkRasterFwdROArgs.bgColor = {1.0f, 1.0f, 1.0f};
		vkRasterBwdROArgs.fwd = vkRasterFwdROArgs;

		for (uint32_t entryIdx = 0; auto &entry : scene.entries) {
			printf("\n%s %d/%zu %s\n", scene.name.c_str(), entryIdx++, scene.entries.size(), entry.imageName.c_str());
			vkRasterFwdROArgs.camera = entry.camera;
			vkRasterBwdROArgs.fwd.camera = entry.camera;

			cuTileRasterFwdROArgs.Update(vkRasterFwdROArgs);
			cuTileRasterFwdRWArgs.Update(vkRasterFwdRWArgs);
			cuTileRasterBwdROArgs.Update(cuTileRasterFwdROArgs, vkRasterBwdROArgs);
			cuTileRasterBwdRWArgs.Update(vkRasterBwdRWArgs);
			float *cuOutPixels = cuTileRasterFwdRWArgs.outPixels;

			// Vulkan
			vkgsraster::Rasterizer::PerfMetrics vkRasterPerfMetrics{};
			if (!noVk) {
				vkRasterResource.UpdateImage(pDevice, entry.camera.width, entry.camera.height, vkRasterizer);
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
					vkRasterizer.CmdForward(pCommandBuffer, vkRasterFwdROArgs, vkRasterFwdRWArgs, vkRasterResource);
				});
				runVkCommand([&] {
					vkRasterizer.CmdBackward(pCommandBuffer, vkRasterBwdROArgs, vkRasterBwdRWArgs, vkRasterResource);
				});
				runVkCommand([&] {
					vkRasterizer.CmdForward(pCommandBuffer, vkRasterFwdROArgs, vkRasterFwdRWArgs, vkRasterResource,
					                        vkRasterPerfQuery);
				});
				cuperftest::ClearDL_DSplats(cuTileRasterBwdRWArgs.dL_dSplats, vkGsModel.splatCount);
				runVkCommand([&] {
					vkRasterizer.CmdBackward(pCommandBuffer, vkRasterBwdROArgs, vkRasterBwdRWArgs, vkRasterResource,
					                         vkRasterPerfQuery);
				});

				if (writeResult) {
					cuperftest::WritePixelsPNG(entry.imageName + "_vk.png", cuOutPixels, entry.camera.width,
					                           entry.camera.height);
					cuperftest::WriteDL_DSplatsJSON(entry.imageName + "_vk.json", cuTileRasterBwdRWArgs.dL_dSplats,
					                                std::min(vkGsModel.splatCount, kMaxWriteSplatCount));
				}
				vkRasterPerfMetrics = vkRasterPerfQuery.GetMetrics();
				printf("vk_forward: %lf ms\n", vkRasterPerfMetrics.forward);
				printf("vk_forward draw: %lf ms\n", vkRasterPerfMetrics.forwardDraw);
				printf("vk_backward: %lf ms\n", vkRasterPerfMetrics.backward);
				printf("vk_backward draw: %lf ms\n", vkRasterPerfMetrics.backwardDraw);
				printf("vk: %lf ms\n", vkRasterPerfMetrics.forward + vkRasterPerfMetrics.backward);
				printf("vk draw: %lf ms\n", vkRasterPerfMetrics.forwardDraw + vkRasterPerfMetrics.backwardDraw);
			}

			// Cuda
			CuTileRasterizer::PerfMetrics cuTileRasterPerfMetrics{};
			if (!noCu) {
				CuTileRasterizer::Forward(cuTileRasterFwdROArgs, cuTileRasterFwdRWArgs, cuTileRasterResource);
				CuTileRasterizer::Backward(cuTileRasterBwdROArgs, cuTileRasterBwdRWArgs, cuTileRasterResource);

				CuTileRasterizer::Forward(cuTileRasterFwdROArgs, cuTileRasterFwdRWArgs, cuTileRasterResource,
				                          cuTileRasterPerfQuery);
				cuperftest::ClearDL_DSplats(cuTileRasterBwdRWArgs.dL_dSplats, vkGsModel.splatCount);
				CuTileRasterizer::Backward(cuTileRasterBwdROArgs, cuTileRasterBwdRWArgs, cuTileRasterResource,
				                           cuTileRasterPerfQuery);
				if (writeResult) {
					cuperftest::WritePixelsPNG(entry.imageName + "_cu.png", cuOutPixels, entry.camera.width,
					                           entry.camera.height);
					cuperftest::WriteDL_DSplatsJSON(entry.imageName + "_cu.json", cuTileRasterBwdRWArgs.dL_dSplats,
					                                std::min(vkGsModel.splatCount, kMaxWriteSplatCount));
				}
				cuTileRasterPerfMetrics = cuTileRasterPerfQuery.GetMetrics();
				printf("cu_forward: %lf ms (numRendered = %d)\n", cuTileRasterPerfMetrics.forward,
				       cuTileRasterResource.numRendered);
				printf("cu_forward draw: %lf ms\n", cuTileRasterPerfMetrics.forwardDraw);
				printf("cu_backward: %lf ms\n", cuTileRasterPerfMetrics.backward);
				printf("cu_backward draw: %lf ms\n", cuTileRasterPerfMetrics.backwardDraw);
				printf("cu: %lf ms\n", cuTileRasterPerfMetrics.forward + cuTileRasterPerfMetrics.backward);
				printf("cu draw: %lf ms\n", cuTileRasterPerfMetrics.forwardDraw + cuTileRasterPerfMetrics.backwardDraw);
			}

			if (!noCu && !noVk) {
				// Speed Up
				printf("speedup_forward: %lf\n", cuTileRasterPerfMetrics.forward / vkRasterPerfMetrics.forward);
				printf("speedup_backward: %lf\n", cuTileRasterPerfMetrics.backward / vkRasterPerfMetrics.backward);
				printf("speedup_forward draw: %lf\n",
				       cuTileRasterPerfMetrics.forwardDraw / vkRasterPerfMetrics.forwardDraw);
				printf("speedup_backward draw: %lf\n",
				       cuTileRasterPerfMetrics.backwardDraw / vkRasterPerfMetrics.backwardDraw);
				printf("speedup: %lf\n", (cuTileRasterPerfMetrics.forward + cuTileRasterPerfMetrics.backward) /
				                             (vkRasterPerfMetrics.forward + vkRasterPerfMetrics.backward));
				printf("speedup draw: %lf\n",
				       (cuTileRasterPerfMetrics.forwardDraw + cuTileRasterPerfMetrics.backwardDraw) /
				           (vkRasterPerfMetrics.forwardDraw + vkRasterPerfMetrics.backwardDraw));
			}

			++sumCount;
			sumCuForward += cuTileRasterPerfMetrics.forward;
			sumCuForwardDraw += cuTileRasterPerfMetrics.forwardDraw;
			sumCuBackward += cuTileRasterPerfMetrics.backward;
			sumCuBackwardDraw += cuTileRasterPerfMetrics.backwardDraw;
			sumVkForward += vkRasterPerfMetrics.forward;
			sumVkForwardDraw += vkRasterPerfMetrics.forwardDraw;
			sumVkBackward += vkRasterPerfMetrics.backward;
			sumVkBackwardDraw += vkRasterPerfMetrics.backwardDraw;
		}
	}

	printf("avg vk_forward: %lf ms\n", sumVkForward / double(sumCount));
	printf("avg vk_forward draw: %lf ms\n", sumVkForwardDraw / double(sumCount));
	printf("avg vk_backward: %lf ms\n", sumVkBackward / double(sumCount));
	printf("avg vk_backward draw: %lf ms\n", sumVkBackwardDraw / double(sumCount));
	printf("avg vk: %lf ms\n", (sumVkForward + sumVkBackward) / double(sumCount));
	printf("avg vk draw: %lf ms\n", (sumVkForwardDraw + sumVkBackwardDraw) / double(sumCount));

	printf("avg cu_forward: %lf ms\n", sumCuForward / double(sumCount));
	printf("avg cu_forward draw: %lf ms\n", sumCuForwardDraw / double(sumCount));
	printf("avg cu_backward: %lf ms\n", sumCuBackward / double(sumCount));
	printf("avg cu_backward draw: %lf ms\n", sumCuBackwardDraw / double(sumCount));
	printf("avg cu: %lf ms\n", (sumCuForward + sumCuBackward) / double(sumCount));
	printf("avg cu draw: %lf ms\n", (sumCuForwardDraw + sumCuBackwardDraw) / double(sumCount));

	printf("avg speedup_forward: %lf\n", sumCuForward / sumVkForward);
	printf("avg speedup_forward draw: %lf\n", sumCuForwardDraw / sumVkForwardDraw);
	printf("avg speedup_backward: %lf\n", sumCuBackward / sumVkBackward);
	printf("avg speedup_backward draw: %lf\n", sumCuBackwardDraw / sumVkBackwardDraw);
	printf("avg speedup: %lf\n", (sumCuForward + sumCuBackward) / (sumVkForward + sumVkBackward));
	printf("avg speedup draw: %lf\n", (sumCuForwardDraw + sumCuBackwardDraw) / (sumVkForwardDraw + sumVkBackwardDraw));

	MemStat::From(cuTileRasterResource).Print("cu");
	MemStat::From(vkRasterResource).Print("vk");

	return 0;
}
