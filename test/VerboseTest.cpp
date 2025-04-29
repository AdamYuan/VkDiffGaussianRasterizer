#include "../src/Rasterizer.hpp"
#include "CuTileRasterizer.hpp"
#include "GSDataset.hpp"
#include "GSModel.hpp"
#include "VkCuBuffer.hpp"

#include <myvk/ExternalMemoryUtil.hpp>
#include <myvk/Instance.hpp>
#include <myvk/Queue.hpp>
#include <myvk/QueueSelector.hpp>

static constexpr uint32_t kMaxWriteSplatCount = 128;
static constexpr uint32_t kDefaultModelIteration = 7000;

namespace cuperftest {
void WritePixelsPNG(const std::filesystem::path &filename, const float *devicePixels, uint32_t width, uint32_t height);
void RandomPixels(float *devicePixels, uint32_t width, uint32_t height);
void ClearDL_DSplats(const CuTileRasterizer::SplatArgs &splats, uint32_t splatCount);
void WriteDL_DSplatsJSON(const std::filesystem::path &filename, const CuTileRasterizer::SplatArgs &splats,
                         uint32_t splatCount);
} // namespace cuperftest

int main(int argc, char **argv) {
	--argc, ++argv;
	static constexpr int kStaticArgCount = 1;
	static constexpr const char *kHelpString =
	    "./VerboseTest [dataset] (-w=[width]) (-h=[height]) (-i=[model iteration]) (-w: write result) (-s: single)\n";
	if (argc < kStaticArgCount) {
		printf(kHelpString);
		return EXIT_FAILURE;
	}

	uint32_t modelIteration = kDefaultModelIteration;
	uint32_t resizeWidth = 0, resizeHeight = 0;
	bool writeResult = false;
	bool single = false;
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
		else if (getValue(arg, "-w=", val))
			resizeWidth = std::stoul(val);
		else if (getValue(arg, "-h=", val))
			resizeHeight = std::stoul(val);
		else if (getValue(arg, "-i=", val))
			modelIteration = std::stoul(val);
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
	GSDataset gsDataset = GSDataset::Load(argv[0]);
	if (gsDataset.IsEmpty()) {
		printf("Empty Dataset %s\n", argv[0]);
		return EXIT_FAILURE;
	}
	gsDataset.ResizeCamera(resizeWidth, resizeHeight);

	vkgsraster::Rasterizer vkRasterizer{pDevice, {.forwardOutputImage = false}};
	vkgsraster::Rasterizer::Resource vkRasterResource = {};
	auto vkRasterVerboseQuery = vkgsraster::Rasterizer::VerboseQuery::Create(pDevice);

	CuTileRasterizer::Resource cuTileRasterResource{};
	CuTileRasterizer::FwdROArgs cuTileRasterFwdROArgs{};
	CuTileRasterizer::FwdRWArgs cuTileRasterFwdRWArgs{};
	CuTileRasterizer::BwdROArgs cuTileRasterBwdROArgs{};
	CuTileRasterizer::BwdRWArgs cuTileRasterBwdRWArgs{};

	uint32_t sumCount = 0;
	double sumCohesionRate = 0, sumAtomicAddRate = 0;

	for (auto &scene : gsDataset.scenes) {
		VkGSModel vkGsModel = VkGSModel::Create(pGenericQueue, vkgsraster::Rasterizer::GetFwdArgsUsage().splatBuffers,
		                                        GSModel::Load(scene.modelFilename), createVkCuBuffer);

		if (vkGsModel.IsEmpty()) {
			printf("Invalid 3DGS Model %s\n", scene.modelFilename.c_str());
			return EXIT_FAILURE;
		}

		printf("model: %s\nsplatCount = %d\n", scene.modelFilename.c_str(), vkGsModel.splatCount);

		vkRasterResource.UpdateBuffer(pDevice, vkGsModel.splatCount, 0.0);
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

		if (single)
			scene.entries = {scene.entries[0]};

		for (uint32_t entryIdx = 0; auto &entry : scene.entries) {
			printf("\n%s %d/%zu %s\n", scene.name.c_str(), entryIdx++, scene.entries.size(), entry.imageName.c_str());
			vkRasterResource.UpdateImage(pDevice, entry.camera.width, entry.camera.height, vkRasterizer);
			vkRasterFwdROArgs.camera = entry.camera;
			vkRasterBwdROArgs.fwd.camera = entry.camera;
			std::size_t pixelBufferSize = 3 * entry.camera.width * entry.camera.height * sizeof(float);
			if (!vkRasterFwdRWArgs.pOutPixelBuffer || vkRasterFwdRWArgs.pOutPixelBuffer->GetSize() < pixelBufferSize) {
				vkRasterFwdRWArgs.pOutPixelBuffer = VkCuBuffer::Create(
				    pDevice, pixelBufferSize, vkgsraster::Rasterizer::GetFwdArgsUsage().outPixelBuffer,
				    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			}
			if (!vkRasterBwdROArgs.pdL_dPixelBuffer ||
			    vkRasterBwdROArgs.pdL_dPixelBuffer->GetSize() < pixelBufferSize) {
				auto pdL_dPixelBuffer = VkCuBuffer::Create(pDevice, pixelBufferSize,
				                                           vkgsraster::Rasterizer::GetBwdArgsUsage().dL_dPixelBuffer,
				                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
				vkRasterBwdROArgs.pdL_dPixelBuffer = pdL_dPixelBuffer;

				cuperftest::RandomPixels(pdL_dPixelBuffer->GetCudaMappedPtr<float>(), entry.camera.width,
				                         entry.camera.height);
			}

			cuTileRasterFwdROArgs.Update(vkRasterFwdROArgs);
			cuTileRasterFwdRWArgs.Update(vkRasterFwdRWArgs);
			cuTileRasterBwdROArgs.Update(cuTileRasterFwdROArgs, vkRasterBwdROArgs);
			cuTileRasterBwdRWArgs.Update(vkRasterBwdRWArgs);
			float *cuOutPixels = cuTileRasterFwdRWArgs.outPixels;

			vkRasterVerboseQuery.Reset();

			// Vulkan
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
				                        vkRasterVerboseQuery);
			});
			cuperftest::ClearDL_DSplats(cuTileRasterBwdRWArgs.dL_dSplats, vkGsModel.splatCount);
			runVkCommand([&] {
				vkRasterizer.CmdBackward(pCommandBuffer, vkRasterBwdROArgs, vkRasterBwdRWArgs, vkRasterResource,
				                         vkRasterVerboseQuery);
			});
			if (writeResult)
				cuperftest::WritePixelsPNG(entry.imageName + "_verb_" + std::to_string(entry.camera.width) + "x" +
				                               std::to_string(entry.camera.height) + ".png",
				                           cuOutPixels, entry.camera.width, entry.camera.height);

			vkgsraster::Rasterizer::VerboseMetrics verbose = vkRasterVerboseQuery.GetMetrics();
			printf("fragments: %d\n", verbose.fragmentCount);
			printf("coherent fragments: %d\n", verbose.coherentFragmentCount);
			double cohesionRate = double(verbose.coherentFragmentCount) / double(verbose.fragmentCount);
			printf("subgroup-splat cohesion rate: %lf\n", cohesionRate);
			printf("atomic adds: %d\n", verbose.atomicAddCount);
			double atomicAddRate = double(verbose.atomicAddCount) / double(verbose.fragmentCount);
			printf("atomic-add rate: %lf\n", atomicAddRate);

			sumCohesionRate += cohesionRate;
			sumAtomicAddRate += atomicAddRate;
			++sumCount;
		}
	}

	printf("avg subgroup-splat cohesion rate: %lf\n", sumCohesionRate / double(sumCount));
	printf("avg atomic-add rate: %lf\n", sumAtomicAddRate / double(sumCount));

	return 0;
}
