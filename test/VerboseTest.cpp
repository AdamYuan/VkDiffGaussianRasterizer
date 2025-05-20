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
myvk::Ptr<myvk::PhysicalDevice> SelectPhysicalDevice(const myvk::Ptr<myvk::Instance> &pInstance);
} // namespace cuperftest

struct VerboseStat {
	double cohesionRate{}, atomicAddRate{};
	void Print(const char *prefix) const {
		printf("%s subgroup-splat cohesion rate: %lf\n", prefix, cohesionRate);
		printf("%s atomic-add rate: %lf\n", prefix, atomicAddRate);
	}
	static VerboseStat Average(auto &&range) {
		VerboseStat stat{};
		uint32_t count = 0;
		for (const VerboseStat &r : range) {
			stat.cohesionRate += r.cohesionRate;
			stat.atomicAddRate += r.atomicAddRate;
			++count;
		}
		stat.cohesionRate /= double(count);
		stat.atomicAddRate /= double(count);
		return stat;
	}
};

int main(int argc, char **argv) {
	--argc, ++argv;
	static constexpr int kStaticArgCount = 1;
	static constexpr const char *kHelpString = "./VerboseTest [dataset] (-w=[width]) (-h=[height]) (-i=[model "
	                                           "iteration]) (-e=[entries per scene]) (-w: write result) (-s: single)\n";
	if (argc < kStaticArgCount) {
		printf(kHelpString);
		return EXIT_FAILURE;
	}

	uint32_t modelIteration = kDefaultModelIteration;
	uint32_t resizeWidth = 0, resizeHeight = 0;
	uint32_t entriesPerScene = 0;
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
		else if (getValue(arg, "-e=", val))
			entriesPerScene = std::stoul(val);
		else {
			printf(kHelpString);
			return EXIT_FAILURE;
		}
	}

	myvk::Ptr<myvk::Device> pDevice;
	myvk::Ptr<myvk::Queue> pGenericQueue;
	{
		auto pInstance = myvk::Instance::Create({});
		auto pPhysicalDevice = cuperftest::SelectPhysicalDevice(pInstance);
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
	if (entriesPerScene) {
		std::mt19937 randGen{};
		gsDataset.RandomCrop(randGen, entriesPerScene);
	}
	if (single)
		gsDataset.SingleCrop();

	uint32_t gsDatasetMaxSplatCount = gsDataset.GetMaxSplatCount();
	uint32_t gsDatasetMaxPixelCount = gsDataset.GetMaxPixelCount();
	VkGSModel vkGsModel = VkGSModel::Create(pDevice, vkgsraster::Rasterizer::GetFwdArgsUsage().splatBuffers,
	                                        gsDatasetMaxSplatCount, createVkCuBuffer);

	vkgsraster::Rasterizer vkRasterizer{pDevice, {.forwardOutputImage = false}};
	vkgsraster::Rasterizer::Resource vkRasterResource = {};
	vkRasterResource.UpdateBuffer(pDevice, gsDatasetMaxSplatCount, 0.0);
	auto vkRasterVerboseQuery = vkgsraster::Rasterizer::VerboseQuery::Create(pDevice);
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
	vkgsraster::Rasterizer::BwdRWArgs vkRasterBwdRWArgs = {
	    .dL_dSplats = VkGSModel::Create(pDevice, vkgsraster::Rasterizer::GetBwdArgsUsage().dL_dSplatBuffers,
	                                    gsDatasetMaxSplatCount, createVkCuBuffer)
	                      .GetSplatArgs(),
	};

	CuTileRasterizer::FwdROArgs cuTileRasterFwdROArgs{};
	CuTileRasterizer::FwdRWArgs cuTileRasterFwdRWArgs{};
	CuTileRasterizer::BwdROArgs cuTileRasterBwdROArgs{};
	CuTileRasterizer::BwdRWArgs cuTileRasterBwdRWArgs{};

	std::vector<VerboseStat> verbStats;

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

		std::vector<VerboseStat> sceneVerbStats;

		for (uint32_t entryIdx = 0; auto &entry : scene.entries) {
			printf("\n%s %d/%zu %s\n", scene.name.c_str(), entryIdx++, scene.entries.size(), entry.imageName.c_str());
			vkRasterResource.UpdateImage(pDevice, entry.camera.width, entry.camera.height, vkRasterizer);
			vkRasterFwdROArgs.camera = entry.camera;
			vkRasterBwdROArgs.fwd.camera = entry.camera;

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
			printf("atomic adds: %d\n", verbose.atomicAddCount);
			VerboseStat entryVerbStat{
			    .cohesionRate = double(verbose.coherentFragmentCount) / double(verbose.fragmentCount),
			    .atomicAddRate = double(verbose.atomicAddCount) / double(verbose.fragmentCount),
			};
			entryVerbStat.Print("");
			sceneVerbStats.push_back(entryVerbStat);
		}
		auto sceneVerbStat = VerboseStat::Average(sceneVerbStats);
		sceneVerbStat.Print(scene.name.c_str());
		verbStats.push_back(sceneVerbStat);
	}

	VerboseStat::Average(verbStats).Print("avg");

	return 0;
}
