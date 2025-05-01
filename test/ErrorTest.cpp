#include "../src/Rasterizer.hpp"
#include "CuTileRasterizer.hpp"
#include "GSDataset.hpp"
#include "GSModel.hpp"
#include "VkCuBuffer.hpp"

#include "ErrorTest.hpp"

#include <myvk/ExternalMemoryUtil.hpp>
#include <myvk/Instance.hpp>
#include <myvk/Queue.hpp>
#include <myvk/QueueSelector.hpp>
#include <random>
#include <span>

static constexpr uint32_t kMaxWriteSplatCount = 128;
static constexpr uint32_t kDefaultModelIteration = 7000;

namespace cuperftest {
void WritePixelsPNG(const std::filesystem::path &filename, const float *devicePixels, uint32_t width, uint32_t height);
void RandomPixels(float *devicePixels, uint32_t width, uint32_t height);
void ClearDL_DSplats(const CuTileRasterizer::SplatArgs &splats, uint32_t splatCount);
void WriteDL_DSplatsJSON(const std::filesystem::path &filename, const CuTileRasterizer::SplatArgs &splats,
                         uint32_t splatCount);
} // namespace cuperftest

const auto RRMSE = [](std::span<const float> yHat, std::span<const float> y) {
	uint32_t count = 0;
	double sum1 = 0, sum2 = 0;
	for (uint32_t i = 0; i < yHat.size(); ++i) {
		// Only compute non-zero terms
		if (std::abs(yHat[i]) < 1e-8f && std::abs(y[i]) < 1e-8f)
			continue;
		if (i < 10)
			printf("%f, %f\n", yHat[i], y[i]);
		auto d = double(yHat[i] - y[i]);
		sum1 += d * d;
		sum2 += double(yHat[i] * yHat[i]);
		++count;
	}
	printf("\n\n");
	return std::sqrt(sum1 / double(count) / sum2);
};

struct MRE {
	float yHatMin, yHatMax;
	double operator()(std::span<const float> yHat, std::span<const float> y) const {
		uint32_t count = 0;
		double sum = 0;
		for (uint32_t i = 0; i < yHat.size(); ++i) {
			if (yHatMin < std::abs(yHat[i]) && std::abs(yHat[i]) < yHatMax) {
				auto d = std::abs(double(yHat[i] - y[i]));
				sum += d / std::abs(double(yHat[i]));
				++count;
			}
		}
		return sum / double(count);
	}
	std::string GetPrefix() const { return "MRE (" + std::to_string(yHatMin) + "~" + std::to_string(yHatMax) + ")"; }
};

void PrintError(const char *prefix, const GSGradient::Error &error) {
	printf("%s mean: %.10lf\n", prefix, error.mean);
	printf("%s scale: %.10lf\n", prefix, error.scale);
	printf("%s rotate: %.10lf\n", prefix, error.rotate);
	printf("%s opacity: %.10lf\n", prefix, error.opacity);
	printf("%s sh0: %.10lf\n", prefix, error.sh0);
}

int main(int argc, char **argv) {
	--argc, ++argv;
	static constexpr int kStaticArgCount = 1;
	static constexpr const char *kHelpString = "./ErrorTest [dataset] (-w=[width]) (-h=[height]) (-i=[model "
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
	GSGradient vkRasterGradient{};
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
	GSGradient cuTileRasterGradient{};

	constexpr uint32_t kMRECount = 2;
	std::array<MRE, kMRECount> mreFuncs = {
	    MRE{1e-2f, 10.0f},
	    MRE{10.0f, std::numeric_limits<float>::infinity()},
	};

	uint32_t sumCount = 0;
	GSGradient::Error sumRRMSE = {};
	std::array<GSGradient::Error, kMRECount> sumMREs{};

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
			{
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
				cuperftest::ClearDL_DSplats(cuTileRasterBwdRWArgs.dL_dSplats, vkGsModel.splatCount);
				runVkCommand([&] {
					vkRasterizer.CmdBackward(pCommandBuffer, vkRasterBwdROArgs, vkRasterBwdRWArgs, vkRasterResource);
				});
				vkRasterGradient.Update(cuTileRasterBwdRWArgs.dL_dSplats, vkGsModel.splatCount);
			}

			// Cuda
			{
				CuTileRasterizer::Forward(cuTileRasterFwdROArgs, cuTileRasterFwdRWArgs, cuTileRasterResource);
				cuperftest::ClearDL_DSplats(cuTileRasterBwdRWArgs.dL_dSplats, vkGsModel.splatCount);
				CuTileRasterizer::Backward(cuTileRasterBwdROArgs, cuTileRasterBwdRWArgs, cuTileRasterResource);
				cuTileRasterGradient.Update(cuTileRasterBwdRWArgs.dL_dSplats, vkGsModel.splatCount);
			}

			GSGradient::Error rrmse = cuTileRasterGradient.GetError(vkRasterGradient, RRMSE);
			std::array<GSGradient::Error, kMRECount> mres{};
			for (uint32_t k = 0; k < kMRECount; ++k)
				mres[k] = cuTileRasterGradient.GetError(vkRasterGradient, mreFuncs[k]);

			PrintError("RRMSE", rrmse);
			for (uint32_t k = 0; k < kMRECount; ++k)
				PrintError(mreFuncs[k].GetPrefix().c_str(), mres[k]);

			++sumCount;
			sumRRMSE += rrmse;
			for (uint32_t k = 0; k < kMRECount; ++k)
				sumMREs[k] += mres[k];
		}
	}

	sumRRMSE /= double(sumCount);
	for (uint32_t k = 0; k < kMRECount; ++k)
		sumMREs[k] /= double(sumCount);
	PrintError("avg RRMSE", sumRRMSE);
	for (uint32_t k = 0; k < kMRECount; ++k)
		PrintError((std::string("avg ") + mreFuncs[k].GetPrefix()).c_str(), sumMREs[k]);

	return 0;
}
