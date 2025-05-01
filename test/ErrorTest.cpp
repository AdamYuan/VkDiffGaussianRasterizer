#include "../src/Rasterizer.hpp"
#include "CuTileRasterizer.hpp"
#include "GSDataset.hpp"
#include "GSModel.hpp"
#include "VkCuBuffer.hpp"

#include "ErrorTest.hpp"

#include <span>
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

GSGradient::Error GSGradient::GetMRE(const GSGradient &r) const {
	const auto getMRE = []<typename T>(const std::vector<T> &yHat, const std::vector<T> &y, uint32_t splatCount) {
		static_assert(sizeof(T) % sizeof(float) == 0);
		std::span yHatFlt{reinterpret_cast<const float *>(yHat.data()),
		                  reinterpret_cast<const float *>(yHat.data() + splatCount)};
		std::span yFlt{reinterpret_cast<const float *>(y.data()), //
		               reinterpret_cast<const float *>(y.data() + splatCount)};
		uint32_t count = 0;
		double sum = 0;
		printf("%zu %zu\n", yHat.size(), yHatFlt.size());
		for (uint32_t i = 0; i < yHatFlt.size(); ++i) {
			if (yHatFlt[i] < 1e-2f)
				continue;
			if (i < 10/*double(std::abs((yHatFlt[i] - yFlt[i]) / yHatFlt[i])) > 1*/) {
				printf("%f %f\n", yHatFlt[i], yFlt[i]);
			}
			sum += double(std::abs((yHatFlt[i] - yFlt[i]) / yHatFlt[i]));
			++count;
		}
		return sum / double(count);
	};

	Error mre{};
	mre.mean = getMRE(means, r.means, splatCount);
	mre.scale = getMRE(scales, r.scales, splatCount);
	mre.rotate = getMRE(rotates, r.rotates, splatCount);
	mre.opacity = getMRE(opacities, r.opacities, splatCount);
	mre.sh0 = getMRE(sh0s, r.sh0s, splatCount);
	return mre;
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

	vkgsraster::Rasterizer vkRasterizer{pDevice, {.forwardOutputImage = false}};
	vkgsraster::Rasterizer::Resource vkRasterResource = {};
	GSGradient vkRasterGradient{};

	CuTileRasterizer::Resource cuTileRasterResource{};
	CuTileRasterizer::FwdROArgs cuTileRasterFwdROArgs{};
	CuTileRasterizer::FwdRWArgs cuTileRasterFwdRWArgs{};
	CuTileRasterizer::BwdROArgs cuTileRasterBwdROArgs{};
	CuTileRasterizer::BwdRWArgs cuTileRasterBwdRWArgs{};
	GSGradient cuTileRasterGradient{};

	uint32_t sumCount = 0;
	GSGradient::Error sumMRE = {};

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

			GSGradient::Error mre = cuTileRasterGradient.GetMRE(vkRasterGradient);

			printf("MRE mean: %lf\n", mre.mean);
			printf("MRE scale: %lf\n", mre.scale);
			printf("MRE rotate: %lf\n", mre.rotate);
			printf("MRE opacity: %lf\n", mre.opacity);
			printf("MRE sh0: %lf\n", mre.sh0);

			++sumCount;
			sumMRE += mre;
		}
	}

	sumMRE /= double(sumCount);
	printf("avg MRE mean: %lf\n", sumMRE.mean);
	printf("avg MRE scale: %lf\n", sumMRE.scale);
	printf("avg MRE rotate: %lf\n", sumMRE.rotate);
	printf("avg MRE opacity: %lf\n", sumMRE.opacity);
	printf("avg MRE sh0: %lf\n", sumMRE.sh0);

	return 0;
}