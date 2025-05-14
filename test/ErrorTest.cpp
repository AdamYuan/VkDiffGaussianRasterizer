#include "../src/Rasterizer.hpp"
#include "../src/RasterizerF32.hpp"
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

constexpr float kZeroThreshold = 1e-8f;

const auto kRMSEFunc = [](std::span<const float> y, std::span<const float> yHat) {
	double sum = 0;
	float yMin = std::numeric_limits<float>::max(), yMax = -std::numeric_limits<float>::max();
	uint32_t count = 0;
	for (uint32_t i = 0; i < y.size(); ++i) {
		float absY = std::abs(y[i]);
		if (absY >= kZeroThreshold) {
			if (i < 10)
				printf("%f, %f\n", y[i], yHat[i]);
			auto d = double(y[i] - yHat[i]);
			sum += d * d;
			yMin = std::min(yMin, y[i]);
			yMax = std::max(yMax, y[i]);
			++count;
		}
	}
	printf("\n\n");
	return std::sqrt(sum / double(count));
};

const auto kSignAccuracyFunc = [](std::span<const float> y, std::span<const float> yHat) {
	uint32_t count = 0;
	uint32_t correctCount = 0;
	for (uint32_t i = 0; i < y.size(); ++i) {
		float absY = std::abs(y[i]);
		if (absY >= kZeroThreshold) {
			correctCount += ((y[i] > 0) == (yHat[i] > 0));
			++count;
		}
	}
	return double(correctCount) / double(count);
};

struct MRERange {
	float yMin, yMax;
};

template <MRERange... Ranges_V> struct MRE {
	static constexpr std::array kRanges = {Ranges_V...};
	static constexpr std::size_t kRangeCount = sizeof...(Ranges_V);
	std::array<double, kRangeCount> errors{};

	struct Func {
		MRE operator()(std::span<const float> y, std::span<const float> yHat) const {
			std::array<double, kRangeCount> sums{};
			std::array<uint32_t, kRangeCount> counts{};
			uint32_t nonZeroCount{};
			for (uint32_t i = 0; i < y.size(); ++i) {
				float absY = std::abs(y[i]);
				for (uint32_t r = 0; r < kRangeCount; ++r) {
					MRERange range = kRanges[r];
					if (range.yMin <= absY && absY < range.yMax) {
						sums[r] += std::abs(double(y[i] - yHat[i])) / double(absY);
						++counts[r];
					}
				}
				if (absY >= kZeroThreshold)
					++nonZeroCount;
			}
			uint32_t count = 0;
			for (uint32_t r = 0; r < kRangeCount; ++r) {
				sums[r] /= double(counts[r]);
				count += counts[r];
			}
			printf("Coverage: %u/%u = %lf\n", count, nonZeroCount, double(count) / double(nonZeroCount));
			return MRE{.errors = sums};
		}
	};

	MRE &operator+=(const MRE &rMRE) {
		for (uint32_t r = 0; r < kRangeCount; ++r)
			errors[r] += rMRE.errors[r];
		return *this;
	}
	MRE &operator/=(double rVal) {
		for (uint32_t r = 0; r < kRangeCount; ++r)
			errors[r] /= rVal;
		return *this;
	}

	void Print(const char *prefix = "") const {
		for (uint32_t r = 0; r < kRangeCount; ++r) {
			printf("%sMRE [%f, %f) = %lf\n", prefix, kRanges[r].yMin, kRanges[r].yMax, errors[r]);
		}
	}
};

using TestMRE = MRE<                                    //
    MRERange{10.0f, std::numeric_limits<float>::max()}, //
    MRERange{0.1f, 10.0f},                              //
    MRERange{1e-3f, 0.1f}                               //
    >;

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

	vkgsraster::RasterizerF32 vkF32Rasterizer{pDevice, {.forwardOutputImage = false}};
	vkgsraster::RasterizerF32::Resource vkF32RasterResource = {};
	vkF32RasterResource.UpdateBuffer(pDevice, gsDatasetMaxSplatCount, 0.0);
	GSGradient vkF32RasterGradient{};
	vkgsraster::RasterizerF32::FwdROArgs vkF32RasterFwdROArgs = {};
	vkgsraster::RasterizerF32::FwdRWArgs vkF32RasterFwdRWArgs = {};
	vkgsraster::RasterizerF32::BwdROArgs vkF32RasterBwdROArgs = {};

	printf("gsDatasetMaxPixelCount: %d\n", gsDatasetMaxPixelCount);
	vkF32RasterFwdRWArgs.pOutPixelBuffer = vkRasterFwdRWArgs.pOutPixelBuffer = VkCuBuffer::Create(
	    pDevice, gsDatasetMaxPixelCount * 3 * sizeof(float), vkgsraster::Rasterizer::GetFwdArgsUsage().outPixelBuffer,
	    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	auto pdL_dPixelBuffer = VkCuBuffer::Create(pDevice, gsDatasetMaxPixelCount * 3 * sizeof(float),
	                                           vkgsraster::Rasterizer::GetBwdArgsUsage().dL_dPixelBuffer,
	                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vkF32RasterBwdROArgs.pdL_dPixelBuffer = vkRasterBwdROArgs.pdL_dPixelBuffer = pdL_dPixelBuffer;
	cuperftest::RandomPixels(pdL_dPixelBuffer->GetCudaMappedPtr<float>(), gsDatasetMaxPixelCount, 1);
	vkgsraster::Rasterizer::BwdRWArgs vkRasterBwdRWArgs = {
	    .dL_dSplats = VkGSModel::Create(pDevice, vkgsraster::Rasterizer::GetBwdArgsUsage().dL_dSplatBuffers,
	                                    gsDatasetMaxSplatCount, createVkCuBuffer)
	                      .GetSplatArgs(),
	};
	vkgsraster::RasterizerF32::BwdRWArgs vkF32RasterBwdRWArgs = {
	    .dL_dSplats =
	        {
	            .pMeanBuffer = vkRasterBwdRWArgs.dL_dSplats.pMeanBuffer,
	            .pScaleBuffer = vkRasterBwdRWArgs.dL_dSplats.pScaleBuffer,
	            .pRotateBuffer = vkRasterBwdRWArgs.dL_dSplats.pRotateBuffer,
	            .pOpacityBuffer = vkRasterBwdRWArgs.dL_dSplats.pOpacityBuffer,
	            .pSHBuffer = vkRasterBwdRWArgs.dL_dSplats.pSHBuffer,
	        },
	};

	CuTileRasterizer::Resource cuTileRasterResource{};
	CuTileRasterizer::FwdROArgs cuTileRasterFwdROArgs{};
	CuTileRasterizer::FwdRWArgs cuTileRasterFwdRWArgs{};
	CuTileRasterizer::BwdROArgs cuTileRasterBwdROArgs{};
	CuTileRasterizer::BwdRWArgs cuTileRasterBwdRWArgs{};

	uint32_t sumCount = 0;
	auto sumMRE = TestMRE{};
	double sumRMSE = 0, sumSignAcc = 0;

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

		vkF32RasterFwdROArgs.splatCount = vkGsModel.splatCount;
		vkF32RasterFwdROArgs.splats = {
		    .pMeanBuffer = vkRasterFwdROArgs.splats.pMeanBuffer,
		    .pScaleBuffer = vkRasterFwdROArgs.splats.pScaleBuffer,
		    .pRotateBuffer = vkRasterFwdROArgs.splats.pRotateBuffer,
		    .pOpacityBuffer = vkRasterFwdROArgs.splats.pOpacityBuffer,
		    .pSHBuffer = vkRasterFwdROArgs.splats.pSHBuffer,
		};
		vkF32RasterFwdROArgs.bgColor = {1.0f, 1.0f, 1.0f};
		vkF32RasterBwdROArgs.fwd = vkF32RasterFwdROArgs;

		for (uint32_t entryIdx = 0; auto &entry : scene.entries) {
			printf("\n%s %d/%zu %s\n", scene.name.c_str(), entryIdx++, scene.entries.size(), entry.imageName.c_str());
			vkRasterFwdROArgs.camera = entry.camera;
			vkRasterBwdROArgs.fwd.camera = entry.camera;

			vkF32RasterFwdROArgs.camera = {
			    .width = entry.camera.width,
			    .height = entry.camera.height,
			    .focalX = entry.camera.focalX,
			    .focalY = entry.camera.focalY,
			    .viewMat = entry.camera.viewMat,
			    .pos = entry.camera.pos,
			};
			vkF32RasterBwdROArgs.fwd.camera = vkF32RasterFwdROArgs.camera;

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
				vkF32RasterResource.UpdateImage(pDevice, entry.camera.width, entry.camera.height, vkF32Rasterizer);
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
					vkF32Rasterizer.CmdForward(pCommandBuffer, vkF32RasterFwdROArgs, vkF32RasterFwdRWArgs,
					                           vkF32RasterResource);
				});
				cuperftest::ClearDL_DSplats(cuTileRasterBwdRWArgs.dL_dSplats, vkGsModel.splatCount);
				runVkCommand([&] {
					vkF32Rasterizer.CmdBackward(pCommandBuffer, vkF32RasterBwdROArgs, vkF32RasterBwdRWArgs,
					                            vkF32RasterResource);
				});
				vkF32RasterGradient.Update(cuTileRasterBwdRWArgs.dL_dSplats, vkGsModel.splatCount);
			}

			TestMRE mre = vkF32RasterGradient.GetError(vkRasterGradient, TestMRE::Func{});
			double rmse = vkF32RasterGradient.GetError(vkRasterGradient, kRMSEFunc);
			double signAcc = vkF32RasterGradient.GetError(vkRasterGradient, kSignAccuracyFunc);
			mre.Print();
			printf("RMSE: %.10lf\n", rmse);
			printf("+/- Accuracy: %.10lf\n", signAcc);

			++sumCount;
			sumMRE += mre;
			sumRMSE += rmse;
			sumSignAcc += signAcc;
		}
	}

	sumMRE /= double(sumCount);
	sumRMSE /= double(sumCount);
	sumSignAcc /= double(sumCount);
	sumMRE.Print("avg ");
	printf("avg RMSE: %.10lf\n", sumRMSE);
	printf("avg +/- Accuracy: %.10lf\n", sumSignAcc);

	return 0;
}
