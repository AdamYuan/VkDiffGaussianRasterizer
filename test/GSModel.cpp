//
// Created by adamyuan on 3/27/25.
//

#include "GSModel.hpp"

#include <bit>
#include <cmath>
#include <fstream>
#include <myvk/Buffer.hpp>
#include <myvk/CommandBuffer.hpp>
#include <ranges>

GSModel GSModel::Load(const std::filesystem::path &filename) {
	std::ifstream fin{filename};
	if (!fin.is_open())
		return {};

	GSModel model{};

	{ // Parse Header
		std::string line;
		while (std::getline(fin, line)) {
			auto view = line | std::views::split(' ') |
			            std::views::transform([](auto rng) { return std::string_view(rng.data(), rng.size()); });
			std::vector<std::string_view> words{view.begin(), view.end()};
			if (words.empty())
				return {};
			if (words.size() >= 3 && words[0] == "element" && words[1] == "vertex")
				model.splatCount = std::stoi(std::string{words[2]});
			if (words[0] == "end_header")
				break;
		}
	}

	model.means.resize(model.splatCount);
	model.scales.resize(model.splatCount);
	model.opacities.resize(model.splatCount);
	model.rotates.resize(model.splatCount);
	model.shs.resize(model.splatCount);

	static constexpr uint32_t kFloatCount = 3 + 3 + kSHSize * 3 + 1 + 3 + 4;
	std::array<float, kFloatCount> buf{};
	for (uint32_t splatIdx = 0; splatIdx < model.splatCount; ++splatIdx) {
		float *pBuf = buf.data();
		fin.read(reinterpret_cast<char *>(pBuf), kFloatCount * sizeof(float));
		static_assert(std::endian::native == std::endian::little);

		model.means[splatIdx][0] = *pBuf++;
		model.means[splatIdx][1] = *pBuf++;
		model.means[splatIdx][2] = *pBuf++;

		pBuf += 3; // Skip nx, ny, nz

		for (uint32_t shIdx = 0; shIdx < kSHSize; ++shIdx) {
			model.shs[splatIdx][shIdx][0] = *pBuf++;
			model.shs[splatIdx][shIdx][1] = *pBuf++;
			model.shs[splatIdx][shIdx][2] = *pBuf++;
		}
		model.opacities[splatIdx] = *pBuf++;
		model.scales[splatIdx][0] = *pBuf++;
		model.scales[splatIdx][1] = *pBuf++;
		model.scales[splatIdx][2] = *pBuf++;
		model.rotates[splatIdx][0] = *pBuf++;
		model.rotates[splatIdx][1] = *pBuf++;
		model.rotates[splatIdx][2] = *pBuf++;
		model.rotates[splatIdx][3] = *pBuf++;

		static_assert(kFloatCount == 62);
	}

	for (uint32_t splatIdx = 0; splatIdx < model.splatCount; ++splatIdx) {
		auto &scale = model.scales[splatIdx];
		scale[0] = std::abs(std::exp(scale[0]));
		scale[1] = std::abs(std::exp(scale[1]));
		scale[2] = std::abs(std::exp(scale[2]));
	}

	for (uint32_t splatIdx = 0; splatIdx < model.splatCount; ++splatIdx) {
		auto &opacity = model.opacities[splatIdx];
		const auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
		opacity = sigmoid(opacity);
	}

	for (uint32_t splatIdx = 0; splatIdx < model.splatCount; ++splatIdx) {
		auto &rotate = model.rotates[splatIdx];
		float dot = rotate[0] * rotate[0]   //
		            + rotate[1] * rotate[1] //
		            + rotate[2] * rotate[2] //
		            + rotate[3] * rotate[3];
		float invLen = 1.0f / std::sqrt(dot);
		rotate[0] *= invLen;
		rotate[1] *= invLen;
		rotate[2] *= invLen;
		rotate[3] *= invLen;
	}

	return model;
}

// An example of the PLY header
#if 0
ply
format binary_little_endian 1.0
element vertex 3616103
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
property float f_rest_1
property float f_rest_2
property float f_rest_3
property float f_rest_4
property float f_rest_5
property float f_rest_6
property float f_rest_7
property float f_rest_8
property float f_rest_9
property float f_rest_10
property float f_rest_11
property float f_rest_12
property float f_rest_13
property float f_rest_14
property float f_rest_15
property float f_rest_16
property float f_rest_17
property float f_rest_18
property float f_rest_19
property float f_rest_20
property float f_rest_21
property float f_rest_22
property float f_rest_23
property float f_rest_24
property float f_rest_25
property float f_rest_26
property float f_rest_27
property float f_rest_28
property float f_rest_29
property float f_rest_30
property float f_rest_31
property float f_rest_32
property float f_rest_33
property float f_rest_34
property float f_rest_35
property float f_rest_36
property float f_rest_37
property float f_rest_38
property float f_rest_39
property float f_rest_40
property float f_rest_41
property float f_rest_42
property float f_rest_43
property float f_rest_44
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
#endif

VkGSModel VkGSModel::Create(const myvk::Ptr<myvk::Queue> &pQueue, VkBufferUsageFlags bufferUsage,
                            const GSModel &model) {
	if (model.IsEmpty())
		return {};

	const auto &pDevice = pQueue->GetDevicePtr();
	auto pCommandPool = myvk::CommandPool::Create(pQueue);
	const auto makeBuffer = [&pDevice, &pCommandPool, bufferUsage](const auto &data) {
		auto pStagingBuffer = myvk::Buffer::CreateStaging(pDevice, data.begin(), data.end());
		auto pBuffer =
		    myvk::Buffer::Create(pDevice, pStagingBuffer->GetSize(), 0, VK_BUFFER_USAGE_TRANSFER_DST_BIT | bufferUsage);

		auto pCommandBuffer = myvk::CommandBuffer::Create(pCommandPool);
		pCommandBuffer->Begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
		pCommandBuffer->CmdCopy(pStagingBuffer, pBuffer, {VkBufferCopy{.size = pStagingBuffer->GetSize()}});
		pCommandBuffer->End();

		auto pFence = myvk::Fence::Create(pDevice);
		pCommandBuffer->Submit(pFence);
		pFence->Wait();

		return pBuffer;
	};

	VkGSModel vkModel{.splatCount = model.splatCount};
	vkModel.pMeanBuffer = makeBuffer(model.means);
	vkModel.pScaleBuffer = makeBuffer(model.scales);
	vkModel.pRotateBuffer = makeBuffer(model.rotates);
	vkModel.pOpacityBuffer = makeBuffer(model.opacities);
	vkModel.pSHBuffer = makeBuffer(model.shs);

	return vkModel;
}
