//
// Created by adamyuan on 3/27/25.
//

#pragma once
#ifndef GSMODEL_HPP
#define GSMODEL_HPP

#include <cinttypes>
#include <concepts>
#include <filesystem>
#include <myvk/BufferBase.hpp>
#include <vector>

#include "../src/Rasterizer.hpp"

struct GSModel {
	static constexpr uint32_t kSHDegree = 3;
	static constexpr uint32_t kSHSize = (kSHDegree + 1) * (kSHDegree + 1);

	using Mean = std::array<float, 3>;
	using Scale = std::array<float, 3>;
	using Rotate = std::array<float, 4>;
	using Opacity = float;
	using SH = std::array<std::array<float, 3>, kSHSize>;

	uint32_t splatCount{};
	std::vector<Mean> means;
	std::vector<Scale> scales;
	std::vector<Rotate> rotates;
	std::vector<Opacity> opacities;
	std::vector<SH> shs;

	static GSModel Load(const std::filesystem::path &filename);
	static uint32_t LoadSplatCount(const std::filesystem::path &filename);
	bool IsEmpty() const { return splatCount == 0; }
};

struct VkGSModel {
	uint32_t splatCount{};
	myvk::Ptr<myvk::BufferBase> pMeanBuffer;    // P * [float3]
	myvk::Ptr<myvk::BufferBase> pScaleBuffer;   // P * [float3]
	myvk::Ptr<myvk::BufferBase> pRotateBuffer;  // P * [float4]
	myvk::Ptr<myvk::BufferBase> pOpacityBuffer; // P * [float]
	myvk::Ptr<myvk::BufferBase> pSHBuffer;      // P * [M * float3]

	void CopyFrom(const myvk::Ptr<myvk::Queue> &pQueue, const GSModel &model);
	static VkGSModel Create(const myvk::Ptr<myvk::Device> &pDevice, VkBufferUsageFlags bufferUsage, uint32_t splatCount,
	                        std::invocable<VkDeviceSize, VkBufferUsageFlags> auto &&createBufferFunc) {
		bufferUsage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VkGSModel vkModel{.splatCount = splatCount};
		vkModel.pMeanBuffer = createBufferFunc(splatCount * sizeof(GSModel::Mean), bufferUsage);
		vkModel.pScaleBuffer = createBufferFunc(splatCount * sizeof(GSModel::Scale), bufferUsage);
		vkModel.pRotateBuffer = createBufferFunc(splatCount * sizeof(GSModel::Rotate), bufferUsage);
		vkModel.pOpacityBuffer = createBufferFunc(splatCount * sizeof(GSModel::Opacity), bufferUsage);
		vkModel.pSHBuffer = createBufferFunc(splatCount * sizeof(GSModel::SH), bufferUsage);
		return vkModel;
	}
	static VkGSModel Create(const myvk::Ptr<myvk::Queue> &pQueue, VkBufferUsageFlags bufferUsage, const GSModel &model,
	                        std::invocable<VkDeviceSize, VkBufferUsageFlags> auto &&createBufferFunc) {
		if (model.IsEmpty())
			return {};

		VkGSModel vkModel = Create(pQueue->GetDevicePtr(), bufferUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		                           model.splatCount, createBufferFunc);
		vkModel.CopyFrom(pQueue, model);

		return vkModel;
	}
	static VkGSModel Create(const myvk::Ptr<myvk::Queue> &pQueue, VkBufferUsageFlags bufferUsage, const GSModel &model);
	static VkGSModel Create(const myvk::Ptr<myvk::Device> &pDevice, VkBufferUsageFlags bufferUsage,
	                        uint32_t splatCount);
	bool IsEmpty() const { return splatCount == 0; }
	vkgsraster::Rasterizer::SplatArgs GetSplatArgs() const;
};

#endif // GSMODEL_HPP
