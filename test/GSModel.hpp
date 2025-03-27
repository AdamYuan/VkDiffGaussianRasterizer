//
// Created by adamyuan on 3/27/25.
//

#pragma once
#ifndef GSMODEL_HPP
#define GSMODEL_HPP

#include <cinttypes>
#include <filesystem>
#include <myvk/BufferBase.hpp>
#include <vector>

struct GSModel {
	static constexpr uint32_t kSHDegree = 3;
	static constexpr uint32_t kSHSize = (kSHDegree + 1) * (kSHDegree + 1);

	uint32_t splatCount{};
	std::vector<std::array<float, 3>> means;
	std::vector<std::array<float, 3>> scales;
	std::vector<std::array<float, 4>> rotates;
	std::vector<float> opacities;
	std::vector<std::array<std::array<float, 3>, kSHSize>> shs;

	static GSModel Load(const std::filesystem::path &filename);
	bool IsEmpty() const { return splatCount == 0; }
};

struct VkGSModel {
	uint32_t splatCount{};
	myvk::Ptr<myvk::BufferBase> pMeanBuffer;    // P * [float3]
	myvk::Ptr<myvk::BufferBase> pScaleBuffer;   // P * [float3]
	myvk::Ptr<myvk::BufferBase> pRotateBuffer;  // P * [float4]
	myvk::Ptr<myvk::BufferBase> pOpacityBuffer; // P * [float]
	myvk::Ptr<myvk::BufferBase> pSHBuffer;      // P * [M * float3]

	static VkGSModel Create(const myvk::Ptr<myvk::Queue> &pQueue, VkBufferUsageFlags bufferUsage, const GSModel &model);
	bool IsEmpty() const { return splatCount == 0; }
};

#endif // GSMODEL_HPP
