//
// Created by adamyuan on 3/17/25.
//

#pragma once

#include <cinttypes>
#include <myvk/Device.hpp>
#include <myvk/Queue.hpp>

namespace VkGSRaster {

class Rasterizer {
private:
	myvk::Ptr<myvk::Device> mpDevice;
	myvk::Ptr<myvk::Queue> mpQueue;

public:
	explicit Rasterizer(const myvk::Ptr<myvk::Queue> &pQueue) : mpDevice{pQueue->GetDevicePtr()}, mpQueue{pQueue} {}

	explicit Rasterizer(uint32_t physicalDeviceID = -1);
	// TODO: Device UUID Initializer

	const myvk::Ptr<myvk::Device> &GetDevice() const { return mpDevice; }

	const myvk::Ptr<myvk::Queue> &GetQueue() const { return mpQueue; }
};

} // namespace VkGSRaster