//
// Created by adamyuan on 3/21/25.
//

#pragma once
#ifndef VKGSRASTER_CONTEXT_HPP
#define VKGSRASTER_CONTEXT_HPP

#include <myvk/Device.hpp>
#include <myvk/Queue.hpp>

namespace vkgsraster {

class Context {
private:
	myvk::Ptr<myvk::Device> mpDevice;
	myvk::Ptr<myvk::Queue> mpQueue;

public:
	explicit Context(const myvk::Ptr<myvk::Queue> &pQueue) : mpDevice{pQueue->GetDevicePtr()}, mpQueue{pQueue} {}
	explicit Context(uint32_t physicalDeviceID = -1);
	// TODO: Device UUID Initializer

	const myvk::Ptr<myvk::Device> &GetDevice() const { return mpDevice; }
	const myvk::Ptr<myvk::Queue> &GetQueue() const { return mpQueue; }
};

} // namespace vkgsraster

#endif
