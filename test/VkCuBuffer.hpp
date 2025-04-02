//
// Created by adamyuan on 4/1/25.
//

#pragma once
#ifndef VKCUBUFFER_HPP
#define VKCUBUFFER_HPP

#include <myvk/BufferBase.hpp>

class VkCuBuffer final : public myvk::BufferBase {
private:
	myvk::Ptr<myvk::Device> mpDevice;

	VkDeviceMemory mDeviceMemory{VK_NULL_HANDLE};
	std::uintptr_t mCudaExtMemory{};
	void *mpCudaMapped{nullptr};

public:
	static myvk::Ptr<VkCuBuffer> Create(const myvk::Ptr<myvk::Device> &pDevice, VkDeviceSize size,
	                                    VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties,
	                                    const std::vector<myvk::Ptr<myvk::Queue>> &pAccessQueues = {});
	~VkCuBuffer() override;
	const myvk::Ptr<myvk::Device> &GetDevicePtr() const override { return mpDevice; }

	void *GetCudaMappedPtr() const { return mpCudaMapped; }
	template <typename T> T *GetCudaMappedPtr() const { return reinterpret_cast<T *>(mpCudaMapped); }
};

#endif // VKCUBUFFER_HPP
